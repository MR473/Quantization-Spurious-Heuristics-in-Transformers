
"""Common code for the standalone DLBA quantization runtime experiment.

The only repository-external inputs are the project's EXISTING fine-tuned FP32
checkpoints and HANS prediction CSVs. This module never modifies those files.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
CONFIG_FILE = ANALYSIS_DIR / "config.json"

LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

# Original four variants are preserved.
# The new variant excludes the final 3-class classifier from INT8.
VARIANTS = (
    "fp32_eager",
    "int8_eager",
    "fp32_compiled",
    "int8_compiled",
    "int8_compiled_fp32_head",
)


def is_compiled_variant(variant):
    return "compiled" in variant


def config_read():
    with CONFIG_FILE.open(encoding="utf-8") as f:
        cfg = json.load(f)

    cfg["seeds"] = [int(s) for s in cfg["seeds"]]

    assert cfg["throughput_batch_size"] in cfg["batch_sizes"]
    assert cfg["benchmark_repeats"] >= 1

    return cfg


def project_path(relative_or_absolute):
    path = Path(relative_or_absolute).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def model_path(cfg, seed):
    return project_path(
        cfg["model_dir_template"].format(seed=seed)
    )


def reference_csv(cfg, seed, name):
    key = (
        "hans_predictions_template"
        if name == "fp32"
        else "reference_int8_predictions_template"
    )
    return project_path(cfg[key].format(seed=seed))


def output_dir(cfg, seed, variant):
    path = (
        project_path(cfg["results_dir"])
        / f"seed{seed}"
        / variant
    )

    path.mkdir(parents=True, exist_ok=True)
    return path


def synchronize(device):
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def current_cuda_allocated_mib(device):
    import torch

    if device.type == "cuda":
        return (
            torch.cuda.memory_allocated(device)
            / (1024 ** 2)
        )

    return float("nan")


def current_cuda_reserved_mib(device):
    import torch

    if device.type == "cuda":
        return (
            torch.cuda.memory_reserved(device)
            / (1024 ** 2)
        )

    return float("nan")


def rss_mib():
    import psutil

    return (
        psutil.Process(os.getpid()).memory_info().rss
        / (1024 ** 2)
    )


def serialized_state_size_mib(model):
    import torch

    with io.BytesIO() as buffer:
        torch.save(model.state_dict(), buffer)
        return buffer.tell() / (1024 ** 2)


def input_examples(cfg, seed):
    """Prefer locally saved HANS with pairID; fallback to official TSV.

    No load_dataset('hans') call; that older script is unsupported.
    """
    saved = reference_csv(cfg, seed, "fp32")

    if saved.is_file():
        data = pd.read_csv(saved)
        source = str(saved)

    else:
        try:
            data = pd.read_csv(
                cfg["hans_tsv_url"],
                sep="\t",
            )

        except Exception as e:
            raise FileNotFoundError(
                f"Missing {saved}. "
                f"Network TSV fallback also failed: {e}. "
                "Provide the HANS FP32 prediction CSV in "
                "the expected results/hans/ folder, "
                "or ensure you can access the official HANS TSV URL."
            ) from e

        data = data.rename(
            columns={
                "sentence1": "premise",
                "sentence2": "hypothesis",
            }
        )

        source = cfg["hans_tsv_url"]

    if (
        "sentence1" in data.columns
        and "premise" not in data.columns
    ):
        data = data.rename(
            columns={
                "sentence1": "premise",
                "sentence2": "hypothesis",
            }
        )

    required = {"pairID", "premise", "hypothesis"}

    if not required <= set(data.columns):
        missing = sorted(required - set(data.columns))
        raise ValueError(
            f"HANS input missing columns {missing}; "
            f"source: {source}"
        )

    if (
        data["pairID"].isna().any()
        or data["pairID"].astype(str).duplicated().any()
    ):
        raise ValueError(
            "HANS pairID contains blanks or duplicates; "
            "cannot validate predictions correctly."
        )

    if data[["premise", "hypothesis"]].isna().any().any():
        raise ValueError(
            "Missing HANS premise or hypothesis detected."
        )

    maximum = cfg.get("max_examples")

    if maximum is not None:
        data = data.iloc[:int(maximum)].copy()

    data = data.reset_index(drop=True)

    if maximum is None and len(data) != 30000:
        raise ValueError(
            f"Expected 30,000 HANS examples, "
            f"found {len(data)} in {source}."
        )

    return data, source


def tokenize(cfg, df, tokenizer):
    """Tokenize once, independently of timed forward passes."""

    return tokenizer(
        df["premise"].tolist(),
        df["hypothesis"].tolist(),
        truncation=True,
        max_length=int(cfg["max_length"]),
        padding=False,
    )


class TokenizedRows:
    def __init__(self, encoded):
        self.encoded = encoded
        self.size = len(encoded["input_ids"])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            k: v[idx]
            for k, v in self.encoded.items()
        }


def loader(cfg, tokenized, tokenizer, batch_size):
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    return DataLoader(
        TokenizedRows(tokenized),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )


def quantize_supported_layers(module, name):
    """Quantize Linear layers except the final MNLI classifier.

    The BERT classifier has three outputs. The selected compiled
    integer matmul kernel requires an output dimension divisible by 8.
    Keeping this small layer in FP32 avoids that specific mismatch.
    """
    import torch

    return (
        isinstance(module, torch.nn.Linear)
        and name.split(".")[-1] != "classifier"
    )


def load_model(cfg, seed, variant, device):
    """Load and optionally quantize/compile a model.

    IMPORTANT:
    - int8_eager uses the ORIGINAL full dynamic W8A8 configuration.
    - int8_compiled also uses the ORIGINAL full dynamic W8A8 configuration.
    - int8_compiled_fp32_head excludes only the final classifier.

    No fine-tuning or retraining is performed.
    """
    import torch

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant}. "
            f"Valid variants: {VARIANTS}"
        )

    path = model_path(cfg, seed)

    if not path.is_dir():
        raise FileNotFoundError(
            f"FP32 checkpoint not found: {path}. "
            "Update model_dir_template in config.json."
        )

    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForSequenceClassification.from_pretrained(
        path
    ).eval()

    quantization_seconds = 0.0

    if variant.startswith("int8"):
        from torchao.quantization import (
            quantize_,
            Int8DynamicActivationInt8WeightConfig,
        )

        t0 = time.perf_counter()

        if variant == "int8_compiled_fp32_head":
            # NEW VARIANT:
            # Quantize BERT's supported Linear layers but
            # preserve the 3-output classifier in FP32.
            quantize_(
                model,
                Int8DynamicActivationInt8WeightConfig(),
                filter_fn=quantize_supported_layers,
            )

            # Verify the final classifier remains in FP32.
            classifier = model.classifier

            assert isinstance(
                classifier, torch.nn.Linear
            ), (
                "Expected the final classifier to remain "
                "a torch.nn.Linear module."
            )

            assert classifier.out_features == 3, (
                "Expected the original 3-class MNLI head."
            )

            assert classifier.weight.dtype == torch.float32, (
                "Expected FP32 classifier weights."
            )

        else:
            # ORIGINAL QUANTIZATION:
            # Preserve exactly the behavior used by your
            # previous HANS FP32-vs-INT8 experiments.
            quantize_(
                model,
                Int8DynamicActivationInt8WeightConfig(),
            )

        quantization_seconds = time.perf_counter() - t0

    # Estimate storage footprint before moving to CUDA.
    state_mib = serialized_state_size_mib(model)

    model = model.to(device).eval()

    # Recognizes both int8_compiled and
    # int8_compiled_fp32_head.
    if is_compiled_variant(variant):
        model = torch.compile(
            model,
            mode=cfg["compile_mode"],
            dynamic=bool(cfg["compile_dynamic"]),
        )

    return (
        model,
        tokenizer,
        quantization_seconds,
        state_mib,
    )


def device_batch(batch, device):
    return {
        k: v.to(device, non_blocking=False)
        for k, v in batch.items()
    }


def pack_environment(cfg, seed, variant, source, device):
    import torch
    import torchao

    compiled = is_compiled_variant(variant)

    if variant == "int8_compiled_fp32_head":
        method = (
            "Dynamic W8A8 TorchAO PTQ; "
            "3-class classifier retained in FP32"
        )
    elif variant.startswith("int8"):
        method = "Dynamic W8A8 TorchAO PTQ"
    else:
        method = "FP32"

    return {
        "seed": seed,
        "variant": variant,
        "device": str(device),
        "torch": torch.__version__,
        "torchao": torchao.__version__,
        "cuda": torch.version.cuda,

        "gpu_name": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda"
            else "N/A"
        ),

        "model_path": str(model_path(cfg, seed)),
        "hans_source": source,
        "max_length": cfg["max_length"],

        "throughput_batch_size":
            cfg["throughput_batch_size"],

        "benchmark_repeats":
            cfg["benchmark_repeats"],

        "compile_mode": (
            cfg["compile_mode"]
            if compiled else None
        ),

        "compile_dynamic": (
            cfg["compile_dynamic"]
            if compiled else None
        ),

        "method": method,
    }


def write_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )
