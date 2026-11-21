#!/usr/bin/env python3
"""Experiment 3: PyTorch / TorchAO / OpenVINO CPU inference on the same HANS samples.

Four independent process runs per seed:
  torch_fp32, torchao_weight_only, openvino_fp32, openvino_int8_sym.

Run from repository root after `uv add openvino nncf`:
  uv run python scripts/experiment3_openvino_cpu.py --seed 42 --num-examples 512 --repeats 2

Saved OpenVINO IR files live under models/experiment3_openvino/ (do NOT commit).
All metrics/plots go to results/runtime_analysis/experiment3_openvino/.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("torch_fp32", "torchao_weight_only", "openvino_fp32", "openvino_int8_sym")
MODEL_FILE = {"openvino_fp32": "fp32.xml", "openvino_int8_sym": "int8_sym.xml"}


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, choices=[42, 123, 456], default=42)
    ap.add_argument("--num-examples", type=int, default=512, help="First N HANS examples, max 30000")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--cache-dir", type=Path, default=ROOT / "models/experiment3_openvino")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/runtime_analysis/experiment3_openvino")
    ap.add_argument("--rebuild", action="store_true", help="Regenerate cached OpenVINO models")
    ap.add_argument("--_role", choices=["driver", "prepare", "worker"], default="driver", help=argparse.SUPPRESS)
    ap.add_argument("--_variant", choices=VARIANTS, help=argparse.SUPPRESS)
    ap.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    return ap


def cpu_rss_mib():
    """Linux current resident set size (includes Python/framework overhead)."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return None


def checkpoint(seed):
    ckpt = ROOT / f"models/fp32/seed{seed}/final"
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing FP32 checkpoint: {ckpt}")
    return ckpt


def load_samples(seed, n):
    import pandas as pd
    path = ROOT / f"results/hans/fp32_seed{seed}_predictions.csv"
    df = pd.read_csv(path, nrows=n)
    required = {"pairID", "premise", "hypothesis", "gold_label", "mnli_prediction_id"}
    if required - set(df.columns):
        raise ValueError(f"HANS reference missing columns: {required - set(df.columns)}")
    if len(df) != n:
        raise ValueError(f"Requested {n} examples but found only {len(df)}")
    if len(df.pairID.unique()) != n:
        raise ValueError("Duplicate pairID; cannot verify evaluation alignment")
    return df


def batches_for(seed, n, batch_size):
    import torch
    from transformers import AutoTokenizer, DataCollatorWithPadding
    samples = load_samples(seed, n)
    tok = AutoTokenizer.from_pretrained(str(checkpoint(seed)), local_files_only=True)
    tokens = tok(samples.premise.tolist(), samples.hypothesis.tolist(),
                 truncation=True, max_length=128, padding=False)
    keys = [x for x in ("input_ids", "attention_mask", "token_type_ids") if x in tokens]
    if len(keys) != 3:
        raise ValueError("Expected BERT to generate input_ids, attention_mask, token_type_ids")
    collate = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")
    bs = []
    for start in range(0, n, batch_size):
        features = [{key: tokens[key][i] for key in keys}
                    for i in range(start, min(n, start + batch_size))]
        b = collate(features)
        bs.append({key: b[key].contiguous() for key in keys})
    assert sum(b["input_ids"].shape[0] for b in bs) == n
    return samples, bs


def setup_torch(variant, seed, threads):
    import torch
    from transformers import AutoModelForSequenceClassification
    torch.set_num_threads(threads)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint(seed)), local_files_only=True).cpu().eval()
    if variant == "torchao_weight_only":
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        quantize_(model, Int8WeightOnlyConfig())
        count = sum(type(layer.weight).__name__ == "Int8Tensor"
                    for layer in model.modules() if isinstance(layer, torch.nn.Linear))
        if count != 74:
            raise AssertionError(f"Expected 74 quantized linear layers; observed {count}")
    return model


def prepare_ov(args):
    """Separate preparation process: conversion does not contaminate benchmark RSS."""
    import torch
    import openvino as ov
    import nncf
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dest = args.cache_dir / f"seed{args.seed}"
    dest.mkdir(parents=True, exist_ok=True)
    # Keep all exported binaries outside git, including files from future runs.
    (args.cache_dir / ".gitignore").write_text("*\n")
    fp = dest / "fp32.xml"
    iq = dest / "int8_sym.xml"
    source_signature = {
        path.name: [path.stat().st_size, path.stat().st_mtime_ns]
        for path in checkpoint(args.seed).iterdir()
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin")
    }
    metadata_path = dest / "model_export_metadata.json"
    cached_ok = all(path.exists() for path in (
        fp, fp.with_suffix(".bin"), iq, iq.with_suffix(".bin"), metadata_path))
    if cached_ok and not args.rebuild:
        old = json.loads(metadata_path.read_text())
        if old.get("checkpoint_signature") == source_signature:
            print("Reusing cached OpenVINO models:", dest, flush=True)
            return
        print("Checkpoint changed since export; rebuilding OpenVINO cache.", flush=True)

    torch.set_num_threads(args.threads)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint(args.seed)), local_files_only=True).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint(args.seed)), local_files_only=True)
    sample = tokenizer(["The cat is sleeping.", "The boy is running."],
                       ["An animal is asleep.", "A child moves."],
                       padding="max_length", max_length=128,
                       truncation=True, return_tensors="pt")

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.wrapped(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=False)[0]

    wrapper = LogitsOnly(model).eval()
    ex = (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"])
    print("Converting BERT to OpenVINO (dynamic batch and sequence)...", flush=True)
    with torch.inference_mode():
        try:
            ov_model = ov.convert_model(wrapper, example_input=ex,
                                        input=[ov.PartialShape([-1, -1]) for _ in ex])
        except Exception as err:
            raise RuntimeError(
                "OpenVINO export failed. Share the traceback; PyTorch/HF/OpenVINO "
                "version compatibility or dynamic shapes may need adjustment."
            ) from err
    if len(ov_model.inputs) != 3:
        raise RuntimeError(f"Expected 3 OpenVINO inputs; got {len(ov_model.inputs)}")
    # Map OpenVINO's source-order inputs to the same order as the wrapper signature.
    for port, name in zip(ov_model.inputs, ("input_ids", "attention_mask", "token_type_ids")):
        port.get_tensor().set_names({name})
    ov.save_model(ov_model, str(fp), compress_to_fp16=False)
    print("Saved uncompressed FP32 OpenVINO IR:", fp, flush=True)

    print("Compressing OpenVINO weights using NNCF INT8_SYM...", flush=True)
    compressed = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)

    # OpenVINO returns ov.Type objects, NOT strings like "i8" or "u8".
    # Comparing str(ov.Type.i8) to "i8" gives a false negative on some versions.
    def inspect_int8_constants(model_to_inspect):
        from collections import Counter
        constant_nodes = [node for node in model_to_inspect.get_ordered_ops()
                          if node.get_type_name() == "Constant"]
        dtype_counts = Counter(node.get_output_element_type(0).to_string()
                               for node in constant_nodes)
        int8_count = sum(node.get_output_element_type(0) in (ov.Type.i8, ov.Type.u8)
                         for node in constant_nodes)
        return int8_count, dict(sorted(dtype_counts.items()))

    compressed_int8_constants, constant_types = inspect_int8_constants(compressed)
    print(f"Compressed graph Constant element types: {constant_types}", flush=True)
    if compressed_int8_constants < 1:
        raise RuntimeError(
            "No INT8 Constant nodes found after NNCF compression. "
            f"Observed element types: {constant_types}"
        )

    ov.save_model(compressed, str(iq), compress_to_fp16=False)
    # Validate what was actually serialized, not just the in-memory graph.
    reloaded_int8_constants, reloaded_types = inspect_int8_constants(
        ov.Core().read_model(str(iq))
    )
    if reloaded_int8_constants < 1:
        raise RuntimeError(
            "INT8 constants were not found after saving/reloading OpenVINO IR. "
            f"Reloaded element types: {reloaded_types}"
        )

    fp32_artifact_bytes = fp.stat().st_size + fp.with_suffix(".bin").stat().st_size
    int8_artifact_bytes = iq.stat().st_size + iq.with_suffix(".bin").stat().st_size
    size_reduction_pct = 100 * (1 - int8_artifact_bytes / fp32_artifact_bytes)
    print(f"Saved compressed OpenVINO IR: {iq}", flush=True)
    print(f"  INT8 Constant nodes: {reloaded_int8_constants}", flush=True)
    print(f"  FP32 IR size: {fp32_artifact_bytes / 2**20:.2f} MiB", flush=True)
    print(f"  INT8 IR size: {int8_artifact_bytes / 2**20:.2f} MiB", flush=True)
    print(f"  IR size reduction: {size_reduction_pct:.2f}%", flush=True)

    meta = {"seed": args.seed, "precision": "NNCF INT8_SYM weight compression",
            "int8_constant_nodes": reloaded_int8_constants,
            "constant_element_types": reloaded_types,
            "fp32_ir_bytes": fp32_artifact_bytes,
            "int8_ir_bytes": int8_artifact_bytes,
            "ir_size_reduction_pct": size_reduction_pct,
            "checkpoint_signature": source_signature,
            "openvino_version": ov.__version__, "nncf_version": nncf.__version__,
            "note": "Different quantizer/layer selection than TorchAO; not a precision-equivalent replacement."}
    metadata_path.write_text(json.dumps(meta, indent=2))


def infer_worker(args):
    """Measure one variant per fresh process. Timings exclude setup and export."""
    import torch
    from transformers import AutoTokenizer  # noqa: F401 (dependency readiness)
    torch.set_num_threads(args.threads)
    samples, batches = batches_for(args.seed, args.num_examples, args.batch_size)
    print(f"[{args._variant}] Prepared {len(batches)} matching HANS batches", flush=True)
    rss_before = cpu_rss_mib()

    if args._variant.startswith("torch"):
        model = setup_torch(args._variant, args.seed, args.threads)
        def forward(batch):
            with torch.inference_mode():
                return model(**batch).logits.detach().cpu().numpy()
        size_mib = None  # May require PyTorch subclass-specific serialization.
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            torch.save(model.state_dict(), Path(tmp) / "model_state.pt")
            size_mib = (Path(tmp) / "model_state.pt").stat().st_size / (1024 ** 2)
    else:
        import openvino as ov
        core = ov.Core()
        file = args.cache_dir / f"seed{args.seed}" / MODEL_FILE[args._variant]
        if not file.exists():
            raise FileNotFoundError(f"Missing converted OpenVINO model: {file}")
        # Force FP32 inference preference, a single stream and a fixed thread budget.
        config = {"INFERENCE_NUM_THREADS": str(args.threads),
                  "NUM_STREAMS": "1", "INFERENCE_PRECISION_HINT": "f32"}
        compiled = core.compile_model(str(file), "CPU", config)
        request = compiled.create_infer_request()
        input_names = ("input_ids", "attention_mask", "token_type_ids")
        def forward(batch):
            inputs = {name: batch[name].numpy() for name in input_names}
            return np.asarray(next(iter(request.infer(inputs).values())))
        size_mib = sum(file.with_suffix(ext).stat().st_size
                       for ext in (".xml", ".bin")) / (1024 ** 2)
    gc.collect()
    # Reduce distortion from allocator-retained temporary loading/serialization buffers.
    if sys.platform.startswith("linux"):
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
    rss_ready = cpu_rss_mib()

    print(f"[{args._variant}] Warm-up {args.warmup} batches", flush=True)
    for i in range(args.warmup):
        arr = forward(batches[i % len(batches)])
        if not np.isfinite(arr).all():
            raise RuntimeError("Nonfinite logits during warm-up")

    all_pass_seconds = []
    first_pass_preds = None
    batch_seconds = []
    for repeat in range(args.repeats):
        preds = []
        start = time.perf_counter()
        for b in batches:
            t0 = time.perf_counter()
            logits = forward(b)
            batch_seconds.append(time.perf_counter() - t0)
            if logits.shape != (b["input_ids"].shape[0], 3):
                raise ValueError(f"Unexpected logits shape: {logits.shape}")
            if repeat == 0:
                preds.extend(np.asarray(logits).argmax(axis=-1).tolist())
        duration = time.perf_counter() - start
        all_pass_seconds.append(duration)
        if repeat == 0:
            first_pass_preds = np.asarray(preds, dtype=np.int64)
        print(f"[{args._variant}] pass {repeat+1}/{args.repeats}: {duration:.3f}s", flush=True)
    rss_after = cpu_rss_mib()

    assert first_pass_preds is not None and len(first_pass_preds) == args.num_examples
    binary = np.where(first_pass_preds == 0, "entailment", "non-entailment")
    hans_accuracy = float(np.mean(binary == samples.gold_label.to_numpy()))
    orig_fp32 = samples.mnli_prediction_id.to_numpy(dtype=np.int64)
    agreement = float(np.mean(first_pass_preds == orig_fp32))
    result = {
        "seed": args.seed, "variant": args._variant,
        "num_examples": args.num_examples, "threads": args.threads,
        "batch_size": args.batch_size, "repeats": args.repeats,
        "model_artifact_mib": size_mib,
        "rss_before_load_mib": rss_before, "rss_ready_mib": rss_ready,
        "rss_after_inference_mib": rss_after,
        "rss_setup_measurement": "After gc and glibc malloc_trim on Linux",
        "mean_batch_forward_ms": statistics.mean(batch_seconds) * 1000,
        "median_batch_forward_ms": statistics.median(batch_seconds) * 1000,
        "median_full_pass_seconds": statistics.median(all_pass_seconds),
        "throughput_examples_per_second": args.num_examples / statistics.median(all_pass_seconds),
        "hans_accuracy": hans_accuracy,
        "three_class_agreement_with_saved_fp32": agreement,
        "full_pass_seconds": all_pass_seconds,
        "platform": platform.platform(),
        "synchronous_single_stream": True,
        "precision_note": ("NNCF weight-only compression, not equivalent to TorchAO's "
                          "74-layer quantization" if args._variant == "openvino_int8_sym" else ""),
    }
    args._output.parent.mkdir(parents=True, exist_ok=True)
    args._output.write_text(json.dumps(result, indent=2))
    print(f"[{args._variant}] accuracy={hans_accuracy:.4%} "
          f"throughput={result['throughput_examples_per_second']:.2f} examples/s", flush=True)


def write_report(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    baseline = float(table.loc[table.variant == "torch_fp32", "throughput_examples_per_second"].iloc[0])
    table["speedup_vs_torch_fp32"] = table["throughput_examples_per_second"] / baseline
    table["hans_accuracy_pct"] = 100 * table["hans_accuracy"]
    table.to_csv(out / "benchmark_summary.csv", index=False)
    metrics = ("variant", "throughput_examples_per_second", "speedup_vs_torch_fp32",
               "model_artifact_mib", "hans_accuracy_pct", "three_class_agreement_with_saved_fp32")
    print("\n===== EXPERIMENT 3 COMPARISON =====", flush=True)
    print(table[list(metrics)].round(3).to_string(index=False), flush=True)

    labels = ["PyTorch\nFP32", "TorchAO\nweight-only", "OpenVINO\nFP32", "OpenVINO\nINT8-SYM"]
    plots = [
        ("01_cpu_throughput", "throughput_examples_per_second", "HANS throughput (examples / second)", "Higher is better"),
        ("02_hans_accuracy", "hans_accuracy_pct", "HANS accuracy (%)", "Compare accuracy before calling a model faster"),
        ("03_artifact_size", "model_artifact_mib", "Saved model artifact size (MiB)", "Formats differ: compare sizes within backends"),
        ("04_ready_process_rss", "rss_ready_mib", "Resident process memory (MiB)", "RSS includes the complete Python and framework process"),
    ]
    for filename, col, ylabel, subtitle in plots:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        y = table[col].to_numpy(dtype=float)
        ax.bar(range(len(y)), y, width=0.65)
        ax.set_xticks(range(len(y)), labels)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ymax = max(float(np.max(y)) * 1.15, 1.0)
        ax.set_ylim(0, ymax)
        for idx, val in enumerate(y):
            ax.text(idx, val + ymax * 0.02, f"{val:.1f}", ha="center", fontsize=9)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(out / f"{filename}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    metadata = {
        "experiment": "3 OpenVINO backend vs TorchAO weight-only",
        "seeds": sorted(table.seed.unique().tolist()),
        "hardware": platform.platform(),
        "inference_mode": "single-request synchronous, one stream per variant",
        "limitations": [
            "OpenVINO's NNCF INT8_SYM compression is not identical to TorchAO weight-only INT8.",
            "OpenVINO's compiled inference uses a fixed CPU thread count and FP32 hint; "
            "compiler and kernel optimizations differ from PyTorch.",
            "PyTorch .pt and OpenVINO IR artifacts have different serialization formats.",
            "RSS is process-wide, not weights-only or peak inference memory.",
            "Each model uses identical HANS inputs and batch size; tokenizer, loading, conversion, "
            "and compilation are excluded from throughput timing.",
            "One seed and small subsets are exploratory; verify full 30000 examples across all seeds.",
        ],
    }
    (out / "experiment3_notes.json").write_text(json.dumps(metadata, indent=2))
    print("Saved reports and plots under", out)


def main(args):
    if not 1 <= args.num_examples <= 30000:
        raise ValueError("--num-examples must be 1..30000")
    if args.threads < 1 or args.batch_size < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("threads, batch-size, repeats >= 1 and warmup >= 0")
    checkpoint(args.seed)
    script = Path(__file__).resolve()
    output = args.out_dir / f"seed{args.seed}_t{args.threads}_b{args.batch_size}_n{args.num_examples}"
    output.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(script), "--seed", str(args.seed), "--threads", str(args.threads),
            "--batch-size", str(args.batch_size), "--num-examples", str(args.num_examples),
            "--repeats", str(args.repeats), "--warmup", str(args.warmup),
            "--cache-dir", str(args.cache_dir), "--out-dir", str(args.out_dir)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    if args.rebuild:
        subprocess.run(base + ["--_role", "prepare", "--rebuild"], check=True, cwd=ROOT, env=env)
    else:
        subprocess.run(base + ["--_role", "prepare"], check=True, cwd=ROOT, env=env)
    results = []
    for name in VARIANTS:
        file = output / f"{name}.json"
        print(f"\n=== Seed {args.seed}: {name} ===", flush=True)
        cmd = base + ["--_role", "worker", "--_variant", name, "--_output", str(file)]
        subprocess.run(cmd, check=True, cwd=ROOT, env=env)
        results.append(json.loads(file.read_text()))
    write_report(results, output)


if __name__ == "__main__":
    args = parser().parse_args()
    if args._role == "prepare":
        prepare_ov(args)
    elif args._role == "worker":
        infer_worker(args)
    else:
        main(args)
