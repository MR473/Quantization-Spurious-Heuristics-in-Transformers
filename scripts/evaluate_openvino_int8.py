#!/usr/bin/env python3
"""Experiment 3C: save per-example HANS predictions for OpenVINO NNCF INT8.

Reads the same FP32 HANS reference CSV, fine-tuned tokenizer, and cached
OpenVINO INT8-SYM model used by experiment3_openvino_cpu.py. Does NOT retrain,
reconvert, requantize, or modify any existing FP32/TorchAO results.

From the repository root:
    uv run python scripts/evaluate_openvino_int8.py --seed 42
    uv run python scripts/evaluate_openvino_int8.py --seed 123
    uv run python scripts/evaluate_openvino_int8.py --seed 456

Outputs (full 30k example runs):
    results/hans/openvino_int8/openvino_int8_seed{seed}_predictions.csv
    results/hans/openvino_int8/openvino_int8_seed{seed}_summary.json

For a smaller smoke test, use --num-examples 512; it writes a different
filename (seed42_n512) rather than overwriting the full-dataset result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}
EXPECTED_COLUMNS = {
    "gold_label", "premise", "hypothesis", "pairID", "heuristic", "subcase",
    "mnli_prediction_id", "mnli_prediction", "hans_prediction", "correct",
    "prob_entailment", "prob_neutral", "prob_contradiction", "prob_non_entailment",
}
PREDICTION_COLUMNS = [
    "mnli_prediction_id", "mnli_prediction", "hans_prediction", "correct",
    "prob_entailment", "prob_neutral", "prob_contradiction", "prob_non_entailment",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seed", type=int, choices=[42, 123, 456], default=42)
    ap.add_argument("--num-examples", type=int, default=30000, help="First N reference rows; 30000 for full HANS")
    ap.add_argument("--batch-size", type=int, default=32, help="Match Experiment 3: 32")
    ap.add_argument("--threads", type=int, default=8, help="Match Experiment 3: 8")
    ap.add_argument("--model-cache", type=Path, default=ROOT / "models/experiment3_openvino")
    ap.add_argument("--reference-dir", type=Path, default=ROOT / "results/hans")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/hans/openvino_int8")
    ap.add_argument("--overwrite", action="store_true", help="Explicitly replace this seed's existing evaluation files")
    return ap.parse_args()


def file_names(args):
    suffix = f"seed{args.seed}" + ("" if args.num_examples == 30000 else f"_n{args.num_examples}")
    return (
        args.out_dir / f"openvino_int8_{suffix}_predictions.csv",
        args.out_dir / f"openvino_int8_{suffix}_summary.json",
    )


def load_reference(args):
    path = args.reference_dir / f"fp32_seed{args.seed}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing FP32 reference predictions: {path}")
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"FP32 reference is missing required columns: {sorted(missing)}")
    if len(df) != 30000:
        raise ValueError(f"Expected full HANS reference with 30000 rows, got {len(df)}")
    if df["pairID"].isna().any() or not df["pairID"].is_unique:
        raise ValueError("Reference pairID must be unique and nonnull")
    if df[["premise", "hypothesis", "gold_label"]].isna().any().any():
        raise ValueError("Reference contains null premise/hypothesis/gold_label")
    if set(df.gold_label.unique()) != {"entailment", "non-entailment"}:
        raise ValueError("Unexpected gold labels in reference")
    if not df.mnli_prediction_id.isin([0, 1, 2]).all():
        raise ValueError("Reference MNLI prediction IDs are outside [0, 1, 2]")
    for pred_id, label in LABELS.items():
        rows = df.loc[df.mnli_prediction_id == pred_id, "mnli_prediction"]
        if not rows.empty and not rows.eq(label).all():
            raise ValueError(f"Reference label mapping does not match expected id2label {LABELS}")
    if args.num_examples < 1 or args.num_examples > len(df):
        raise ValueError("--num-examples must be between 1 and 30000")
    return df.iloc[:args.num_examples].copy(), path


def assert_matching_export(args):
    """Prevent accidentally using an OpenVINO export from changed FP32 weights."""
    ckpt = ROOT / f"models/fp32/seed{args.seed}/final"
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing fine-tuned checkpoint/tokenizer: {ckpt}")
    cache_dir = args.model_cache / f"seed{args.seed}"
    model_path = cache_dir / "int8_sym.xml"
    metadata_path = cache_dir / "model_export_metadata.json"
    if not model_path.is_file() or not model_path.with_suffix(".bin").is_file():
        raise FileNotFoundError(
            f"OpenVINO INT8 cached model not found: {model_path}. "
            "Generate it with experiment3_openvino_cpu.py first."
        )
    if not metadata_path.is_file():
        raise FileNotFoundError(f"OpenVINO export metadata not found: {metadata_path}")
    meta = json.loads(metadata_path.read_text())
    current_signature = {
        path.name: [path.stat().st_size, path.stat().st_mtime_ns]
        for path in ckpt.iterdir()
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin")
    }
    if meta.get("checkpoint_signature") != current_signature:
        raise ValueError(
            "The FP32 checkpoint differs from the checkpoint used to export OpenVINO. "
            "Rebuild the OpenVINO cache deliberately using Experiment 3 before evaluating."
        )
    if meta.get("int8_constant_nodes", 0) < 1:
        raise ValueError("Cache metadata does not verify INT8 Constant nodes")
    return ckpt, model_path, meta


def make_batches(reference, checkpoint_dir, batch_size):
    """Match Experiment 3 tokenization and dynamic per-batch padding exactly."""
    from transformers import AutoTokenizer, DataCollatorWithPadding

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True)
    tokens = tokenizer(
        reference.premise.tolist(), reference.hypothesis.tolist(),
        truncation=True, max_length=128, padding=False,
    )
    keys = ("input_ids", "attention_mask", "token_type_ids")
    if any(key not in tokens for key in keys):
        raise ValueError(f"Tokenizer must produce all of {keys}")
    collate = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    for start in range(0, len(reference), batch_size):
        end = min(start + batch_size, len(reference))
        features = [{key: tokens[key][i] for key in keys} for i in range(start, end)]
        batch = collate(features)
        yield {key: np.ascontiguousarray(batch[key].numpy()) for key in keys}


def softmax(logits):
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def build_prediction_frame(reference, logits):
    if logits.shape != (len(reference), 3) or not np.isfinite(logits).all():
        raise ValueError(f"Expected finite logits of shape ({len(reference)}, 3); got {logits.shape}")
    p = softmax(logits)
    prediction_id = p.argmax(axis=1).astype(np.int64)
    prediction_label = np.array([LABELS[int(i)] for i in prediction_id])
    hans_label = np.where(prediction_id == 0, "entailment", "non-entailment")
    result = reference.copy()
    result["mnli_prediction_id"] = prediction_id
    result["mnli_prediction"] = prediction_label
    result["hans_prediction"] = hans_label
    result["correct"] = hans_label == reference["gold_label"].to_numpy()
    result["prob_entailment"] = p[:, 0]
    result["prob_neutral"] = p[:, 1]
    result["prob_contradiction"] = p[:, 2]
    result["prob_non_entailment"] = p[:, 1] + p[:, 2]

    # Validate the exact contract consumed by scripts/analyze_quantization_effects.py.
    assert result["pairID"].equals(reference["pairID"]), "pairID alignment changed"
    for field in ("gold_label", "heuristic", "subcase", "premise", "hypothesis"):
        assert result[field].equals(reference[field]), f"Reference field changed: {field}"
    assert result["correct"].dtype == bool
    assert result["correct"].equals(result.hans_prediction.eq(result.gold_label))
    probs = result[["prob_entailment", "prob_neutral", "prob_contradiction"]].to_numpy(dtype=float)
    assert np.isfinite(probs).all() and np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(probs.sum(axis=1), 1, atol=1e-6)
    assert np.allclose(result.prob_non_entailment, probs[:, 1] + probs[:, 2], atol=1e-6)
    return result


def verify_against_existing_benchmark(args, results):
    benchmark_path = (
        ROOT / "results/runtime_analysis/experiment3_openvino"
        / f"seed{args.seed}_t{args.threads}_b{args.batch_size}_n{args.num_examples}"
        / "openvino_int8_sym.json"
    )
    ref_ids = pd.read_csv(
        args.reference_dir / f"fp32_seed{args.seed}_predictions.csv",
        usecols=["mnli_prediction_id"], nrows=args.num_examples,
    ).mnli_prediction_id.to_numpy(dtype=np.int64)
    accuracy = float(results.correct.mean())
    agreement = float(np.mean(results.mnli_prediction_id.to_numpy(dtype=np.int64) == ref_ids))
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text())
        for field, measured, target in (
            ("HANS accuracy", accuracy, float(benchmark["hans_accuracy"])),
            ("FP32 three-class agreement", agreement, float(benchmark["three_class_agreement_with_saved_fp32"])),
        ):
            if not np.isclose(measured, target, rtol=0, atol=1e-10):
                raise RuntimeError(
                    f"Mismatch with Experiment 3 benchmark ({benchmark_path}): "
                    f"{field} newly {measured:.8f}, previously {target:.8f}. "
                    "Check input alignment, model cache, tokenizer, and CPU settings. "
                    "No predictions have been saved."
                )
        print(f"Validated accuracy and agreement against existing benchmark: {benchmark_path}")
    else:
        print("Note: No matching Experiment 3 benchmark JSON; consistency check skipped.")
    return accuracy, agreement, str(benchmark_path) if benchmark_path.is_file() else None


def main():
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be >= 1")
    output_csv, output_json = file_names(args)
    if not args.overwrite and (output_csv.exists() or output_json.exists()):
        raise FileExistsError(
            f"Existing evaluation output detected ({output_csv} or {output_json}). "
            "Use --overwrite only if you intentionally want to replace it."
        )
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    ref, reference_path = load_reference(args)
    ckpt, model_path, cache_meta = assert_matching_export(args)
    import torch
    import openvino as ov

    torch.set_num_threads(args.threads)
    config = {
        "INFERENCE_NUM_THREADS": str(args.threads),
        "NUM_STREAMS": "1",
        "INFERENCE_PRECISION_HINT": "f32",
    }
    core = ov.Core()
    compiled = core.compile_model(str(model_path), "CPU", config)
    request = compiled.create_infer_request()
    logits_parts = []
    processed = 0
    print(f"Seed {args.seed}: evaluating {len(ref):,} HANS examples with cached OpenVINO INT8 ...")
    for i, batch in enumerate(make_batches(ref, ckpt, args.batch_size), start=1):
        outputs = request.infer(batch)
        arr = np.asarray(next(iter(outputs.values()))).copy()
        if arr.shape != (batch["input_ids"].shape[0], 3):
            raise RuntimeError(f"Unexpected OpenVINO logits shape: {arr.shape}")
        logits_parts.append(arr)
        processed += len(arr)
        if i % 100 == 0 or processed == len(ref):
            print(f"  Processed {processed:,} / {len(ref):,} examples", flush=True)
    logits = np.concatenate(logits_parts, axis=0)
    results = build_prediction_frame(ref, logits)
    accuracy, agreement, matched_benchmark = verify_against_existing_benchmark(args, results)

    summary = {
        "experiment": "3C OpenVINO NNCF INT8_SYM per-example HANS evaluation",
        "seed": args.seed,
        "samples": len(results),
        "checkpoint": str(ckpt.relative_to(ROOT)),
        "cached_model": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
        "reference_predictions": str(reference_path.relative_to(ROOT)) if reference_path.is_relative_to(ROOT) else str(reference_path),
        "threads": args.threads,
        "batch_size": args.batch_size,
        "max_length": 128,
        "accuracy": accuracy,
        "three_class_agreement_with_saved_fp32": agreement,
        "by_heuristic_and_label": [
            {
                "heuristic": h,
                "gold_label": g,
                "n": int(len(part)),
                "accuracy": float(part.correct.mean()),
            }
            for (h, g), part in results.groupby(["heuristic", "gold_label"], sort=True)
        ],
        "int8_constant_nodes_at_export": cache_meta["int8_constant_nodes"],
        "matched_benchmark": matched_benchmark,
        "note": "Weight-compressed OpenVINO inference; probabilities and 3-class predictions saved for paired flip/margin analysis",
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_suffix(".csv.tmp")
    results.to_csv(tmp_csv, index=False)
    tmp_csv.replace(output_csv)
    output_json.write_text(json.dumps(summary, indent=2))
    print(f"\nAccuracy: {accuracy * 100:.4f}%")
    print(f"Three-class agreement with FP32: {agreement * 100:.4f}%")
    print(f"Saved predictions: {output_csv}")
    print(f"Saved summary:     {output_json}")


if __name__ == "__main__":
    main()
