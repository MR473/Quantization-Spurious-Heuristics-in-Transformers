#!/usr/bin/env python3
"""Reproducible, isolated CPU inference benchmark: FP32 vs TorchAO weight-only INT8.

Place in scripts/benchmark_fp32_weight_only_cpu.py and run from repository root.
The script never modifies checkpoints or existing experimental result files.

Uses the exact quantizer from scripts/evaluate_weight_only_int8.py:
    quantize_(model, Int8WeightOnlyConfig())
All timing excludes tokenizer preprocessing and data loading, and runs each
model in a separate Python process to avoid cross-variant memory contamination.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

VARIANTS = ("fp32", "weight_only_int8")
OUT_DEFAULT = "results/runtime_analysis/fp32_vs_weight_only_cpu"


def current_rss_mib():
    """Process resident set in MiB. Uses standard OS interfaces; optional."""
    try:
        if sys.platform.startswith("linux"):
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 4)
        if sys.platform == "darwin":
            proc = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                check=True, capture_output=True, text=True,
            )
            return round(float(proc.stdout.strip()) / 1024.0, 4)
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass
    return None


def benchmark_worker(args):
    # Import torch only within the worker, after thread-related environment vars.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import gc
    import tempfile

    import pandas as pd
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    root = args.repo_root
    ckpt = root / f"models/fp32/seed{args.seed}/final"
    input_csv = root / f"results/hans/fp32_seed{args.seed}_predictions.csv"
    if not ckpt.is_dir():
        raise FileNotFoundError(f"Missing local FP32 checkpoint: {ckpt}")
    if not input_csv.is_file():
        raise FileNotFoundError(f"Missing HANS examples: {input_csv}")
    print(f"\n[{args.variant} / seed {args.seed}] Reading identical HANS inputs", flush=True)

    # Random, fixed subset if requested; full dataset when --num-examples=30000.
    df = pd.read_csv(input_csv, usecols=["pairID", "premise", "hypothesis"])
    assert len(df) == 30000 and df["pairID"].is_unique, "Unexpected HANS reference"
    if args.num_examples < 30000:
        df = df.sample(n=args.num_examples, random_state=2026).sort_index()
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
    tokens = tokenizer(
        df["premise"].tolist(),
        df["hypothesis"].tolist(),
        truncation=True,
        max_length=128,
        padding=False,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in tokens]
    batches = [
        collator([{key: tokens[key][j] for key in keys} for j in range(i, min(i + args.batch_size, len(df)))])
        for i in range(0, len(df), args.batch_size)
    ]
    assert len(batches) > 0
    print(f"  Prepared {len(df):,} examples in {len(batches)} pre-collated batches", flush=True)
    rss_before_model = current_rss_mib()

    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt), local_files_only=True,
    ).cpu().eval()
    quantized_layers = 0
    if args.variant == "weight_only_int8":
        from torchao.quantization import quantize_, Int8WeightOnlyConfig
        quantize_(model, Int8WeightOnlyConfig())
        model.eval()
        quantized_layers = sum(
            type(module.weight).__name__ == "Int8Tensor"
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        )
        assert quantized_layers == 74, (
            f"Expected 74 quantized linear layers, got {quantized_layers}. "
            "Check TorchAO version and model checkpoint."
        )
    gc.collect()
    rss_model_ready = current_rss_mib()

    def run_forward(batch):
        return model(**batch).logits

    # Consistent warm-up: same initial representative batch for each variant.
    latency_batch = batches[len(batches) // 2]
    with torch.inference_mode():
        for _ in range(args.warmup):
            run_forward(latency_batch)
        latency_ms = []
        for _ in range(args.latency_iterations):
            start = time.perf_counter()
            logits = run_forward(latency_batch)
            latency_ms.append(1000 * (time.perf_counter() - start))
        assert logits.shape[-1] == 3
        print("  Warm-up/latency complete", flush=True)

        # Throughput times only forward passes on the SAME pre-collated batches.
        durations = []
        for iteration in range(args.repeats):
            start = time.perf_counter()
            for batch in batches:
                logits = run_forward(batch)
            elapsed = time.perf_counter() - start
            durations.append(elapsed)
            print(f"  Full pass {iteration+1}/{args.repeats}: {elapsed:.3f}s", flush=True)

    # Serialize AFTER benchmarking; serialization can itself allocate memory.
    state_mib = None
    state_error = None
    with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
        try:
            torch.save(model.state_dict(), tmp.name)
            state_mib = round(os.path.getsize(tmp.name) / (1024.0 ** 2), 5)
        except Exception as exc:
            state_error = f"{type(exc).__name__}: {exc}"

    latency_ms.sort()
    times = sorted(durations)
    speeds = [len(df) / d for d in durations]
    result = {
        "seed": args.seed,
        "variant": args.variant,
        "quantized_linear_layers": quantized_layers,
        "device": "cpu",
        "execution": "eager",
        "num_examples": len(df),
        "batch_size": args.batch_size,
        "latency_batch_size": int(latency_batch["input_ids"].shape[0]),
        "latency_batch_seq_len": int(latency_batch["input_ids"].shape[1]),
        "max_length": 128,
        "threads": args.threads,
        "warmup_iters": args.warmup,
        "latency_iters": args.latency_iterations,
        "repeats": args.repeats,
        "median_batch_forward_ms": round(statistics.median(latency_ms), 6),
        "median_full_pass_sec": round(statistics.median(durations), 6),
        "median_examples_per_second": round(statistics.median(speeds), 6),
        "all_pass_seconds": [round(d, 6) for d in durations],
        "all_batch_forward_ms": [round(d, 6) for d in latency_ms],
        "serialized_state_mib": state_mib,
        "serialization_error": state_error,
        "rss_before_model_mib": rss_before_model,
        "rss_model_ready_mib": rss_model_ready,
        "rss_model_delta_mib": (
            round(rss_model_ready - rss_before_model, 4)
            if rss_model_ready is not None and rss_before_model is not None else None
        ),
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "subset_selection": "all_30000" if len(df) == 30000 else "random_2026_fixed_subset",
        "timing_scope": "CPU model forward only, pre-tokenized and pre-collated HANS inputs",
        "memory_note": "RSS before/after model is an approximate process measure, not isolated model allocation; quantization setup can affect allocator RSS",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / f"seed{args.seed}_{args.variant}_n{args.num_examples}.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace.")
    output.write_text(json.dumps(result, indent=2))
    print(f"  Wrote {output}", flush=True)


def plot_results(df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib unavailable: saved CSV but skipped charts.")
        return

    configs = ["fp32", "weight_only_int8"]
    names = ["FP32", "Weight-only INT8"]
    specs = [
        ("serialized_state_mib", "Serialized model size", "MiB", "01_serialized_size.png"),
        ("rss_model_ready_mib", "CPU process memory at inference-ready state", "MiB", "02_process_rss.png"),
        ("median_batch_forward_ms", f"Batch forward latency (batch {int(df['batch_size'].iloc[0])})", "Milliseconds per batch", "03_batch_latency.png"),
        ("median_examples_per_second", "CPU inference throughput", "HANS examples / second", "04_throughput.png"),
    ]
    for key, title, ylabel, filename in specs:
        vals, low, high = [], [], []
        for variant in configs:
            group = [float(v) for v in df.loc[df["variant"] == variant, key].dropna()]
            if not group:
                vals.append(float("nan")); low.append(0); high.append(0); continue
            avg = statistics.mean(group)
            vals.append(avg); low.append(avg - min(group)); high.append(max(group) - avg)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for i, (label, val) in enumerate(zip(names, vals)):
            ax.bar(i, val, width=0.6, label=label)
            if math.isfinite(val):
                ax.text(i, val + max(vals) * .025, f"{val:,.2f}", ha="center", va="bottom", fontsize=11)
        if len(df["seed"].unique()) > 1:
            ax.errorbar(range(2), vals, yerr=[low, high], fmt="none", color="black", capsize=5,
                        label="Observed min–max across seeds")
        ax.set_xticks(range(2), names)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=14)
        ax.set_ylim(bottom=0, top=max(v for v in vals if math.isfinite(v)) * 1.20)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(out_dir / filename.replace(".png", f".{ext}"), dpi=240, bbox_inches="tight")
        plt.close(fig)


def orchestrator(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not (args.repo_root / "pyproject.toml").exists():
        raise FileNotFoundError(
            f"Expected project root with pyproject.toml: {args.repo_root}. "
            "Place this script inside the repository's scripts/ directory."
        )
    records = []
    for seed in args.seeds:
        for variant in VARIANTS:
            output = args.out_dir / f"seed{seed}_{variant}_n{args.num_examples}.json"
            if output.exists() and not args.overwrite:
                previous = json.loads(output.read_text())
                expected = {
                    "batch_size": args.batch_size,
                    "threads": args.threads,
                    "repeats": args.repeats,
                    "warmup_iters": args.warmup,
                    "latency_iters": args.latency_iterations,
                    "num_examples": args.num_examples,
                }
                mismatch = {k: (previous.get(k), v) for k, v in expected.items()
                            if previous.get(k) != v}
                if mismatch or previous.get("platform") != platform.platform():
                    raise RuntimeError(
                        f"Existing benchmark config/device differs: {output}. "
                        f"Mismatches: {mismatch}. Use --overwrite to rerun."
                    )
                print(f"Using existing result {output} (pass --overwrite to rerun)", flush=True)
            else:
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--worker", "--repo-root", str(args.repo_root),
                    "--out-dir", str(args.out_dir), "--seed", str(seed),
                    "--variant", variant, "--num-examples", str(args.num_examples),
                    "--batch-size", str(args.batch_size), "--threads", str(args.threads),
                    "--repeats", str(args.repeats), "--warmup", str(args.warmup),
                    "--latency-iterations", str(args.latency_iterations), "--overwrite",
                ]
                print("\nRunning isolated benchmark:", seed, variant, flush=True)
                subprocess.run(command, check=True, cwd=args.repo_root)
            records.append(json.loads(output.read_text()))

    import pandas as pd
    df = pd.DataFrame(records).sort_values(["seed", "variant"]).reset_index(drop=True)
    df.drop(columns=["all_pass_seconds", "all_batch_forward_ms"], errors="ignore").to_csv(
        args.out_dir / f"benchmark_results_n{args.num_examples}.csv", index=False
    )
    rows = []
    for seed in args.seeds:
        pair = df.set_index(["seed", "variant"]).loc[seed]
        fp = pair.loc["fp32"]
        wo = pair.loc["weight_only_int8"]
        rows.append({
            "seed": seed,
            "num_examples": args.num_examples,
            "size_reduction_pct": (
                100 * (1 - wo["serialized_state_mib"] / fp["serialized_state_mib"])
                if pd.notna(wo["serialized_state_mib"]) and pd.notna(fp["serialized_state_mib"]) else None
            ),
            "cpu_process_rss_change_mib": (
                wo["rss_model_ready_mib"] - fp["rss_model_ready_mib"]
                if pd.notna(wo["rss_model_ready_mib"]) and pd.notna(fp["rss_model_ready_mib"]) else None
            ),
            "latency_speedup_x_fp32_over_int8": fp["median_batch_forward_ms"] / wo["median_batch_forward_ms"],
            "throughput_speedup_x_int8_over_fp32": wo["median_examples_per_second"] / fp["median_examples_per_second"],
            "fp32_throughput_examples_sec": fp["median_examples_per_second"],
            "weight_only_throughput_examples_sec": wo["median_examples_per_second"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.out_dir / f"paired_comparison_n{args.num_examples}.csv", index=False)
    plot_results(df, args.out_dir)
    print("\n=== Paired CPU benchmark comparison ===")
    print(comparison.round(3).to_string(index=False))
    print(f"\nSaved benchmark JSON, CSV and PNG/SVG figures to {args.out_dir}")
    print("Note: resident-process RSS includes Python, tokenized inputs, and allocator overhead.")
    print("Do not compare these CPU times against earlier CUDA or Apple MPS timings.")


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path(OUT_DEFAULT))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], choices=[42, 123, 456])
    parser.add_argument("--seed", type=int, choices=[42, 123, 456], help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--num-examples", type=int, default=3000,
                        help="Fixed random HANS sample; 30000 for the complete HANS evaluation")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--latency-iterations", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = get_parser().parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    if not args.out_dir.is_absolute():
        args.out_dir = args.repo_root / args.out_dir
    args.out_dir = args.out_dir.expanduser().resolve()
    assert 1 <= args.num_examples <= 30000
    assert args.batch_size > 0 and args.threads > 0
    assert args.repeats > 0 and args.warmup >= 0 and args.latency_iterations > 0
    if args.worker:
        if args.seed is None or args.variant is None:
            raise ValueError("Worker requires --seed and --variant")
        benchmark_worker(args)
    else:
        orchestrator(args)


if __name__ == "__main__":
    main()
