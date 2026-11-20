#!/usr/bin/env python3
"""Experiment 2: CPU operator profiling and RSS diagnostics for FP32 vs weight-only INT8.

Run from project root (script in scripts/):
    uv run python scripts/experiment2_cpu_profiler_memory.py \
      --seed 42 --threads 8 --batch-size 32 --num-examples 256

- Uses the identical sampled HANS examples and fine-tuned checkpoint for both variants.
- Loads TorchAO Int8WeightOnlyConfig (74 quantized linear layers), CPU eager.
- Profiles in separate processes so each model has an independent RSS history.
- Saves stage-by-stage RSS, profiler operators, comparisons, and PNG/SVG figures.
- Does NOT overwrite old benchmark data or experimental prediction files.
- Profiler timings are diagnostic, not substitutes for Experiment 1B timings.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

VARIANTS = ("fp32", "weight_only_int8")


def rss_snapshot(stage):
    """Return Linux /proc memory counters (MiB), including RSS and high-water mark.

    VmRSS is the resident process footprint, not isolated model-weight memory.
    VmHWM is the maximum RSS *so far* in this process, not per-stage peak.
    """
    snap = {"stage": stage, "rss_mib": None, "peak_rss_so_far_mib": None,
            "pss_mib": None, "private_dirty_mib": None, "private_clean_mib": None}
    if sys.platform != "linux":
        return snap
    labels = {"VmRSS:": "rss_mib", "VmHWM:": "peak_rss_so_far_mib"}
    for line in Path("/proc/self/status").read_text().splitlines():
        chunks = line.split()
        if chunks and chunks[0] in labels and len(chunks) >= 2:
            snap[labels[chunks[0]]] = round(int(chunks[1]) / 1024, 4)
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            fields = line.split()
            mapping = {"Pss:": "pss_mib", "Private_Dirty:": "private_dirty_mib",
                       "Private_Clean:": "private_clean_mib"}
            if fields and fields[0] in mapping and len(fields) >= 2:
                snap[mapping[fields[0]]] = round(int(fields[1]) / 1024, 4)
    except (OSError, PermissionError):
        pass
    return snap


def trim_glibc():
    """Ask glibc to return free heap pages; diagnostic only, not a guaranteed reset."""
    if not sys.platform.startswith("linux"):
        return {"attempted": False, "reason": "Not Linux"}
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        fn = libc.malloc_trim
        fn.argtypes = [ctypes.c_size_t]
        fn.restype = ctypes.c_int
        returned = int(fn(0))
        return {"attempted": True, "glibc_returned_pages": bool(returned),
                "note": "May not release allocations managed by PyTorch or other allocators"}
    except (AttributeError, OSError) as e:
        return {"attempted": False, "reason": str(e)}


def safe_write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def worker(args):
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from torch.profiler import ProfilerActivity, profile

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    # Model, inputs and profiler remain on CPU (no CUDA tensors are created).

    checkpoint = args.repo_root / f"models/fp32/seed{args.seed}/final"
    hans_file = args.repo_root / f"results/hans/fp32_seed{args.seed}_predictions.csv"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint}")
    if not hans_file.is_file():
        raise FileNotFoundError(f"Missing HANS CSV: {hans_file}")

    # Reproduce the exact input sampling and pre-collation used in Experiment 1.
    hans = pd.read_csv(hans_file, usecols=["pairID", "premise", "hypothesis"])
    assert len(hans) == 30_000 and hans["pairID"].is_unique
    if args.num_examples < 30_000:
        hans = hans.sample(n=args.num_examples, random_state=2026).sort_index()
    sample_hash = hashlib.sha256("\n".join(hans["pairID"].astype(str)).encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    encodings = tokenizer(hans["premise"].tolist(), hans["hypothesis"].tolist(),
                          truncation=True, max_length=128, padding=False)
    input_keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in encodings]
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    batches = [collator([{k: encodings[k][j] for k in input_keys}
                         for j in range(i, min(i + args.batch_size, len(hans)))])
               for i in range(0, len(hans), args.batch_size)]
    assert batches
    print(f"[{args.variant}] {len(hans):,} same sampled HANS items, "
          f"{len(batches)} pre-collated CPU batches", flush=True)

    timeline = []

    def snap(stage):
        obj = rss_snapshot(stage)
        timeline.append(obj)
        print(f"  {stage:26s} RSS={obj['rss_mib']} MiB "
              f"HWM={obj['peak_rss_so_far_mib']} MiB", flush=True)

    gc.collect()
    snap("inputs_ready_before_model")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=True
    ).cpu().eval()
    if model.config.id2label != {0: "entailment", 1: "neutral", 2: "contradiction"}:
        raise ValueError("Unexpected model label mapping")
    snap("fp32_loaded")
    quantized_linear_layers = 0
    if args.variant == "weight_only_int8":
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        snap("quantizer_imported")
        quantize_(model, Int8WeightOnlyConfig())
        model.eval()
        quantized_linear_layers = sum(
            type(mod.weight).__name__ == "Int8Tensor"
            for mod in model.modules() if isinstance(mod, torch.nn.Linear)
        )
        if quantized_linear_layers != 74:
            raise AssertionError(f"Expected 74 quantized linear layers, got {quantized_linear_layers}")
        snap("quantization_finished")

    gc.collect()
    snap("model_ready_after_gc")

    def forward(batch):
        output = model(**batch).logits
        assert output.shape == (batch["input_ids"].shape[0], 3)
        return output

    with torch.inference_mode():
        for _ in range(args.warmup):
            forward(batches[len(batches) // 2])
    gc.collect()
    snap("after_warmup")

    # Unprofiled diagnostic speed, not the full-dataset Experiment 1B benchmark.
    unprofiled_seconds = []
    with torch.inference_mode():
        for _ in range(args.timing_repeats):
            start = time.perf_counter()
            for batch in batches:
                forward(batch)
            unprofiled_seconds.append(time.perf_counter() - start)
    snap("after_unprofiled_forwards")

    # Diagnostic to check whether post-quantization RSS comes from reclaimable memory.
    gc.collect()
    snap("after_post_inference_gc")
    trim_result = trim_glibc() if not args.no_trim else {"attempted": False, "reason": "Disabled by --no-trim"}
    snap("after_malloc_trim")

    # The PyTorch profiler perturbs timings and allocates extra memory;
    # do not use these values as production latency or steady-state RSS.
    print("  Profiling CPU operators...", flush=True)
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                     profile_memory=True, with_stack=False) as profiler:
            for _ in range(args.profile_repeats):
                for batch in batches:
                    forward(batch)

    events = profiler.key_averages()
    operations = []
    for event in events:
        name = str(event.key)
        if name.startswith("ProfilerStep"):
            continue
        operations.append({
            "variant": args.variant,
            "operator": name,
            "count": int(event.count),
            "self_cpu_time_ms": round(float(event.self_cpu_time_total) / 1000, 6),
            "total_cpu_time_ms": round(float(event.cpu_time_total) / 1000, 6),
            "self_cpu_memory_allocated_mib": round(float(event.self_cpu_memory_usage) / (1024 ** 2), 6),
            "total_cpu_memory_allocated_mib": round(float(event.cpu_memory_usage) / (1024 ** 2), 6),
        })
    operations.sort(key=lambda row: row["self_cpu_time_ms"], reverse=True)
    total_self = sum(r["self_cpu_time_ms"] for r in operations)
    for row in operations:
        row["self_cpu_share_pct"] = round(
            100 * row["self_cpu_time_ms"] / total_self, 4
        ) if total_self > 0 else 0

    snap("after_profiler")
    files = {
        "json": args.out_dir / f"seed{args.seed}_{args.variant}_diagnostics.json",
        "operators": args.out_dir / f"seed{args.seed}_{args.variant}_operators.csv",
        "memory": args.out_dir / f"seed{args.seed}_{args.variant}_memory_stages.csv",
    }
    if args.chrome_trace:
        trace_file = args.out_dir / f"seed{args.seed}_{args.variant}_trace.json"
        profiler.export_chrome_trace(str(trace_file))
        files["trace"] = trace_file

    lookup = {obj["stage"]: obj["rss_mib"] for obj in timeline}
    base = lookup["inputs_ready_before_model"]
    ready = lookup["model_ready_after_gc"]
    result = {
        "seed": args.seed, "variant": args.variant,
        "device": "cpu", "execution": "eager",
        "threads": args.threads, "batch_size": args.batch_size,
        "num_examples": len(hans), "sample_pairid_sha256": sample_hash,
        "num_batches": len(batches), "max_tokens": 128,
        "profile_repeats": args.profile_repeats,
        "timing_repeats": args.timing_repeats,
        "warmup": args.warmup,
        "quantized_linear_layers": quantized_linear_layers,
        "rss_input_baseline_mib": base,
        "rss_after_fp32_load_mib": lookup["fp32_loaded"],
        "rss_after_quantization_mib": lookup.get("quantization_finished"),
        "rss_ready_after_gc_mib": ready,
        "rss_ready_increment_from_inputs_mib": round(ready - base, 4) if ready is not None and base is not None else None,
        "rss_after_warmup_mib": lookup["after_warmup"],
        "rss_after_trim_mib": lookup["after_malloc_trim"],
        "rss_reduction_from_trim_mib": (
            round(lookup["after_post_inference_gc"] - lookup["after_malloc_trim"], 4)
            if lookup["after_post_inference_gc"] is not None and lookup["after_malloc_trim"] is not None else None
        ),
        "unprofiled_pass_seconds": [round(v, 5) for v in unprofiled_seconds],
        "unprofiled_median_examples_sec": round(
            len(hans) / statistics.median(unprofiled_seconds), 4),
        "profiler_total_self_cpu_ms": round(total_self, 4),
        "top_cpu_operator": operations[0]["operator"] if operations else None,
        "top_cpu_operator_share_pct": operations[0]["self_cpu_share_pct"] if operations else None,
        "glibc_trim": trim_result,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "notes": [
            "The FP32 model is loaded before TorchAO weight-only quantization; snapshots show its conversion lifecycle.",
            "Process RSS and PSS include Python, tokenization, libraries, overhead and allocator effects.",
            "PyTorch profiling perturbs CPU timing and memory; compare op attribution, not profiled wall-clock speed.",
            "The unprofiled short-run throughput is diagnostic only; Experiment 1B is the primary performance result.",
            "malloc_trim is a diagnostic intervention after unprofiled timing, not part of normal serving.",
        ],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(v.exists() for v in files.values()) and not args.overwrite:
        raise FileExistsError("Existing Experiment 2 output(s); rerun with another --out-dir")
    write_csv(files["operators"], operations,
              ["variant", "operator", "count", "self_cpu_time_ms", "total_cpu_time_ms",
               "self_cpu_memory_allocated_mib", "total_cpu_memory_allocated_mib", "self_cpu_share_pct"])
    write_csv(files["memory"], timeline)
    safe_write_json(files["json"], result)
    print(f"  Saved {args.variant}: {files['json'].name}, {files['operators'].name}, memory CSV", flush=True)


def generate_plots(summary, timeline, operations, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def save(fig, base):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(out_dir / f"{base}.{ext}", bbox_inches="tight", dpi=220)
        plt.close(fig)

    variant_labels = {"fp32": "FP32", "weight_only_int8": "Weight-only INT8"}
    stage_labels = {
        "inputs_ready_before_model": "Inputs ready",
        "fp32_loaded": "FP32 loaded",
        "quantizer_imported": "Quantizer import",
        "quantization_finished": "Quantized",
        "model_ready_after_gc": "After GC",
        "after_warmup": "After warm-up",
        "after_unprofiled_forwards": "After inference",
        "after_post_inference_gc": "After 2nd GC",
        "after_malloc_trim": "After trim",
    }

    # 01 — memory lifecycle; quantized process includes two extra stages.
    fig, ax = plt.subplots(figsize=(11.4, 5.5))
    pos = list(range(len(stage_labels)))
    stages = list(stage_labels)
    for variant in VARIANTS:
        sub = [r for r in timeline if r["variant"] == variant]
        by_stage = {r["stage"]: r.get("rss_mib") for r in sub}
        points = [(pos[i], by_stage.get(s)) for i, s in enumerate(stages)
                  if by_stage.get(s) is not None]
        ax.plot([p[0] for p in points], [p[1] for p in points],
                marker="o", linewidth=2.2, label=variant_labels[variant])
    ax.set_xticks(pos, [stage_labels[s] for s in stages], rotation=28, ha="right")
    ax.set_ylabel("Resident process memory (MiB)")
    ax.set_title("CPU process RSS across model creation and inference")
    ax.grid(axis="y", alpha=0.23)
    ax.legend(frameon=False)
    save(fig, "01_memory_lifecycle")

    # 02 — comparison of post-setup process RSS increments.
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    vals = [next(row for row in summary if row["variant"] == v)
            ["rss_ready_increment_from_inputs_mib"] for v in VARIANTS]
    bars = ax.bar([variant_labels[v] for v in VARIANTS], vals)
    for bar, value in zip(bars, vals):
        ax.annotate(f"{value:,.1f} MiB", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set_ylim(0, max(vals) * 1.22 if max(vals) else 1)
    ax.set_ylabel("RSS increase from inputs-ready baseline (MiB)")
    ax.set_title("Model setup footprint: pre-model vs inference-ready")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, "02_model_setup_rss")

    # 03 — top CPU ops, per-variant independent rank and percent.
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, variant in zip(axs, VARIANTS):
        ops = sorted([op for op in operations if op["variant"] == variant],
                     key=lambda r: r["self_cpu_time_ms"], reverse=True)[:10][::-1]
        names = [o["operator"].replace("aten::", "") for o in ops]
        values = [o["self_cpu_share_pct"] for o in ops]
        ax.barh(names, values)
        ax.set_title(variant_labels[variant])
        ax.set_xlabel("Share of profiled CPU self-time (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Dominant CPU operators — diagnostic profiler", fontsize=14)
    save(fig, "03_cpu_operator_attribution")

    # 04 — absolute time per inference for the largest ops (diagnostic).
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    from collections import defaultdict
    groups = defaultdict(dict)
    for o in operations:
        groups[o["operator"]][o["variant"]] = o["self_cpu_time_ms"]
    selected = sorted(groups, key=lambda k: sum(groups[k].values()), reverse=True)[:12]
    selected = selected[::-1]
    n_inf = {v: next(row for row in summary if row["variant"] == v)["num_batches"] *
             next(row for row in summary if row["variant"] == v)["profile_repeats"]
             for v in VARIANTS}
    ys = np.arange(len(selected))
    height = 0.38
    for idx, variant in enumerate(VARIANTS):
        vals = [groups[op].get(variant, 0) / n_inf[variant] for op in selected]
        ax.barh(ys + (idx-0.5)*height, vals, height=height, label=variant_labels[variant])
    ax.set_yticks(ys, [n.replace("aten::", "") for n in selected])
    ax.set_xlabel("Profiled operator self-time / forward batch (ms)")
    ax.set_title("CPU kernels contributing most to inference work")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    save(fig, "04_cpu_operator_per_batch")


def orchestrate(args):
    if not (args.repo_root / "pyproject.toml").is_file():
        raise FileNotFoundError("Run from a project with pyproject.toml, or pass --repo-root")
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already contains data: {args.out_dir}. "
            "Choose a new --out-dir; original benchmark results are not touched."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--repo-root", str(args.repo_root), "--out-dir", str(args.out_dir),
               "--seed", str(args.seed), "--variant", variant,
               "--threads", str(args.threads), "--batch-size", str(args.batch_size),
               "--num-examples", str(args.num_examples), "--warmup", str(args.warmup),
               "--timing-repeats", str(args.timing_repeats),
               "--profile-repeats", str(args.profile_repeats)]
        if args.chrome_trace:
            cmd.append("--chrome-trace")
        if args.no_trim:
            cmd.append("--no-trim")
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"\nRunning Experiment 2 isolated worker: {variant}", flush=True)
        subprocess.run(cmd, cwd=args.repo_root, check=True)

    summaries, timeline, operations = [], [], []
    for variant in VARIANTS:
        prefix = args.out_dir / f"seed{args.seed}_{variant}"
        summary = json.loads((args.out_dir / f"seed{args.seed}_{variant}_diagnostics.json").read_text())
        summaries.append(summary)
        with (args.out_dir / f"seed{args.seed}_{variant}_memory_stages.csv").open(newline="") as f:
            timeline.extend({"variant": variant, **row} for row in csv.DictReader(f))
        with (args.out_dir / f"seed{args.seed}_{variant}_operators.csv").open(newline="") as f:
            operations.extend({**row, "variant": variant} for row in csv.DictReader(f))

    for r in timeline:
        for c in ("rss_mib", "pss_mib", "peak_rss_so_far_mib", "private_dirty_mib", "private_clean_mib"):
            r[c] = float(r[c]) if r.get(c) not in (None, "") else None
    for r in operations:
        for c in ("self_cpu_time_ms", "total_cpu_time_ms", "self_cpu_share_pct",
                  "self_cpu_memory_allocated_mib", "total_cpu_memory_allocated_mib"):
            r[c] = float(r[c])
        r["count"] = int(r["count"])
    assert summaries[0]["sample_pairid_sha256"] == summaries[1]["sample_pairid_sha256"], \
        "FP32 and weight-only experiments used different samples"
    assert summaries[1]["quantized_linear_layers"] == 74
    write_csv(args.out_dir / "experiment2_summary.csv", [
        {k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in summaries
    ])
    write_csv(args.out_dir / "experiment2_memory_timeline.csv", timeline)
    write_csv(args.out_dir / "experiment2_operator_attribution.csv", operations)
    generate_plots(summaries, timeline, operations, args.out_dir)

    fp, wo = summaries
    report = {
        "seed": args.seed, "threads": args.threads, "batch_size": args.batch_size,
        "num_examples": args.num_examples,
        "rss_setup_delta_wo_minus_fp32_mib": round(
            wo["rss_ready_increment_from_inputs_mib"] - fp["rss_ready_increment_from_inputs_mib"], 3),
        "rss_after_trim_wo_minus_fp32_mib": round(
            wo["rss_after_trim_mib"] - fp["rss_after_trim_mib"], 3),
        "unprofiled_wo_over_fp32_speed_ratio": round(
            wo["unprofiled_median_examples_sec"] / fp["unprofiled_median_examples_sec"], 4),
        "cpu_operator_top_fp32": fp["top_cpu_operator"],
        "cpu_operator_top_wo": wo["top_cpu_operator"],
        "warning": "Profiler results are diagnostic; short-run throughput is not comparable with the 3-seed Experiment 1B benchmark. RSS is not model-only memory."
    }
    safe_write_json(args.out_dir / "experiment2_comparison.json", report)
    print("\n=== Experiment 2 summary ===")
    print(f"Setup RSS increment difference (weight-only minus FP32): "
          f"{report['rss_setup_delta_wo_minus_fp32_mib']:+.2f} MiB")
    print(f"RSS after glibc trim difference: {report['rss_after_trim_wo_minus_fp32_mib']:+.2f} MiB")
    print("Largest CPU operator (FP32):", report["cpu_operator_top_fp32"])
    print("Largest CPU operator (weight-only):", report["cpu_operator_top_wo"])
    print(f"Saved diagnostics and 4 PNG/SVG figures under {args.out_dir}")
    print("No original experiment results modified.")


def make_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42, choices=[42, 123, 456])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-examples", type=int, default=256)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=2)
    p.add_argument("--profile-repeats", type=int, default=2)
    p.add_argument("--chrome-trace", action="store_true",
                   help="Export large Chrome traces; off by default")
    p.add_argument("--no-trim", action="store_true", help="Skip diagnostic glibc malloc_trim")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting Experiment 2 files in --out-dir ONLY")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=VARIANTS, default=None, help=argparse.SUPPRESS)
    return p


def main():
    args = make_parser().parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.out_dir is None:
        args.out_dir = Path("results/runtime_analysis/experiment2_profiler") / (
            f"seed{args.seed}_t{args.threads}_b{args.batch_size}_n{args.num_examples}"
        )
    if not args.out_dir.is_absolute():
        args.out_dir = args.repo_root / args.out_dir
    args.out_dir = args.out_dir.expanduser().resolve()
    if not (1 <= args.num_examples <= 30000):
        raise ValueError("num-examples must be between 1 and 30,000")
    if not all(x > 0 for x in (args.threads, args.batch_size, args.profile_repeats, args.timing_repeats)):
        raise ValueError("Threads/batch/measurement repeats must all be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.worker:
        if args.variant is None:
            raise ValueError("--worker requires --variant")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        worker(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
