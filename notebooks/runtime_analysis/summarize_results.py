#!/usr/bin/env python3
"""Consolidate runtime-analysis benchmark CSVs and fair FP32/INT8 pairs.

Allows original eager and new compiled FP32-head INT8 comparisons while
explicitly detecting protocol mismatches and failed/stale experiment status.
Never reruns or modifies any checkpoint or benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime_common import config_read, project_path

METRICS = [
    "serialized_state_mib",
    "cuda_model_resident_delta_mib",
    "median_peak_cuda_allocated_mib",
    "median_peak_extra_cuda_allocated_mib",
    "median_peak_cuda_reserved_mib",
    "median_throughput_examples_per_second",
    "median_full_hans_seconds",
    "batch1_median_forward_latency_ms",
    "batch1_p95_forward_latency_ms",
    "batch32_median_forward_latency_ms",
    "batch32_p95_forward_latency_ms",
    "runtime_quantization_setup_seconds",
]

PAIRS = [
    ("eager_original_int8", "fp32_eager", "int8_eager", "Original W8A8 dynamic INT8 versus FP32; same eager execution"),
    ("compiled_original_int8", "fp32_compiled", "int8_compiled", "Original W8A8 dynamic INT8 versus FP32; same compiled execution"),
    ("compiled_fp32_head_int8", "fp32_compiled", "int8_compiled_fp32_head", "Modified W8A8 (FP32 classifier) versus FP32; same compiled execution"),
]


def safe_float(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (ValueError, TypeError):
        return np.nan


def protocol_differences(a, b):
    """Check that the pair used the same inputs, execution and batch treatment."""
    differences = []
    for key in ("device", "max_length", "throughput_batch_size", "num_hans_examples", "hans_source", "partial_batch_policy", "padded_duplicate_examples_per_pass"):
        # Missing padding metadata on both sides means an older, matching protocol.
        x, y = a.get(key, None), b.get(key, None)
        if key == "hans_source":
            # Different per-seed files may still represent identical rows; for
            # a same-seed pair insist on same source to avoid silent mismatch.
            pass
        if pd.isna(x) and pd.isna(y):
            continue
        if x != y:
            differences.append(f"{key}: FP32={x!r}, INT8={y!r}")
    # Only compiled pairs will have compilation metadata; verify these match.
    if "compiled" in str(a.get("variant", "")) or "compiled" in str(b.get("variant", "")):
        for key in ("compile_mode", "compile_dynamic"):
            if a.get(key) != b.get(key):
                differences.append(f"{key}: FP32={a.get(key)!r}, INT8={b.get(key)!r}")
    return differences


def flatten_csv(root, name, dest, statuses):
    tables = []
    for folder in statuses:
        path = folder / name
        if path.is_file():
            frame = pd.read_csv(path)
            frame["source_folder"] = str(folder)
            tables.append(frame)
    if tables:
        pd.concat(tables, ignore_index=True).to_csv(root / dest, index=False)
    elif (root / dest).exists():
        (root / dest).unlink()


def main():
    cfg = config_read()
    output = project_path(cfg["results_dir"])
    output.mkdir(parents=True, exist_ok=True)
    summaries, directories, statuses = [], [], []

    for folder in sorted(output.glob("seed*/*")):
        if not folder.is_dir():
            continue
        summary_file = folder / "benchmark_summary.json"
        status_file = folder / "benchmark_status.json"
        # A previous successful summary can remain after a later failed run.
        # Require a successful status and a matching successful summary.
        if not summary_file.is_file() and not status_file.is_file():
            continue
        entry = {"folder": str(folder), "status": "excluded", "reason": ""}
        if not summary_file.is_file() or not status_file.is_file():
            entry["reason"] = "Missing benchmark_summary.json or benchmark_status.json"
        else:
            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                status = json.loads(status_file.read_text(encoding="utf-8"))
                if status.get("status") != "success" or summary.get("status") != "success":
                    entry["reason"] = f"Benchmark last recorded status: {status.get('status')!r}, summary={summary.get('status')!r}"
                elif str(summary.get("variant")) != folder.name or f"seed{summary.get('seed')}" != folder.parent.name:
                    entry["reason"] = "Summary variant/seed does not match directory"
                else:
                    summaries.append(summary)
                    directories.append(folder)
                    entry["status"] = "included"
            except (ValueError, OSError) as e:
                entry["reason"] = f"Failed reading status/summary: {e}"
        statuses.append(entry)

    pd.DataFrame(statuses, columns=["folder", "status", "reason"]).to_csv(output / "benchmark_inclusion_status.csv", index=False)
    if not summaries:
        print("No successfully completed benchmark summaries. Run the benchmark notebooks first.")
        for stale in ("all_benchmark_variants.csv", "paired_fp32_int8_comparison.csv", "all_throughput_repeats.csv", "all_latency_samples.csv"):
            (output / stale).unlink(missing_ok=True)
        return

    all_df = pd.DataFrame(summaries).sort_values(["seed", "variant"])
    all_df.to_csv(output / "all_benchmark_variants.csv", index=False)
    flatten_csv(output, "throughput_repeats.csv", "all_throughput_repeats.csv", directories)
    flatten_csv(output, "latency_samples.csv", "all_latency_samples.csv", directories)

    rows, eligibility = [], []
    for seed in sorted(all_df["seed"].unique()):
        subset = all_df[all_df["seed"] == seed].set_index("variant", drop=False)
        for mode, fp_variant, int_variant, description in PAIRS:
            status = {"seed": int(seed), "mode": mode, "fp32_variant": fp_variant, "int8_variant": int_variant, "eligible": False, "reason": ""}
            if fp_variant not in subset.index or int_variant not in subset.index:
                status["reason"] = "Missing one or both successful benchmark variants"
                eligibility.append(status)
                continue
            fp = subset.loc[fp_variant]
            iq = subset.loc[int_variant]
            problems = protocol_differences(fp, iq)
            if problems:
                status["reason"] = "; ".join(problems)
                eligibility.append(status)
                continue
            status["eligible"] = True
            eligibility.append(status)
            rec = {"seed": int(seed), "mode": mode, "fp32_variant": fp_variant,
                   "int8_variant": int_variant, "comparison_note": description,
                   "throughput_batch_size": fp.get("throughput_batch_size"),
                   "num_hans_examples": fp.get("num_hans_examples"),
                   "partial_batch_policy": fp.get("partial_batch_policy", "not recorded"),
                   "compile_mode": fp.get("compile_mode", None),
                   "device": fp.get("device", None)}
            for metric in METRICS:
                x, y = safe_float(fp.get(metric)), safe_float(iq.get(metric))
                rec[f"fp32_{metric}"] = x
                rec[f"int8_{metric}"] = y
                rec[f"int8_change_pct_{metric}"] = 100 * (y / x - 1) if np.isfinite(x) and np.isfinite(y) and x != 0 else np.nan
            fp_th = rec.get("fp32_median_throughput_examples_per_second", np.nan)
            iq_th = rec.get("int8_median_throughput_examples_per_second", np.nan)
            rec["int8_over_fp32_throughput_x"] = iq_th / fp_th if np.isfinite(fp_th) and np.isfinite(iq_th) and fp_th > 0 else np.nan
            fp_time = rec.get("fp32_median_full_hans_seconds", np.nan)
            iq_time = rec.get("int8_median_full_hans_seconds", np.nan)
            rec["int8_inference_speedup_x"] = fp_time / iq_time if np.isfinite(fp_time) and np.isfinite(iq_time) and iq_time > 0 else np.nan
            rows.append(rec)

    pd.DataFrame(eligibility).to_csv(output / "pair_eligibility_status.csv", index=False)
    paired_file = output / "paired_fp32_int8_comparison.csv"
    if rows:
        paired = pd.DataFrame(rows)
        paired.to_csv(paired_file, index=False)
        cols = ["seed", "mode", "fp32_variant", "int8_variant", "int8_over_fp32_throughput_x"]
        print("Matched comparisons:\n", paired[cols].to_string(index=False))
    else:
        paired_file.unlink(missing_ok=True)
        print("No matched comparisons yet. See pair_eligibility_status.csv for reasons.")
    print(f"Saved: {output / 'all_benchmark_variants.csv'}")
    print(f"Saved: {output / 'benchmark_inclusion_status.csv'}")
    print(f"Saved: {output / 'pair_eligibility_status.csv'}")
    print("NOTE: FP32-head INT8 is a DISTINCT quantization variant; check 04 prediction equivalence before relating to original HANS behavioral results.")


if __name__ == "__main__":
    main()
