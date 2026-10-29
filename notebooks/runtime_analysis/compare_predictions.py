#!/usr/bin/env python3
"""Compare HANS predictions by pairID, including the compiled INT8 FP32-head variant.

Reads prediction CSVs only: never retrains, edits checkpoints or overwrites inputs.
Writes comparison summaries and mismatch examples under results/runtime_analysis/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from runtime_common import config_read, project_path, reference_csv

PROBS = ("prob_entailment", "prob_neutral", "prob_contradiction")
COLUMNS = ("pairID", "mnli_prediction_id", *PROBS)


def read_preds(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"pairID": str})
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if data["pairID"].isna().any() or data["pairID"].duplicated().any() or data["pairID"].str.strip().eq("").any():
        raise ValueError(f"{path}: missing or duplicated pairID")
    ids = pd.to_numeric(data["mnli_prediction_id"], errors="coerce")
    if ids.isna().any() or ~ids.isin([0, 1, 2]).all():
        raise ValueError(f"{path}: MNLI predictions must be integers 0, 1 or 2")
    for col in PROBS:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    arr = data[list(PROBS)].to_numpy(dtype=float)
    if not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{path}: probabilities must be finite and in [0, 1]")
    return data[list(COLUMNS)].copy()


def compare(seed, label, baseline_variant, candidate_variant, ref_path, new_path, tolerance, expected_n):
    reference = read_preds(ref_path)
    candidate = read_preds(new_path)
    if expected_n is not None and (len(reference) != expected_n or len(candidate) != expected_n):
        raise ValueError(f"Expected {expected_n} rows; reference={len(reference)}, candidate={len(candidate)}")
    if set(reference["pairID"]) != set(candidate["pairID"]):
        only_ref = len(set(reference["pairID"]) - set(candidate["pairID"]))
        only_new = len(set(candidate["pairID"]) - set(reference["pairID"]))
        raise ValueError(f"Nonmatching pairID sets: {only_ref} reference-only, {only_new} candidate-only")

    merged = reference.merge(candidate, on="pairID", how="inner", suffixes=("_ref", "_new"), validate="one_to_one")
    rid = merged["mnli_prediction_id_ref"].to_numpy(dtype=int)
    nid = merged["mnli_prediction_id_new"].to_numpy(dtype=int)
    pdiff = np.column_stack([
        (merged[f"{col}_new"] - merged[f"{col}_ref"]).to_numpy(dtype=float)
        for col in PROBS
    ])
    absdiff = np.abs(pdiff)
    three_flips = rid != nid
    binary_flips = (rid == 0) != (nid == 0)
    max_row_diff = absdiff.max(axis=1)

    detail_mask = three_flips | (max_row_diff > tolerance)
    detail = pd.DataFrame({
        "pairID": merged["pairID"],
        "mnli_prediction_id_reference": rid,
        "mnli_prediction_id_candidate": nid,
        "reference_hans_prediction": np.where(rid == 0, "entailment", "non-entailment"),
        "candidate_hans_prediction": np.where(nid == 0, "entailment", "non-entailment"),
        "mnli_3way_changed": three_flips,
        "hans_binary_changed": binary_flips,
        "delta_p_entailment_candidate_minus_reference": pdiff[:, 0],
        "delta_p_neutral_candidate_minus_reference": pdiff[:, 1],
        "delta_p_contradiction_candidate_minus_reference": pdiff[:, 2],
        "max_absolute_probability_difference": max_row_diff,
    }).loc[detail_mask].copy()

    result = {
        "seed": int(seed),
        "comparison": label,
        "reference_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "reference_file": str(ref_path),
        "candidate_file": str(new_path),
        "examples": len(merged),
        "mnli_3way_disagreement_n": int(three_flips.sum()),
        "mnli_3way_disagreement_percent": float(three_flips.mean() * 100),
        "hans_binary_disagreement_n": int(binary_flips.sum()),
        "hans_binary_disagreement_percent": float(binary_flips.mean() * 100),
        "probability_mae": float(absdiff.mean()),
        "probability_p99_abs_difference": float(np.quantile(absdiff, 0.99)),
        "probability_max_abs_difference": float(absdiff.max()),
        "fraction_probability_elements_over_tolerance": float((absdiff > tolerance).mean()),
        "tolerance": float(tolerance),
        "exact_label_agreement": bool(not three_flips.any()),
        "all_probabilities_within_tolerance": bool(np.all(absdiff <= tolerance)),
        "num_flagged_examples": len(detail),
    }
    return result, detail


def get_checks(cfg, root, seed):
    directory = root / f"seed{seed}"
    output = lambda variant: directory / variant / "predictions.csv"
    return [
        ("Original FP32 reference vs FP32 eager CUDA", "original_fp32", "fp32_eager", reference_csv(cfg, seed, "fp32"), output("fp32_eager")),
        ("Original INT8 reference vs INT8 eager CUDA", "original_int8", "int8_eager", reference_csv(cfg, seed, "int8"), output("int8_eager")),
        ("FP32 compiled vs FP32 eager CUDA", "fp32_eager", "fp32_compiled", output("fp32_eager"), output("fp32_compiled")),
        ("Original INT8 compiled vs INT8 eager CUDA", "int8_eager", "int8_compiled", output("int8_eager"), output("int8_compiled")),
        ("INT8 compiled FP32 head vs INT8 eager CUDA", "int8_eager", "int8_compiled_fp32_head", output("int8_eager"), output("int8_compiled_fp32_head")),
        ("INT8 compiled FP32 head vs original INT8 reference", "original_int8", "int8_compiled_fp32_head", reference_csv(cfg, seed, "int8"), output("int8_compiled_fp32_head")),
    ]



def generated_prediction_issue(path: Path, root: Path, cfg: dict) -> str | None:
    """Reject stale/failed worker output, without constraining original saved HANS CSVs."""
    try:
        path.relative_to(root)
    except ValueError:
        return None  # Original prediction CSV in results/hans/
    status_path = path.parent / "predictions_status.json"
    metadata_path = path.parent / "prediction_metadata.json"
    if not status_path.is_file() or not metadata_path.is_file():
        return "Missing successful predictions_status.json or prediction_metadata.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return f"Unreadable prediction worker metadata: {e}"
    if status.get("status") != "success":
        return f"Prediction worker status is {status.get('status')!r}"
    if str(metadata.get("variant")) != path.parent.name:
        return "Prediction metadata variant does not match folder"
    if int(metadata.get("max_length", -1)) != int(cfg["max_length"]):
        return "Prediction max_length differs from current configuration"
    if int(metadata.get("throughput_batch_size", -1)) != int(cfg["throughput_batch_size"]):
        return "Prediction batch size differs from current configuration"
    if "compiled" in path.parent.name and metadata.get("compile_mode") != cfg.get("compile_mode"):
        return "Prediction compilation mode differs from current configuration"
    return None

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--tolerance", type=float, default=None)
    args = ap.parse_args()
    cfg = config_read()
    tolerance = float(cfg["prediction_probability_tolerance"] if args.tolerance is None else args.tolerance)
    if not 0 <= tolerance <= 1:
        raise ValueError("Probability tolerance must be between 0 and 1")
    root = project_path(cfg["results_dir"])
    root.mkdir(parents=True, exist_ok=True)
    detail_dir = root / "prediction_differences"
    detail_dir.mkdir(parents=True, exist_ok=True)
    expected_n = None if cfg.get("max_examples") is not None else 30000

    rows, statuses = [], []
    for seed in args.seeds:
        for label, baseline, candidate, ref, new in get_checks(cfg, root, seed):
            status = {"seed": seed, "comparison": label, "reference_file": str(ref), "candidate_file": str(new)}
            if not ref.is_file() or not new.is_file():
                status.update(status="skipped", reason="Missing reference and/or candidate predictions")
                print(f"SKIP seed {seed}: {label} (missing prediction file)")
                statuses.append(status)
                continue
            issues = [generated_prediction_issue(p, root, cfg) for p in (ref, new)]
            issues = [reason for reason in issues if reason]
            if issues:
                status.update(status="skipped", reason="; ".join(issues))
                print(f"SKIP seed {seed}: {label} ({'; '.join(issues)})")
                statuses.append(status)
                continue
            try:
                rec, detail = compare(seed, label, baseline, candidate, ref, new, tolerance, expected_n)
            except Exception as e:
                status.update(status="error", reason=str(e))
                print(f"ERROR seed {seed}: {label}: {e}")
                statuses.append(status)
                continue
            filename = f"seed{seed}_{baseline}_vs_{candidate}.csv"
            path = detail_dir / filename
            detail.to_csv(path, index=False)
            rec["flagged_examples_csv"] = str(path)
            rows.append(rec)
            status.update(status="completed", reason="", details_file=str(path))
            statuses.append(status)
            print(f"OK seed {seed}: {label} | MNLI flips={rec['mnli_3way_disagreement_n']} | HANS flips={rec['hans_binary_disagreement_n']} | P-MAE={rec['probability_mae']:.6g}")

    status_file = root / "prediction_equivalence_status.csv"
    pd.DataFrame(statuses).to_csv(status_file, index=False)
    report = root / "prediction_equivalence_checks.csv"
    if rows:
        pd.DataFrame(rows).to_csv(report, index=False)
        print(f"Saved comparisons: {report}")
    else:
        if report.exists():
            report.unlink()  # avoid reporting stale equivalence conclusions
        print("No valid comparisons available; check prediction_equivalence_status.csv")
    print(f"Saved comparison availability: {status_file}")
    print("A compiled FP32-head INT8 prediction match must be measured; do not assume equality from architecture alone.")


if __name__ == "__main__":
    main()
