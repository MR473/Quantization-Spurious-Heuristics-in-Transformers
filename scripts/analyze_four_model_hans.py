#!/usr/bin/env python3
"""Experiment 3D: four-model HANS behavioral analysis using existing prediction CSVs.

Compare FP32, TorchAO W8A8, TorchAO weight-only INT8, and OpenVINO INT8
across seeds 42, 123, and 456. Does not execute any models or overwrite the
existing three-model analysis in results/analysis/.

Run from repository root:
    uv run python scripts/analyze_four_model_hans.py

Output: results/analysis/four_model_comparison/ (use --overwrite to rerun).

Caveat: totals across seeds are seed-example pairs, not 90k independent
examples; the same HANS examples are evaluated under three checkpoints.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 123, 456]
MODELS = ["FP32", "W8A8", "Weight-only", "OpenVINO INT8"]
QUANTIZED = MODELS[1:]
HEURISTICS = ["lexical_overlap", "subsequence", "constituent"]
LABELS = ["entailment", "non-entailment"]
PROB_COLS = ["prob_entailment", "prob_neutral", "prob_contradiction"]
REQUIRED = {
    "pairID", "gold_label", "heuristic", "subcase", "mnli_prediction_id",
    "hans_prediction", "correct", *PROB_COLS, "prob_non_entailment",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true", help="Explicitly replace existing four-model reports")
    return p.parse_args()


def paths(root, seed):
    folder = root / "results" / "hans"
    return {
        "FP32": folder / f"fp32_seed{seed}_predictions.csv",
        "W8A8": folder / f"int8_seed{seed}_predictions.csv",
        "Weight-only": folder / "weight_only_int8" / f"weight_only_int8_seed{seed}_predictions.csv",
        "OpenVINO INT8": folder / "openvino_int8" / f"openvino_int8_seed{seed}_predictions.csv",
    }


def load_and_validate(root, seed):
    source_paths = paths(root, seed)
    missing_files = [str(p) for p in source_paths.values() if not p.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing required prediction file(s):\n  " + "\n  ".join(missing_files))

    frames = {name: pd.read_csv(path) for name, path in source_paths.items()}
    fp = frames["FP32"]
    if len(fp) != 30_000:
        raise ValueError(f"Seed {seed}: expected 30000 FP32 rows, got {len(fp)}")
    if fp["pairID"].isna().any() or not fp["pairID"].is_unique:
        raise ValueError(f"Seed {seed}: FP32 pairIDs must be unique and non-null")

    aligned = {}
    reference_ids = fp["pairID"].astype(str)
    if reference_ids.duplicated().any():
        raise ValueError(f"Seed {seed}: FP32 pairID values are not unique after string conversion")

    for model, df in frames.items():
        absent = REQUIRED.difference(df.columns)
        if absent:
            raise ValueError(f"Seed {seed}, {model}: missing columns {sorted(absent)}")
        if len(df) != len(fp):
            raise ValueError(f"Seed {seed}, {model}: unexpected number of rows {len(df)}")

        df = df.copy()
        df["pairID"] = df["pairID"].astype(str)
        if df["pairID"].isna().any() or df["pairID"].duplicated().any():
            raise ValueError(f"Seed {seed}, {model}: invalid pairIDs")
        if set(df["pairID"]) != set(reference_ids):
            raise ValueError(f"Seed {seed}, {model}: different HANS pairID set from FP32")
        aligned[model] = df.set_index("pairID").loc[reference_ids].reset_index()

    fp = aligned["FP32"]
    for model, df in aligned.items():
        for field in ["gold_label", "heuristic", "subcase"]:
            if not df[field].equals(fp[field]):
                raise ValueError(f"Seed {seed}, {model}: {field} mismatch after pairID alignment")
        if df["gold_label"].isna().any() or df["heuristic"].isna().any():
            raise ValueError(f"Seed {seed}, {model}: missing gold labels/heuristics")

        ids = pd.to_numeric(df["mnli_prediction_id"], errors="coerce")
        if not ids.isin([0, 1, 2]).all():
            raise ValueError(f"Seed {seed}, {model}: expected MNLI classes 0,1,2")
        predicted_binary = np.where(ids.to_numpy() == 0, "entailment", "non-entailment")
        if not np.array_equal(df["hans_prediction"].to_numpy(), predicted_binary):
            raise ValueError(f"Seed {seed}, {model}: HANS prediction mapping differs")
        expected_correct = df["hans_prediction"].eq(df["gold_label"])
        # Pandas generally infers this column as bool from 'True' and 'False'.
        if not np.array_equal(df["correct"].astype(str).str.lower().to_numpy(),
                              np.where(expected_correct, "true", "false")):
            raise ValueError(f"Seed {seed}, {model}: incorrect correctness flags")
        df["correct"] = expected_correct.astype(bool)
        df["mnli_prediction_id"] = ids.astype(int)

        probs = df[PROB_COLS].to_numpy(dtype=float)
        if (not np.isfinite(probs).all() or
            (probs < -1e-7).any() or (probs > 1 + 1e-7).any() or
            not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)):
            raise ValueError(f"Seed {seed}, {model}: invalid probability columns")
        if not np.allclose(df["prob_non_entailment"].to_numpy(dtype=float),
                           probs[:, 1] + probs[:, 2], atol=1e-4):
            raise ValueError(f"Seed {seed}, {model}: inconsistent non-entailment probability")
        if not np.array_equal(probs.argmax(axis=1), ids.to_numpy()):
            # Invalid class mapping or probabilities make margin and flips unreliable.
            raise ValueError(f"Seed {seed}, {model}: argmax(probabilities) != class ID")

    observed_groups = set(zip(fp["heuristic"], fp["gold_label"]))
    expected_groups = set((h, lab) for h in HEURISTICS for lab in LABELS)
    if observed_groups != expected_groups:
        raise ValueError(f"Seed {seed}: unexpected HANS heuristic/label groups: {observed_groups ^ expected_groups}")
    return aligned


def margin(df):
    return df["prob_entailment"].to_numpy(dtype=float) - np.maximum(
        df["prob_neutral"].to_numpy(dtype=float),
        df["prob_contradiction"].to_numpy(dtype=float),
    )


def seed_analysis(seed, data):
    fp = data["FP32"]
    fp_correct = fp["correct"].to_numpy(dtype=bool)
    fp_ids = fp["mnli_prediction_id"].to_numpy(dtype=int)
    fp_hans = fp["hans_prediction"].to_numpy()
    fp_margin = margin(fp)

    group_masks = [("overall", "all", np.ones(len(fp), dtype=bool))]
    for h in HEURISTICS:
        for lab in LABELS:
            group_masks.append((h, lab,
                                ((fp["heuristic"] == h) & (fp["gold_label"] == lab)).to_numpy()))

    accuracy, flips, margins, agreement, overlaps = [], [], [], [], []
    harm_masks = {}

    for model, df in data.items():
        correct = df["correct"].to_numpy(dtype=bool)
        ids = df["mnli_prediction_id"].to_numpy(dtype=int)
        hans = df["hans_prediction"].to_numpy()
        prediction_change = ids != fp_ids
        binary_change = hans != fp_hans
        for h, lab, mask in group_masks:
            n = int(mask.sum())
            accuracy.append({"seed": seed, "model": model, "heuristic": h,
                             "gold_label": lab, "n": n,
                             "accuracy": float(correct[mask].mean())})
            agreement.append({"seed": seed, "model": model, "heuristic": h,
                              "gold_label": lab, "n": n,
                              "three_class_agreement": float(1 - prediction_change[mask].mean()),
                              "binary_hans_agreement": float(1 - binary_change[mask].mean()),
                              "three_class_changed": int(prediction_change[mask].sum()),
                              "binary_hans_changed": int(binary_change[mask].sum())})
        if model == "FP32":
            continue
        harm = fp_correct & ~correct
        benefit = ~fp_correct & correct
        harm_masks[model] = harm
        shift = margin(df) - fp_margin
        for h, lab, mask in group_masks:
            n = int(mask.sum())
            flips.append({"seed": seed, "model": model, "heuristic": h,
                          "gold_label": lab, "n": n,
                          "harmful": int((mask & harm).sum()),
                          "beneficial": int((mask & benefit).sum()),
                          "net_accuracy_change": float(((mask & benefit).sum() - (mask & harm).sum()) / n)})
            margins.append({"seed": seed, "model": model, "heuristic": h,
                            "gold_label": lab, "n": n,
                            "mean_shift": float(shift[mask].mean()),
                            "median_shift": float(np.median(shift[mask])),
                            "positive_fraction": float((shift[mask] > 0).mean())})

    for a, b in combinations(QUANTIZED, 2):
        for h, lab, mask in group_masks:
            ah = harm_masks[a] & mask
            bh = harm_masks[b] & mask
            overlaps.append({"seed": seed, "model_a": a, "model_b": b,
                             "heuristic": h, "gold_label": lab, "n": int(mask.sum()),
                             "both_harmful": int((ah & bh).sum()),
                             "only_a_harmful": int((ah & ~bh).sum()),
                             "only_b_harmful": int((~ah & bh).sum())})
    return accuracy, flips, margins, agreement, overlaps


def summarise(df, keys, metrics):
    return (df.groupby(keys, as_index=False)
            .agg(**{f"mean_{metric}": (metric, "mean") for metric in metrics},
                 **{f"std_{metric}": (metric, "std") for metric in metrics}))


def main():
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    out = (args.out_dir or (root / "results/analysis/four_model_comparison")).expanduser().resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already contains files: {out}\nUse --overwrite to replace only reports in that directory.")

    buckets = {"accuracy": [], "flips": [], "margins": [], "agreement": [], "overlap": []}
    for seed in SEEDS:
        data = load_and_validate(root, seed)
        acc, flip, mar, agree, overlap = seed_analysis(seed, data)
        for name, new_rows in zip(buckets, (acc, flip, mar, agree, overlap)):
            buckets[name].extend(new_rows)
        print(f"Seed {seed}: validated four models, 30,000 aligned HANS pairs")

    tables = {name: pd.DataFrame(rows) for name, rows in buckets.items()}
    out.mkdir(parents=True, exist_ok=True)
    names = {
        "accuracy": "hans_accuracy_by_seed.csv",
        "flips": "hans_flips_by_seed.csv",
        "margins": "hans_margin_shifts_by_seed.csv",
        "agreement": "hans_prediction_agreement_by_seed.csv",
        "overlap": "hans_harmful_flip_pairwise_overlap.csv",
    }
    for name, filename in names.items():
        tables[name].to_csv(out / filename, index=False)

    keys = ["model", "heuristic", "gold_label"]
    acc_summary = summarise(tables["accuracy"], keys, ["accuracy"])
    flip_summary = summarise(tables["flips"], keys, ["harmful", "beneficial", "net_accuracy_change"])
    # Add pooled counts to make clear these are seed-level totals.
    pooled = tables["flips"].groupby(keys, as_index=False).agg(
        pooled_harmful_seed_examples=("harmful", "sum"),
        pooled_beneficial_seed_examples=("beneficial", "sum"))
    flip_summary = flip_summary.merge(pooled, on=keys, validate="one_to_one")
    margin_summary = summarise(tables["margins"], keys, ["mean_shift", "median_shift", "positive_fraction"])
    agreement_summary = summarise(tables["agreement"], keys,
                                  ["three_class_agreement", "binary_hans_agreement"])
    for filename, df in [
        ("hans_accuracy_summary.csv", acc_summary),
        ("hans_flip_summary.csv", flip_summary),
        ("hans_margin_summary.csv", margin_summary),
        ("hans_prediction_agreement_summary.csv", agreement_summary),
    ]:
        df.to_csv(out / filename, index=False)

    overall = acc_summary.query('heuristic == "overall"').copy()
    overall["mean_accuracy_pct"] = overall["mean_accuracy"] * 100
    print("\nOVERALL HANS ACCURACY (%), mean across seeds:")
    print(overall[["model", "mean_accuracy_pct"]].round(4).to_string(index=False))

    subset = acc_summary.query('heuristic == "lexical_overlap" and gold_label == "non-entailment"').copy()
    subset["mean_accuracy_pct"] = subset["mean_accuracy"] * 100
    print("\nLEXICAL-OVERLAP / NON-ENTAILMENT ACCURACY (%):")
    print(subset[["model", "mean_accuracy_pct"]].round(4).to_string(index=False))

    print("\nLEXICAL-OVERLAP / NON-ENTAILMENT FLIPS (pooled over 3 seed-runs):")
    q = flip_summary.query('heuristic == "lexical_overlap" and gold_label == "non-entailment"')
    print(q[["model", "pooled_harmful_seed_examples", "pooled_beneficial_seed_examples"]].to_string(index=False))

    print("\nLEXICAL-OVERLAP / NON-ENTAILMENT MARGIN SHIFTS:")
    q = margin_summary.query('heuristic == "lexical_overlap" and gold_label == "non-entailment"')
    print(q[["model", "mean_mean_shift", "mean_positive_fraction"]].round(4).to_string(index=False))

    print(f"\nSaved {len(names) + 4} CSV reports in: {out}")
    print("No models were rerun. Previous results/analysis/ CSVs are unchanged.")


if __name__ == "__main__":
    main()
