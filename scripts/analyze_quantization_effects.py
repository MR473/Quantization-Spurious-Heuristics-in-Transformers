from pathlib import Path
import json

import numpy as np
import pandas as pd


SEEDS = [42, 123, 456]
MODELS = ["FP32", "W8A8", "Weight-only"]
HEURISTICS = ["lexical_overlap", "subsequence", "constituent"]
LABELS = ["entailment", "non-entailment"]

OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED = {
    "pairID", "gold_label", "heuristic", "subcase",
    "mnli_prediction_id", "hans_prediction", "correct",
    "prob_entailment", "prob_neutral", "prob_contradiction",
    "prob_non_entailment",
}


def paths(seed):
    return {
        "FP32": Path(f"results/hans/fp32_seed{seed}_predictions.csv"),
        "W8A8": Path(f"results/hans/int8_seed{seed}_predictions.csv"),
        "Weight-only": Path(
            f"results/hans/weight_only_int8/"
            f"weight_only_int8_seed{seed}_predictions.csv"
        ),
    }


def validate(data, seed):
    reference = data["FP32"]

    assert len(reference) == 30000, f"Unexpected FP32 rows: seed {seed}"
    assert reference["pairID"].is_unique, "Duplicate pairID values"

    for model, df in data.items():
        missing = REQUIRED - set(df.columns)
        assert not missing, f"{model} missing columns: {missing}"
        assert len(df) == len(reference), f"{model} row count mismatch"

        for column in ["pairID", "gold_label", "heuristic", "subcase"]:
            assert df[column].equals(reference[column]), (
                f"{model} alignment mismatch: {column}, seed {seed}"
            )

        assert df["correct"].dtype == bool, (
            f"{model} correct column is not Boolean"
        )

        expected_correct = df["hans_prediction"].eq(df["gold_label"])
        assert df["correct"].equals(expected_correct), (
            f"{model} correctness mismatch"
        )

        probs = df[
            ["prob_entailment", "prob_neutral", "prob_contradiction"]
        ].to_numpy(dtype=float)

        assert np.isfinite(probs).all(), f"{model} nonfinite probabilities"
        assert ((probs >= 0) & (probs <= 1)).all(), (
            f"{model} probabilities outside [0,1]"
        )
        assert np.allclose(probs.sum(axis=1), 1, atol=1e-4), (
            f"{model} probabilities do not sum to one"
        )

        expected_non_entailment = probs[:, 1] + probs[:, 2]
        assert np.allclose(
            df["prob_non_entailment"].to_numpy(dtype=float),
            expected_non_entailment,
            atol=1e-4,
        ), f"{model} non-entailment probability mismatch"

        expected_hans = np.where(
            df["mnli_prediction_id"].to_numpy() == 0,
            "entailment",
            "non-entailment",
        )
        assert np.array_equal(
            df["hans_prediction"].to_numpy(), expected_hans
        ), f"{model} HANS mapping mismatch"


def margin(df):
    return df["prob_entailment"] - df[
        ["prob_neutral", "prob_contradiction"]
    ].max(axis=1)


accuracy_rows = []
flip_rows = []
margin_rows = []
overlap_rows = []

for seed in SEEDS:
    data = {
        name: pd.read_csv(path)
        for name, path in paths(seed).items()
    }

    validate(data, seed)

    fp = data["FP32"]
    fp_margin = margin(fp)

    for model in MODELS:
        df = data[model]

        accuracy_rows.append({
            "seed": seed,
            "model": model,
            "heuristic": "overall",
            "gold_label": "all",
            "n": len(df),
            "accuracy": df["correct"].mean(),
        })

        for heuristic in HEURISTICS:
            for label in LABELS:
                mask = (
                    (fp["heuristic"] == heuristic)
                    & (fp["gold_label"] == label)
                )

                accuracy_rows.append({
                    "seed": seed,
                    "model": model,
                    "heuristic": heuristic,
                    "gold_label": label,
                    "n": int(mask.sum()),
                    "accuracy": df.loc[mask, "correct"].mean(),
                })

        if model == "FP32":
            continue

        harmful = fp["correct"] & ~df["correct"]
        beneficial = ~fp["correct"] & df["correct"]
        shift = margin(df) - fp_margin

        for heuristic in HEURISTICS:
            for label in LABELS:
                mask = (
                    (fp["heuristic"] == heuristic)
                    & (fp["gold_label"] == label)
                )

                flip_rows.append({
                    "seed": seed,
                    "model": model,
                    "heuristic": heuristic,
                    "gold_label": label,
                    "n": int(mask.sum()),
                    "harmful": int((mask & harmful).sum()),
                    "beneficial": int((mask & beneficial).sum()),
                    "net_accuracy_change": (
                        int((mask & beneficial).sum())
                        - int((mask & harmful).sum())
                    ) / int(mask.sum()),
                })

                selected = shift[mask]

                margin_rows.append({
                    "seed": seed,
                    "model": model,
                    "heuristic": heuristic,
                    "gold_label": label,
                    "n": int(mask.sum()),
                    "mean_shift": selected.mean(),
                    "median_shift": selected.median(),
                    "positive_fraction": (selected > 0).mean(),
                })

    wa = data["W8A8"]
    wo = data["Weight-only"]

    wa_harm = fp["correct"] & ~wa["correct"]
    wo_harm = fp["correct"] & ~wo["correct"]

    for heuristic in HEURISTICS:
        for label in LABELS:
            mask = (
                (fp["heuristic"] == heuristic)
                & (fp["gold_label"] == label)
            )

            overlap_rows.append({
                "seed": seed,
                "heuristic": heuristic,
                "gold_label": label,
                "both": int((mask & wa_harm & wo_harm).sum()),
                "w8a8_only": int(
                    (mask & wa_harm & ~wo_harm).sum()
                ),
                "weight_only_only": int(
                    (mask & ~wa_harm & wo_harm).sum()
                ),
            })

    print(f"Seed {seed}: validated and analyzed")


tables = {
    "hans_accuracy_by_seed.csv": pd.DataFrame(accuracy_rows),
    "hans_flips_by_seed.csv": pd.DataFrame(flip_rows),
    "hans_margin_shifts_by_seed.csv": pd.DataFrame(margin_rows),
    "hans_harmful_flip_overlap.csv": pd.DataFrame(overlap_rows),
}

for filename, table in tables.items():
    table.to_csv(OUTPUT_DIR / filename, index=False)

accuracy = tables["hans_accuracy_by_seed.csv"]
flips = tables["hans_flips_by_seed.csv"]
margins = tables["hans_margin_shifts_by_seed.csv"]

accuracy_summary = (
    accuracy.groupby(["model", "heuristic", "gold_label"], as_index=False)
    .agg(
        mean_accuracy=("accuracy", "mean"),
        std_accuracy=("accuracy", "std"),
    )
)

accuracy_summary.to_csv(
    OUTPUT_DIR / "hans_accuracy_summary.csv", index=False
)

flip_summary = (
    flips.groupby(["model", "heuristic", "gold_label"], as_index=False)
    .agg(
        total_harmful=("harmful", "sum"),
        total_beneficial=("beneficial", "sum"),
        mean_accuracy_change=("net_accuracy_change", "mean"),
    )
)

flip_summary.to_csv(
    OUTPUT_DIR / "hans_flip_summary.csv", index=False
)

margin_summary = (
    margins.groupby(["model", "heuristic", "gold_label"], as_index=False)
    .agg(
        mean_shift=("mean_shift", "mean"),
        mean_positive_fraction=("positive_fraction", "mean"),
    )
)

margin_summary.to_csv(
    OUTPUT_DIR / "hans_margin_summary.csv", index=False
)

print("\n=== OVERALL HANS ACCURACY (%) ===")
overall = accuracy_summary[
    accuracy_summary["heuristic"] == "overall"
].copy()
overall["mean_accuracy"] *= 100
print(overall[["model", "mean_accuracy"]].round(4).to_string(index=False))

print("\n=== LEXICAL-OVERLAP NON-ENTAILMENT (%) ===")
lexical = accuracy_summary[
    (accuracy_summary["heuristic"] == "lexical_overlap")
    & (accuracy_summary["gold_label"] == "non-entailment")
].copy()
lexical["mean_accuracy"] *= 100
print(lexical[["model", "mean_accuracy"]].round(4).to_string(index=False))

print("\n=== LEXICAL-OVERLAP NON-ENTAILMENT MARGIN SHIFTS ===")
lexical_margin = margin_summary[
    (margin_summary["heuristic"] == "lexical_overlap")
    & (margin_summary["gold_label"] == "non-entailment")
]
print(
    lexical_margin[
        ["model", "mean_shift", "mean_positive_fraction"]
    ].round(4).to_string(index=False)
)

print(f"\nSaved analysis tables to {OUTPUT_DIR}")
