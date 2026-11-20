import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("results/calibration")
thresholds = json.loads((root / "selected_thresholds.json").read_text())

rows = []

for seed in [42, 123, 456]:
    threshold = thresholds[str(seed)]["threshold"]
    df = pd.read_csv(f"results/hans/int8_seed{seed}_predictions.csv")

    probs = df[
        ["prob_entailment", "prob_neutral", "prob_contradiction"]
    ].to_numpy()

    original = probs.argmax(axis=1)

    calibrated = np.where(
        probs[:, 0] >= threshold,
        0,
        np.where(probs[:, 1] >= probs[:, 2], 1, 2),
    )

    # HANS maps neutral and contradiction to non-entailment.
    original_hans = np.where(original == 0, "entailment", "non-entailment")
    calibrated_hans = np.where(calibrated == 0, "entailment", "non-entailment")

    gold = df["gold_label"].to_numpy()

    groups = {
        "overall": np.ones(len(df), dtype=bool),
        "entailment": gold == "entailment",
        "non_entailment": gold == "non-entailment",
        "lexical_non_entailment": (
            (df["heuristic"] == "lexical_overlap")
            & (gold == "non-entailment")
        ).to_numpy(),
    }

    print(f"\nSeed {seed} | threshold {threshold:.2f}")

    for group, mask in groups.items():
        before = np.mean(original_hans[mask] == gold[mask])
        after = np.mean(calibrated_hans[mask] == gold[mask])
        delta = (after - before) * 100

        print(
            f"{group:25s} "
            f"{before*100:7.2f}% -> {after*100:7.2f}% "
            f"({delta:+.2f} pp)"
        )

        rows.append({
            "seed": seed,
            "threshold": threshold,
            "group": group,
            "original_accuracy": before,
            "calibrated_accuracy": after,
            "delta_pp": delta,
            "n": int(mask.sum()),
        })

    print("Changed HANS binary predictions:", int(
        np.sum(original_hans != calibrated_hans)
    ))

output = root / "hans_calibration_evaluation.csv"
pd.DataFrame(rows).to_csv(output, index=False)
print(f"\nSaved: {output}")
