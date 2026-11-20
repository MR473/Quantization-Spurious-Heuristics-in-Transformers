import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path("results/calibration")
thresholds = json.loads(
    (root / "selected_thresholds.json").read_text()
)

rows = []

for seed in [42, 123, 456]:
    threshold = float(thresholds[str(seed)]["threshold"])

    path = root / f"int8_seed{seed}_validation_mismatched.csv"
    df = pd.read_csv(path)

    probs = df[
        ["prob_entailment", "prob_neutral", "prob_contradiction"]
    ].to_numpy()

    gold = df["gold_label_id"].to_numpy()
    original = probs.argmax(axis=1)

    calibrated = np.where(
        probs[:, 0] >= threshold,
        0,
        np.where(probs[:, 1] >= probs[:, 2], 1, 2),
    )

    # Verify the saved predictions match reconstructed argmax.
    assert np.array_equal(
        original, df["prediction_id"].to_numpy()
    )

    original_accuracy = np.mean(original == gold)
    calibrated_accuracy = np.mean(calibrated == gold)
    delta_pp = (calibrated_accuracy - original_accuracy) * 100

    print(f"\nSeed {seed} | Threshold: {threshold:.2f}")
    print(f"Original accuracy:   {original_accuracy:.4%}")
    print(f"Calibrated accuracy: {calibrated_accuracy:.4%}")
    print(f"Change:              {delta_pp:+.3f} pp")
    print(
        "Changed predictions:",
        int(np.sum(original != calibrated))
    )

    rows.append({
        "seed": seed,
        "threshold": threshold,
        "original_accuracy": float(original_accuracy),
        "calibrated_accuracy": float(calibrated_accuracy),
        "delta_pp": float(delta_pp),
        "changed_predictions": int(np.sum(original != calibrated)),
    })

output = root / "mnli_mismatched_calibration_evaluation.csv"
pd.DataFrame(rows).to_csv(output, index=False)

print(f"\nSaved: {output}")
print(
    "Average change:",
    f"{np.mean([r['delta_pp'] for r in rows]):+.3f} pp"
)
