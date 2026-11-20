import json
from pathlib import Path

import numpy as np
import pandas as pd

out_dir = Path("results/calibration")
thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
results = {}

for seed in [42, 123, 456]:
    path = out_dir / f"int8_seed{seed}_validation_matched.csv"
    df = pd.read_csv(path)

    probs = df[
        ["prob_entailment", "prob_neutral", "prob_contradiction"]
    ].to_numpy()

    labels = df["gold_label_id"].to_numpy()
    original_predictions = probs.argmax(axis=1)
    original_accuracy = np.mean(original_predictions == labels)

    scores = []

    for threshold in thresholds:
        predictions = np.where(
            probs[:, 0] >= threshold,
            0,
            np.where(probs[:, 1] >= probs[:, 2], 1, 2),
        )

        accuracy = np.mean(predictions == labels)
        scores.append((float(threshold), float(accuracy)))

    # Select the best accuracy; ties prefer threshold closest to 0.5.
    best_threshold, best_accuracy = min(
        scores,
        key=lambda x: (-x[1], abs(x[0] - 0.5)),
    )

    results[str(seed)] = {
        "threshold": best_threshold,
        "calibration_accuracy": best_accuracy,
        "original_argmax_accuracy": float(original_accuracy),
    }

    print(f"\nSeed {seed}")
    print(f"Original argmax accuracy: {original_accuracy:.4%}")
    print(f"Selected threshold:      {best_threshold:.2f}")
    print(f"Threshold accuracy:      {best_accuracy:.4%}")
    print(f"Change:                  {(best_accuracy-original_accuracy)*100:+.3f} pp")

output_path = out_dir / "selected_thresholds.json"
output_path.write_text(json.dumps(results, indent=2))

print(f"\nSaved: {output_path}")
