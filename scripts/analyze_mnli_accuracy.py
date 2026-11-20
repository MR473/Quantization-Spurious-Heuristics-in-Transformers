import json
from pathlib import Path

import pandas as pd

SEEDS = [42, 123, 456]
SPLITS = ["matched", "mismatched"]

RESULTS_DIR = Path("results/mnli")
OUTPUT_DIR = Path("results/analysis")


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    return json.loads(path.read_text())


def main():
    records = []

    for seed in SEEDS:
        fp32 = load_json(RESULTS_DIR / f"fp32_seed{seed}_results.json")
        w8a8 = load_json(RESULTS_DIR / f"int8_seed{seed}_results.json")

        assert w8a8["seed"] == seed

        for split in SPLITS:
            weight_only = load_json(
                RESULTS_DIR
                / "weight_only_int8"
                / f"weight_only_int8_seed{seed}_validation_{split}_summary.json"
            )

            assert weight_only["seed"] == seed
            assert weight_only["split"] == f"validation_{split}"

            accuracies = {
                "FP32": fp32[split][f"mnli_{split}_accuracy"],
                "W8A8": w8a8[f"{split}_accuracy"],
                "Weight-only": weight_only["accuracy"],
            }

            for model, accuracy in accuracies.items():
                assert 0 <= accuracy <= 1
                records.append({
                    "seed": seed,
                    "split": split,
                    "model": model,
                    "accuracy": accuracy,
                })

    by_seed = pd.DataFrame(records)
    assert len(by_seed) == 18

    summary = (
        by_seed.groupby(["split", "model"], sort=False)["accuracy"]
        .agg(mean_accuracy="mean", std_accuracy="std")
        .reset_index()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_seed.to_csv(OUTPUT_DIR / "mnli_accuracy_by_seed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "mnli_accuracy_summary.csv", index=False)

    print("MNLI accuracy analysis completed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
