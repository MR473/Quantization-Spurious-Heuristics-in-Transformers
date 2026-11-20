from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("results/analysis")
OUT = Path("results/visualization/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "W8A8": "#D97732",
    "Weight-only": "#377EB8",
}

MODELS = ["W8A8", "Weight-only"]

mnli = pd.read_csv(BASE / "mnli_accuracy_by_seed.csv")
hans = pd.read_csv(BASE / "hans_accuracy_by_seed.csv")

records = []

for split, label in [
    ("matched", "MNLI matched"),
    ("mismatched", "MNLI mismatched"),
]:
    subset = mnli[mnli["split"] == split]
    pivot = subset.pivot(
        index="seed", columns="model", values="accuracy"
    )

    for model in MODELS:
        for seed, change in (
            (pivot[model] - pivot["FP32"]) * 100
        ).items():
            records.append({
                "metric": label,
                "seed": seed,
                "model": model,
                "change_pp": change,
            })

for heuristic, gold_label, label in [
    ("overall", "all", "HANS overall"),
    (
        "lexical_overlap",
        "non-entailment",
        "HANS lexical non-ent.",
    ),
]:
    subset = hans[
        (hans["heuristic"] == heuristic)
        & (hans["gold_label"] == gold_label)
    ]

    pivot = subset.pivot(
        index="seed", columns="model", values="accuracy"
    )

    for model in MODELS:
        for seed, change in (
            (pivot[model] - pivot["FP32"]) * 100
        ).items():
            records.append({
                "metric": label,
                "seed": seed,
                "model": model,
                "change_pp": change,
            })

changes = pd.DataFrame(records)

assert len(changes) == 24
assert not changes["change_pp"].isna().any()

changes.to_csv(
    OUT / "figure1_accuracy_changes_data.csv",
    index=False,
)

fig, axes = plt.subplots(
    1, 2, figsize=(10, 3.8),
    gridspec_kw={"width_ratios": [1.7, 1]},
)

panels = [
    (
        axes[0],
        ["MNLI matched", "MNLI mismatched", "HANS overall"],
        "A. Standard accuracy and HANS",
    ),
    (
        axes[1],
        ["HANS lexical non-ent."],
        "B. Lexical-overlap failures",
    ),
]

for ax, metrics, title in panels:
    x = np.arange(len(metrics))
    width = 0.32

    for i, model in enumerate(MODELS):
        means = []
        lows = []
        highs = []

        for metric in metrics:
            values = changes.loc[
                (changes["metric"] == metric)
                & (changes["model"] == model),
                "change_pp",
            ].to_numpy()

            assert len(values) == 3

            means.append(values.mean())
            lows.append(values.mean() - values.min())
            highs.append(values.max() - values.mean())

        positions = x + (i - 0.5) * width

        ax.bar(
            positions,
            means,
            width=width,
            color=COLORS[model],
            label=model,
            zorder=2,
        )

        ax.errorbar(
            positions,
            means,
            yerr=np.array([lows, highs]),
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            zorder=3,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy change vs FP32 (pp)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

axes[0].legend(
    loc="lower left",
    fontsize=9,
    frameon=False,
)





fig.tight_layout()

for extension in ["png", "svg"]:
    fig.savefig(
        OUT / f"figure1_accuracy_changes.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close(fig)

print("\n=== FIGURE 1 DATA (MEAN CHANGE IN PP) ===")
print(
    changes.groupby(["metric", "model"])["change_pp"]
    .mean()
    .unstack()
    .round(4)
    .to_string()
)

print(f"\nSaved Figure 1 to {OUT}")

# ============================================================
# Figure 2: HANS subgroup accuracy changes relative to FP32
# ============================================================

hans = pd.read_csv(BASE / "hans_accuracy_by_seed.csv")

heuristic_names = {
    "lexical_overlap": "Lexical overlap",
    "subsequence": "Subsequence",
    "constituent": "Constituent",
}

groups = [
    (heuristic, label)
    for heuristic in ["lexical_overlap", "subsequence", "constituent"]
    for label in ["entailment", "non-entailment"]
]

fig, ax = plt.subplots(figsize=(9.5, 4.5))

y = np.arange(len(groups))
bar_height = 0.34

for i, model in enumerate(MODELS):
    means = []
    lower_errors = []
    upper_errors = []

    for heuristic, label in groups:
        subset = hans[
            (hans["heuristic"] == heuristic)
            & (hans["gold_label"] == label)
        ]

        pivot = subset.pivot(
            index="seed",
            columns="model",
            values="accuracy",
        )

        changes = (
            (pivot[model] - pivot["FP32"]) * 100
        ).to_numpy()

        assert len(changes) == 3
        assert np.isfinite(changes).all()

        mean = changes.mean()

        means.append(mean)
        lower_errors.append(mean - changes.min())
        upper_errors.append(changes.max() - mean)

    positions = y + (i - 0.5) * bar_height

    ax.barh(
        positions,
        means,
        height=bar_height,
        color=COLORS[model],
        label=model,
        zorder=2,
    )

    ax.errorbar(
        means,
        positions,
        xerr=np.array([lower_errors, upper_errors]),
        fmt="none",
        ecolor="black",
        elinewidth=0.9,
        capsize=2.5,
        zorder=3,
    )

labels = [
    f"{heuristic_names[h]} — "
    f"{'Entailment' if label == 'entailment' else 'Non-entailment'}"
    for h, label in groups
]

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.invert_yaxis()

ax.axvline(0, color="black", linewidth=0.9)
ax.set_xlabel("Accuracy change relative to FP32 (percentage points)")


ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=2,
    frameon=False,
    fontsize=9,
)



fig.tight_layout(rect=[0, 0.10, 1, 1])

for extension in ["png", "svg"]:
    fig.savefig(
        OUT / f"figure2_hans_subgroup_changes.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close(fig)

print("\n=== FIGURE 2: HANS SUBGROUP CHANGES (PP) ===")

summary = []

for heuristic, label in groups:
    subset = hans[
        (hans["heuristic"] == heuristic)
        & (hans["gold_label"] == label)
    ]

    pivot = subset.pivot(
        index="seed",
        columns="model",
        values="accuracy",
    )

    summary.append({
        "heuristic": heuristic,
        "gold_label": label,
        "W8A8_change_pp": (
            (pivot["W8A8"] - pivot["FP32"]) * 100
        ).mean(),
        "Weight_only_change_pp": (
            (pivot["Weight-only"] - pivot["FP32"]) * 100
        ).mean(),
    })

summary = pd.DataFrame(summary)
summary.to_csv(
    OUT / "figure2_hans_subgroup_changes_data.csv",
    index=False,
)

print(summary.round(4).to_string(index=False))
print(f"\nSaved Figure 2 to {OUT}")
