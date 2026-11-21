#!/usr/bin/env python3
"""Publication-style HANS figures for the four-model DLBA comparison.

Run from the project root after scripts/analyze_four_model_hans.py:
    uv run python scripts/plot_four_model_hans.py

Source:
    results/analysis/four_model_comparison/
Optional runtime source:
    results/runtime_analysis/experiment3_openvino/three_seed_results.csv
Output:
    results/visualization/four_model_hans/

The script uses saved CSVs only. It does NOT rerun inference or modify analysis files.
Rerunning updates the plots in the output folder, not the source CSVs.
"""
from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SEEDS = [42, 123, 456]
MODELS = ["FP32", "W8A8", "Weight-only", "OpenVINO INT8"]
QUANTIZED = MODELS[1:]
COLORS = {
    "FP32": "#355C9A",            # navy blue
    "W8A8": "#C94F62",            # muted coral
    "Weight-only": "#DB783A",     # amber
    "OpenVINO FP32": "#8360B3",   # lavender
    "OpenVINO INT8": "#10947E",   # teal
}
HARMFUL = "#D45158"
BENEFICIAL = "#2D927E"
TEXT = "#233044"
GRID = "#DDE4EE"
GROUPS = [
    ("lexical_overlap", "entailment", "Lexical overlap · entailment"),
    ("lexical_overlap", "non-entailment", "Lexical overlap · non-entailment"),
    ("subsequence", "entailment", "Subsequence · entailment"),
    ("subsequence", "non-entailment", "Subsequence · non-entailment"),
    ("constituent", "entailment", "Constituent · entailment"),
    ("constituent", "non-entailment", "Constituent · non-entailment"),
]
SHORT_GROUPS = ["Lexical E", "Lexical NE", "Subsequence E", "Subsequence NE",
                "Constituent E", "Constituent NE"]


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1],
                   help="Repository root (default: parent of scripts/)")
    p.add_argument("--analysis-dir", type=Path, default=None,
                   help="Directory containing Experiment 3D CSVs")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory to save PNG and SVG figures")
    p.add_argument("--runtime-csv", type=Path, default=None,
                   help="Optional three_seed_results.csv for speed/accuracy plot")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "grid.color": GRID,
        "grid.alpha": .55,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "legend.frameon": False,
        "svg.fonttype": "none",  # editable labels when viewed in SVG editors
    })


def polish(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def save(fig, name, out, figs, caption, dpi):
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight", pad_inches=.18)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=.18)
    plt.close(fig)
    figs.append((name, caption))
    print(f"  ✓ {name}.png / .svg")


def filtered(df, h, gold):
    return df[(df["heuristic"] == h) & (df["gold_label"] == gold)]


def group_means(df, metric, models=MODELS):
    """Return 6 × N matrix of cross-seed means in stable subgroup/model order."""
    values = np.zeros((len(GROUPS), len(models)), dtype=float)
    for i, (h, lab, _) in enumerate(GROUPS):
        sub = filtered(df, h, lab)
        for j, model in enumerate(models):
            vals = sub.loc[sub["model"] == model, metric].to_numpy(dtype=float)
            if len(vals) != len(SEEDS):
                raise ValueError(f"Expected {len(SEEDS)} seeds for {model}, {h}, {lab}: found {len(vals)}")
            values[i, j] = vals.mean()
    return values


def read_tables(folder):
    names = {
        "accuracy": "hans_accuracy_by_seed.csv",
        "flips": "hans_flips_by_seed.csv",
        "margins": "hans_margin_shifts_by_seed.csv",
        "agreement": "hans_prediction_agreement_by_seed.csv",
        "overlap": "hans_harmful_flip_pairwise_overlap.csv",
    }
    missing = [str(folder / file) for file in names.values() if not (folder / file).is_file()]
    if missing:
        raise FileNotFoundError(
            "Four-model analysis CSVs missing:\n  " + "\n  ".join(missing) +
            "\nRun: uv run python scripts/analyze_four_model_hans.py"
        )
    d = {key: pd.read_csv(folder / file) for key, file in names.items()}
    needed = {
        "accuracy": {"seed", "model", "heuristic", "gold_label", "n", "accuracy"},
        "flips": {"seed", "model", "heuristic", "gold_label", "n", "harmful", "beneficial", "net_accuracy_change"},
        "margins": {"seed", "model", "heuristic", "gold_label", "n", "mean_shift", "positive_fraction"},
        "agreement": {"seed", "model", "heuristic", "gold_label", "n", "three_class_agreement", "binary_hans_agreement"},
        "overlap": {"seed", "model_a", "model_b", "heuristic", "gold_label", "both_harmful", "only_a_harmful", "only_b_harmful"},
    }
    for name, df in d.items():
        if absent := needed[name] - set(df.columns):
            raise ValueError(f"{names[name]} missing columns: {sorted(absent)}")
        if df.empty:
            raise ValueError(f"{names[name]} is empty")
        df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
        if set(df["seed"]) != set(SEEDS):
            raise ValueError(f"{names[name]}: expected seeds {SEEDS}, got {sorted(df['seed'].unique())}")
    expect_groups = {("overall", "all")} | {(h, gold) for h, gold, _ in GROUPS}
    for key in ("accuracy", "flips", "margins", "agreement"):
        df = d[key]
        target_models = MODELS if key in ("accuracy", "agreement") else QUANTIZED
        if set(df["model"]) != set(target_models):
            raise ValueError(f"{names[key]} models must be {target_models}; observed {df['model'].unique()}")
        if set(zip(df.heuristic, df.gold_label)) != expect_groups:
            raise ValueError(f"{names[key]}: incomplete HANS groups")
        keys = ["seed", "model", "heuristic", "gold_label"]
        if df.duplicated(keys).any():
            raise ValueError(f"{names[key]}: repeated seed/model/subgroup rows")
        n_expected = len(SEEDS) * len(target_models) * len(expect_groups)
        if len(df) != n_expected:
            raise ValueError(f"{names[key]}: expected {n_expected} rows; got {len(df)}")
        n_target = np.where(df["heuristic"].eq("overall"), 30000, 5000)
        if not np.array_equal(df["n"].to_numpy(), n_target):
            raise ValueError(f"{names[key]}: expected overall n=30000 and six n=5000 subgroups")

    if not d["accuracy"]["accuracy"].between(0, 1).all():
        raise ValueError("Accuracy must be a fraction in [0,1]")
    if not d["agreement"]["three_class_agreement"].between(0,1).all() or not d["agreement"]["binary_hans_agreement"].between(0,1).all():
        raise ValueError("Agreement rates must be in [0,1]")
    if not d["margins"]["positive_fraction"].between(0,1).all():
        raise ValueError("Positive margin fractions must be in [0,1]")
    flip = d["flips"]
    if not np.allclose((flip["beneficial"] - flip["harmful"]) / flip["n"], flip["net_accuracy_change"], atol=1e-8):
        raise ValueError("Flip counts inconsistent with net accuracy change")
    if (flip[["harmful", "beneficial"]] < 0).any().any():
        raise ValueError("Negative flip counts")

    for seed in SEEDS:
        for h, gold in expect_groups:
            fp = d["accuracy"].query("seed == @seed and model == 'FP32' and heuristic == @h and gold_label == @gold")["accuracy"].iloc[0]
            for model in QUANTIZED:
                quant = d["accuracy"].query("seed == @seed and model == @model and heuristic == @h and gold_label == @gold")["accuracy"].iloc[0]
                net = d["flips"].query("seed == @seed and model == @model and heuristic == @h and gold_label == @gold")["net_accuracy_change"].iloc[0]
                if not np.isclose(quant - fp, net, atol=1e-8):
                    raise ValueError(f"Accuracy/flip mismatch: seed={seed}, model={model}, {h} {gold}")

    overlap = d["overlap"]
    if set(zip(overlap.heuristic, overlap.gold_label)) != expect_groups:
        raise ValueError("Pairwise overlap missing HANS groups")
    from itertools import combinations
    expected_pairs = set(combinations(QUANTIZED, 2))
    if set(zip(overlap.model_a, overlap.model_b)) != expected_pairs:
        raise ValueError("Overlap file missing a model pair")
    if overlap.duplicated(["seed", "model_a", "model_b", "heuristic", "gold_label"]).any():
        raise ValueError("Duplicate overlap rows")
    if len(overlap) != len(SEEDS) * len(expected_pairs) * len(expect_groups):
        raise ValueError("Incomplete pairwise overlap rows")
    return d


def main_accuracy(d, out, index, dpi):
    a = filtered(d["accuracy"], "overall", "all")
    g = a.groupby("model")["accuracy"].agg(["mean", "std"]).reindex(MODELS) * 100
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(MODELS))
    ax.bar(x, g["mean"], color=[COLORS[m] for m in MODELS], width=.64, zorder=3)
    ax.errorbar(x, g["mean"], yerr=g["std"], fmt="none", ecolor=TEXT,
                elinewidth=1.2, capsize=5, capthick=1.2, zorder=4)
    for i, val in enumerate(g["mean"]):
        ax.text(i, val+5.1, f"{val:.2f}%", ha="center", weight="bold", fontsize=10)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel("HANS accuracy (%)")
    ax.set_title("Overall HANS accuracy across three seeds", loc="left", pad=16)
    ax.set_ylim(0, 79)
    polish(ax)
    fig.text(.12, .01, "Bars: seed means · Error bars: ±1 seed standard deviation (not confidence intervals)", color="#617187", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, "01_overall_hans_accuracy", out, index, "Overall HANS accuracy; mean ± 1 SD across three fine-tuning seeds.", dpi)


def subgroup_heatmap(d, out, index, dpi):
    a = group_means(d["accuracy"], "accuracy") * 100
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    im = ax.imshow(a, cmap="viridis", vmin=0, vmax=100, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_yticks(range(len(GROUPS)), [name for _, _, name in GROUPS])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=8)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            val = a[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color=TEXT if val > 70 else "white", fontweight="bold", fontsize=10)
    ax.set_title("HANS accuracy by heuristic and gold label", loc="left", pad=22)
    cb = fig.colorbar(im, ax=ax, shrink=.79, pad=.02)
    cb.set_label("Accuracy (%)")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    save(fig, "02_subgroup_accuracy_heatmap", out, index, "Six HANS subgroups; mean accuracy across seeds for all four models.", dpi)


def accuracy_delta(d, out, index, dpi):
    a = group_means(d["accuracy"], "accuracy") * 100
    delta = a[:,1:] - a[:,[0]]
    fig, ax = plt.subplots(figsize=(11.4, 6.0))
    y = np.arange(len(GROUPS))
    dh = .22
    for i, model in enumerate(QUANTIZED):
        ax.barh(y + (i-1)*dh, delta[:,i], height=dh*.92, color=COLORS[model], label=model, zorder=3)
    ax.axvline(0, color=TEXT, linewidth=1)
    ax.set_yticks(y, [name for _, _, name in GROUPS])
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change vs FP32 (percentage points)")
    ax.set_title("Where quantization changes HANS accuracy", loc="left", pad=15)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1,-.26))
    polish(ax, "x")
    fig.tight_layout(rect=(0,.05,1,1))
    save(fig, "03_subgroup_accuracy_change", out, index, "Per-subgroup mean accuracy change in percentage points vs seed-matched FP32.", dpi)


def seed_consistency(d, out, index, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for ax, (h, gold, title) in zip(axes, [("overall", "all", "Overall HANS"),
                                         ("lexical_overlap", "non-entailment", "Lexical-overlap non-entailment")]):
        a = filtered(d["accuracy"], h, gold)
        for model in MODELS:
            q = a[a.model == model].sort_values("seed")
            ax.plot(q.seed, q.accuracy * 100, "o-", color=COLORS[model], linewidth=2.1,
                    markersize=6.2, label=model)
        ax.set_xticks(SEEDS)
        ax.set_xlabel("Fine-tuning seed")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title, loc="left", pad=10)
        polish(ax)
    fig.suptitle("Do the effects repeat across seeds?", x=.06, ha="left", fontsize=14, fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(.5,-.02))
    fig.tight_layout(rect=(0,.065,1,.94), w_pad=3)
    save(fig, "04_seed_consistency", out, index, "Overall and lexical non-entailment accuracy by seed.", dpi)


def flip_mirrors(d, out, index, dpi):
    fig, axes = plt.subplots(3, 1, figsize=(11, 11.3), sharey=True)
    y = np.arange(len(GROUPS))
    for ax, model in zip(axes, QUANTIZED):
        q = d["flips"].query("model == @model")
        h = [int(filtered(q, g, lab).harmful.sum()) for g, lab, _ in GROUPS]
        b = [int(filtered(q, g, lab).beneficial.sum()) for g, lab, _ in GROUPS]
        ax.barh(y, -np.array(h), height=.58, color=HARMFUL, label="Harmful", zorder=3)
        ax.barh(y, b, height=.58, color=BENEFICIAL, label="Beneficial", zorder=3)
        ax.axvline(0, color=TEXT, linewidth=1.15)
        ax.set_yticks(y, [name for _, _, name in GROUPS])
        ax.invert_yaxis()
        ax.set_title(model, loc="left", pad=8, color=COLORS[model])
        ax.set_xlabel("Pooled flip count across three seed-runs (negative = harmful)")
        polish(ax, "x")
        # Intentional independent horizontal scales keep small counts readable.
        mx = max(8, max(h))
        bx = max(8, max(b))
        ax.set_xlim(-mx*1.16, bx*1.25)
        for k in range(len(y)):
            if h[k]:
                ax.text(-h[k]-mx*.017, y[k], str(h[k]), ha="right", va="center", fontsize=8)
            if b[k]:
                ax.text(b[k]+bx*.025, y[k], str(b[k]), ha="left", va="center", fontsize=8)
    fig.suptitle("Harmful versus beneficial prediction flips", x=.09, ha="left", fontsize=15, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=HARMFUL, label="Harmful: FP32 correct → quantized wrong"),
                        Patch(facecolor=BENEFICIAL, label="Beneficial: FP32 wrong → quantized correct")],
               loc="lower center", ncol=2, bbox_to_anchor=(.53,-.015))
    fig.text(.095, .022, "Counts pool seed–example pairs (3 checkpoints × 30k HANS examples), not independent samples. Each panel uses its own x-scale.",
             fontsize=8.4, color="#617187")
    fig.tight_layout(rect=(0,.07,1,.966), h_pad=2.0)
    save(fig, "05_harmful_beneficial_flips", out, index, "Harmful/beneficial flips vs FP32 in all six subgroups; three seed-runs pooled.", dpi)


def overall_flips(d, out, index, dpi):
    q = filtered(d["flips"], "overall", "all").groupby("model")[["harmful","beneficial"]].sum().reindex(QUANTIZED)
    fig, ax = plt.subplots(figsize=(9.2,4.7))
    x = np.arange(len(QUANTIZED)); width=.32
    harmful_bars = ax.bar(x-width/2, q.harmful, width, color=HARMFUL, label="Harmful", zorder=3)
    benefit_bars = ax.bar(x+width/2, q.beneficial, width, color=BENEFICIAL, label="Beneficial", zorder=3)
    for bars in (harmful_bars, benefit_bars):
        ax.bar_label(bars, padding=3, fontsize=9, color=TEXT, fmt="%.0f")
    ax.set_xticks(x, QUANTIZED)
    ax.set_ylabel("Flip count (pooled across seed-runs)")
    ax.set_title("Total HANS prediction flips vs FP32", loc="left", pad=15)
    ax.legend(ncol=2)
    polish(ax)
    ymax = max(q.harmful.max(), q.beneficial.max())
    ax.set_ylim(0, ymax*1.23 if ymax else 10)
    fig.text(.12,.01,"Pooled counts are over 90,000 seed–example evaluations, not 90,000 independent HANS sentences.", fontsize=8.5, color="#617187")
    fig.tight_layout(rect=(0,.035,1,1))
    save(fig, "06_overall_flip_totals", out, index, "All-HANS harmful and beneficial flip counts pooled across three seeds.", dpi)


def margin_shifts(d, out, index, dpi):
    a = group_means(d["margins"], "mean_shift", QUANTIZED)
    y = np.arange(len(GROUPS))
    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    height = .23
    for i, model in enumerate(QUANTIZED):
        ax.barh(y+(i-1)*height, a[:,i], height=height*.94, color=COLORS[model], label=model, zorder=3)
    ax.axvline(0, color=TEXT, linewidth=1)
    ax.set_yticks(y, [name for _,_,name in GROUPS]); ax.invert_yaxis()
    ax.set_xlabel("Mean change in entailment margin vs FP32")
    ax.set_title("Does quantization move predictions toward entailment?", loc="left", pad=15)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1,-.26))
    polish(ax,"x")
    fig.text(.11,.02,"Margin = P(entailment) − max[P(neutral), P(contradiction)]. Positive shift favors entailment in the score, not necessarily the final label.", fontsize=8.5, color="#617187")
    fig.tight_layout(rect=(0,.065,1,1))
    save(fig, "07_entailment_margin_shifts", out, index, "Mean change in entailment margin; negative/positive values retain their signs.", dpi)


def positive_margin_fraction(d, out, index, dpi):
    a = group_means(d["margins"], "positive_fraction", QUANTIZED) * 100
    fig, ax = plt.subplots(figsize=(10.8, 5.7))
    y=np.arange(len(GROUPS)); height=.22
    for i, model in enumerate(QUANTIZED):
        ax.barh(y+(i-1)*height,a[:,i], height=height*.94,color=COLORS[model],label=model,zorder=3)
    ax.axvline(50, color="#7D8797", linestyle="--", linewidth=1, alpha=.85)
    ax.set_yticks(y,[name for _,_,name in GROUPS]); ax.invert_yaxis()
    ax.set_xlim(0,100)
    ax.set_xlabel("Examples with a positive entailment-margin shift (%)")
    ax.set_title("How often does the entailment margin increase?",loc="left",pad=15)
    ax.legend(ncol=3,loc="lower right",bbox_to_anchor=(1,-.26))
    polish(ax,"x")
    fig.tight_layout(rect=(0,.06,1,1))
    save(fig,"08_positive_margin_fraction",out,index,"Across-seed mean percentage of subgroup examples with positive entailment-margin change.",dpi)


def agreement_plot(d, out, index, dpi):
    q=filtered(d["agreement"],"overall","all")
    q=q[q.model.isin(QUANTIZED)].groupby("model")[["three_class_agreement","binary_hans_agreement"]].mean().reindex(QUANTIZED)*100
    fig,ax=plt.subplots(figsize=(9.8,5))
    y=np.arange(len(QUANTIZED)); height=.31
    bars1=ax.barh(y-height/2,q.three_class_agreement, height,color="#7899BD",label="Three-class (entailment/neutral/contradiction)",zorder=3)
    bars2=ax.barh(y+height/2,q.binary_hans_agreement,height,color="#2E9C8F",label="Binary HANS (entailment/non-entailment)",zorder=3)
    for bars in (bars1,bars2):
        for b in bars:
            ax.text(b.get_width()+.3,b.get_y()+b.get_height()/2,f"{b.get_width():.2f}%",va="center",fontsize=9)
    ax.set_yticks(y,QUANTIZED);ax.invert_yaxis()
    ax.set_xlim(0,105)
    ax.set_xlabel("Agreement with FP32 (%)")
    ax.set_title("Are the predicted classes preserved?",loc="left",pad=15)
    ax.legend(loc="lower right",bbox_to_anchor=(1,-.23))
    polish(ax,"x")
    fig.tight_layout(rect=(0,.04,1,1))
    save(fig,"09_prediction_agreement",out,index,"Three-class versus binary HANS agreement with FP32, averaged across seeds.",dpi)


def flip_overlap_heatmaps(d, out, index, dpi):
    o = d["overlap"]
    fig = plt.figure(figsize=(13.2, 6.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .055],
                          left=.16, right=.93, bottom=.23, top=.79, wspace=.48)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    for ax, (h, lab, title) in zip(axes, [
        ("overall", "all", "All HANS examples"),
        ("lexical_overlap", "non-entailment", "Lexical-overlap non-entailment"),
    ]):
        q = filtered(o, h, lab)
        matrix = np.full((len(QUANTIZED), len(QUANTIZED)), np.nan)
        shared = {}
        pairs = q.groupby(["model_a", "model_b"])[
            ["both_harmful", "only_a_harmful", "only_b_harmful"]
        ].sum()
        for (a, b), r in pairs.iterrows():
            both = float(r["both_harmful"])
            union = both + float(r["only_a_harmful"]) + float(r["only_b_harmful"])
            jaccard = both / union if union else np.nan
            i, j = QUANTIZED.index(a), QUANTIZED.index(b)
            matrix[i, j] = matrix[j, i] = jaccard
            shared[i, j] = shared[j, i] = int(both)
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap="YlGnBu",
                       vmin=0, vmax=1, aspect="equal")
        labels = ["W8A8", "Weight-only", "OpenVINO INT8"]
        ax.set_xticks(range(len(QUANTIZED)), labels, rotation=18, ha="right")
        ax.set_yticks(range(len(QUANTIZED)), labels)
        ax.set_title(title, pad=16, fontsize=12)
        for i in range(len(QUANTIZED)):
            for j in range(len(QUANTIZED)):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="#808A99")
                elif np.isfinite(matrix[i, j]):
                    value = matrix[i, j]
                    ax.text(j, i, f"{value:.2f}\n({shared[i, j]} shared)",
                            ha="center", va="center", color="white" if value > .5 else TEXT,
                            fontsize=8.5, fontweight="bold")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center", color=TEXT)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Jaccard: both harmful / harmful in either", rotation=90, labelpad=13)
    fig.suptitle("Overlap of harmful predictions among INT8 implementations",
                 x=.105, ha="left", fontsize=14, fontweight="bold", y=.97)
    fig.text(.16, .055,
             "Same seed–example pairs across models; parentheses show pooled shared harmful flips over three seed-runs.",
             fontsize=8.7, color="#617187")
    save(fig, "10_harmful_flip_overlap", out, index,
         "Pairwise harmful-flip Jaccard overlap; diagonal excluded.", dpi)


def runtime_tradeoff(runtime_csv, d, out, index, dpi):
    if not runtime_csv.is_file():
        print("  – skipped speed/accuracy plot: runtime CSV not found")
        return
    raw=pd.read_csv(runtime_csv)
    required={"seed","variant","throughput_examples_per_second","hans_accuracy_pct"}
    if missing:=required-set(raw.columns):
        warnings.warn(f"Skipping runtime tradeoff: missing {sorted(missing)} in {runtime_csv}")
        return
    variants={
        "torch_fp32":"FP32",
        "torchao_weight_only":"Weight-only",
        "openvino_fp32":"OpenVINO FP32",
        "openvino_int8_sym":"OpenVINO INT8",
    }
    raw=raw[raw.variant.isin(variants)].copy()
    if set(raw.variant)!=set(variants) or set(raw.seed)!=set(SEEDS) or raw.duplicated(["seed","variant"]).any():
        warnings.warn("Skipping runtime tradeoff: incomplete or duplicated four-variant / three-seed results")
        return
    acc=raw.copy()
    acc["name"]=acc.variant.map(variants)
    avg=acc.groupby("name")[["throughput_examples_per_second","hans_accuracy_pct"]].agg(["mean","std"])
    fig,ax=plt.subplots(figsize=(9.6,5.6))
    for name in variants.values():
        x=avg.loc[name,("hans_accuracy_pct","mean")]
        y=avg.loc[name,("throughput_examples_per_second","mean")]
        xerr=avg.loc[name,("hans_accuracy_pct","std")]
        yerr=avg.loc[name,("throughput_examples_per_second","std")]
        ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt="o",markersize=11,
                    markerfacecolor=COLORS[name],markeredgecolor="white",markeredgewidth=1.5,
                    ecolor=COLORS[name],capsize=4,linewidth=1.2,zorder=3)
        offsets = {
            "FP32": (12, -13),
            "Weight-only": (-12, -18),
            "OpenVINO FP32": (-12, 13),
            "OpenVINO INT8": (12, 10),
        }
        offset = offsets[name]
        ax.annotate(name, (x, y), xytext=offset, textcoords="offset points",
                    ha="left" if offset[0] > 0 else "right", fontsize=9,
                    fontweight="bold")
    ax.set_xlabel("Mean HANS accuracy (%)")
    ax.set_ylabel("Mean CPU throughput (examples/second)")
    ax.set_title("Speed–accuracy tradeoff on the same CPU",loc="left",pad=15)
    polish(ax)
    fig.text(.1,.015,"Bars: ±1 SD across three seeds (not confidence intervals). W8A8 CUDA results excluded: different inference device and method.",fontsize=8.2,color="#617187")
    fig.tight_layout(rect=(0,.04,1,1))
    save(fig,"11_cpu_speed_accuracy_tradeoff",out,index,"CPU-only comparison from Experiment 3 runtime benchmarks; all variants use seed-matched full HANS runs.",dpi)


def write_index(out, index, inputs, d):
    rows=[]
    for i,(name,desc) in enumerate(index,1):
        rows.append({"figure":name,"png":f"{name}.png","svg":f"{name}.svg","caption":desc})
    with (out/"figure_index.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["figure","png","svg","caption"]);w.writeheader();w.writerows(rows)
    overall=filtered(d["accuracy"],"overall","all").groupby("model")["accuracy"].agg(["mean","std"]).reindex(MODELS)
    summary=overall.rename(columns={"mean":"mean_hans_accuracy","std":"sd_hans_accuracy"})
    flip=filtered(d["flips"],"overall","all").groupby("model")[["harmful","beneficial"]].sum().reindex(QUANTIZED)
    summary=summary.join(flip.rename(columns={"harmful":"pooled_harmful_flips","beneficial":"pooled_beneficial_flips"}))
    summary.to_csv(out/"plot_summary.csv")
    text=["# DLBA Experiment 3D — four-model HANS visuals","",f"**Source:** `{inputs}`", "", "The figures use original Experiment 3D CSVs; no model inference is rerun.",
          "Averages and standard deviations are computed over three trained seeds (42, 123, 456).",
          "Whiskers show SD, *not* statistical confidence intervals.",
          "Pooled flip counts are across seed–example pairs, not independent test sentences.",
          "Flip definitions: harmful = FP32 correct and quantized incorrect; beneficial = FP32 incorrect and quantized correct.",
          "Entailment margin = P(entailment) − max(P(neutral), P(contradiction).",
          "OpenVINO INT8 here means NNCF weight compression + OpenVINO CPU inference; W8A8 was a distinct TorchAO implementation.",
          "The optional runtime tradeoff plot excludes W8A8 because its original throughput results were collected on CUDA.",
          "","## Figures", ""]
    text += [f"- **{name}** — {desc}" for name, desc in index]
    (out/"README_FIGURES.md").write_text("\n".join(text)+"\n",encoding="utf-8")


def main():
    a=args_parser()
    repo=a.repo_root.expanduser().resolve()
    folder=(a.analysis_dir or (repo/"results/analysis/four_model_comparison")).expanduser().resolve()
    out=(a.out_dir or (repo/"results/visualization/four_model_hans")).expanduser().resolve()
    runtime=(a.runtime_csv or (repo/"results/runtime_analysis/experiment3_openvino/three_seed_results.csv")).expanduser().resolve()
    if a.dpi<72 or a.dpi>600:
        raise ValueError("--dpi must be between 72 and 600")
    style()
    print(f"Reading four-model HANS tables from: {folder}")
    d=read_tables(folder)
    out.mkdir(parents=True,exist_ok=True)
    index=[]
    print(f"Saving figures to: {out}")
    main_accuracy(d,out,index,a.dpi)
    subgroup_heatmap(d,out,index,a.dpi)
    accuracy_delta(d,out,index,a.dpi)
    seed_consistency(d,out,index,a.dpi)
    flip_mirrors(d,out,index,a.dpi)
    overall_flips(d,out,index,a.dpi)
    margin_shifts(d,out,index,a.dpi)
    positive_margin_fraction(d,out,index,a.dpi)
    agreement_plot(d,out,index,a.dpi)
    flip_overlap_heatmaps(d,out,index,a.dpi)
    runtime_tradeoff(runtime,d,out,index,a.dpi)
    write_index(out,index,folder,d)
    print(f"\nDone: {len(index)} figures, both PNG (300 dpi default) and editable SVG.")
    print(f"Index: {out/'README_FIGURES.md'}")
    print("No existing model predictions or analysis tables were changed.")


if __name__=="__main__":
    main()
