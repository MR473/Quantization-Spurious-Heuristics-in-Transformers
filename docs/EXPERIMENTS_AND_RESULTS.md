# Experiments and Results

## 1. Research objective

We investigate whether post-training INT8 quantization changes the sensitivity of fine-tuned BERT models to spurious linguistic heuristics in natural language inference (NLI), even when accuracy on the standard MNLI validation splits remains approximately stable. HANS is used as a targeted diagnostic rather than as a direct measurement of internal feature use.

## 2. Experimental setup

- **Base model:** `google-bert/bert-base-uncased`, fine-tuned for three-way MNLI classification (entailment, neutral, contradiction).
- **Fine-tuning:** three epochs, learning rate `2e-5`, batch size 32, maximum sequence length 128.
- **Seeds:** 42, 123, 456. Each seed has its own fine-tuned FP32 checkpoint at `models/fp32/seed{seed}/final` (checkpoints are excluded from Git).
- **MNLI validation:** matched (9,815 examples) and mismatched (9,832 examples).
- **HANS:** 30,000 examples, equally divided into six 5,000-example heuristic/label groups: lexical overlap, subsequence, and constituent, each with entailment and non-entailment gold labels.
- **Configurations:** FP32; dynamic activation INT8 plus weight INT8 (**W8A8**, TorchAO `Int8DynamicActivationInt8WeightConfig`); and **weight-only INT8** (TorchAO `Int8WeightOnlyConfig`). Both quantized configurations target 74 linear layers and were evaluated on CPU.
- **Hardware and framework:** experiments used an Apple Silicon MacBook with PyTorch and TorchAO. The original FP32 HANS evaluations used Apple MPS; an additional CPU FP32 check was performed.

### Label mapping

The model predicts one of three MNLI classes: `0 = entailment`, `1 = neutral`, `2 = contradiction`. For HANS, the **three-class argmax prediction** is mapped to `entailment` if it is class 0 and `non-entailment` otherwise. This is not the same as making a binary decision by comparing `P(entailment)` against `P(neutral) + P(contradiction)`.

## 3. Main results

All values below are arithmetic means across the three seeds. Accuracy differences are expressed in **percentage points (pp)** relative to the corresponding FP32 results.

| Evaluation | FP32 | W8A8 | Weight-only INT8 | W8A8 change | Weight-only change |
|---|---:|---:|---:|---:|---:|
| MNLI matched | 84.6358% | 84.7444% | 84.6561% | +0.1087 pp | +0.0204 pp |
| MNLI mismatched | 84.8454% | 84.8895% | 84.8827% | +0.0441 pp | +0.0373 pp |
| HANS overall | 57.8989% | 57.0333% | 57.6967% | −0.8656 pp | −0.2022 pp |
| HANS lexical-overlap non-entailment | 35.7600% | 31.5533% | 34.8267% | −4.2067 pp | −0.9333 pp |

MNLI accuracy changes were small in both quantized configurations. The largest HANS subgroup decline occurred on **lexical-overlap non-entailment**, where W8A8 lost about 4.21 pp and weight-only INT8 lost about 0.93 pp relative to FP32. This is consistent with greater sensitivity to the lexical-overlap heuristic after W8A8 quantization, but does not prove a change in the model's internal reliance on that feature.

### HANS subgroup changes

| Heuristic | Gold label | W8A8 change vs FP32 | Weight-only change vs FP32 |
|---|---|---:|---:|
| Lexical overlap | Entailment | +0.5000 pp | +0.0400 pp |
| Lexical overlap | Non-entailment | −4.2067 pp | −0.9333 pp |
| Subsequence | Entailment | +0.1333 pp | +0.0067 pp |
| Subsequence | Non-entailment | −0.9867 pp | −0.2333 pp |
| Constituent | Entailment | +0.0867 pp | +0.0267 pp |
| Constituent | Non-entailment | −0.7200 pp | −0.1200 pp |

The FP32 non-entailment baselines were 35.76% for lexical overlap, 5.52% for subsequence, and 10.32% for constituent. The very low latter two baselines create **floor effects**: accuracy has little room to decline further, so subgroup change magnitudes should not be interpreted as directly comparable measures of shortcut dependence.

## 4. Prediction-flip analysis

A **harmful flip** is a HANS example predicted correctly by FP32 but incorrectly by the quantized configuration. A **beneficial flip** is an example predicted incorrectly by FP32 but correctly after quantization. The following counts refer only to the **lexical-overlap non-entailment** subgroup (5,000 examples per seed).

| Seed | W8A8 harmful | W8A8 beneficial | Weight-only harmful | Weight-only beneficial |
|---|---:|---:|---:|---:|
| 42 | 211 | 32 | 45 | 14 |
| 123 | 304 | 11 | 81 | 6 |
| 456 | 171 | 12 | 40 | 6 |
| **Total across seed evaluations** | **686** | **55** | **166** | **26** |

The net changes were −631 correct predictions for W8A8 and −140 for weight-only INT8 across the 15,000 seed-example evaluations. These are **not 15,000 independent examples**: the same 5,000 HANS items were evaluated with three independently fine-tuned checkpoints.

For harmful flips in this subgroup, 147 seed-example evaluations were harmful under both quantization methods, 539 were harmful only under W8A8, and 19 only under weight-only INT8. Thus 147/166 (88.6%) of weight-only harmful flips also occurred with W8A8, whereas 539/686 (78.6%) of W8A8 harmful flips were not harmful under weight-only INT8.

## 5. Entailment-margin analysis

For the original three-way MNLI output probabilities, define the entailment margin as

\[
m = P(\text{entailment}) - \max(P(\text{neutral}), P(\text{contradiction})).
\]

The shift is the quantized margin minus the FP32 margin on the same example. A positive shift means the model's **three-class entailment margin** increased; it is not a binary margin based on the combined non-entailment probability.

Mean shifts on HANS **non-entailment** examples across the three seeds:

| Heuristic | W8A8 mean shift | Weight-only mean shift | W8A8 shifts positive | Weight-only shifts positive |
|---|---:|---:|---:|---:|
| Lexical overlap | +0.0677 | +0.0152 | 82.32% | 75.09% |
| Subsequence | +0.0192 | +0.0046 | 53.82% | 53.91% |
| Constituent | +0.0062 | +0.0018 | 45.92% | 52.67% |

The lexical-overlap subgroup has the clearest positive margin shift under W8A8. Positive mean shifts for the other groups should not be described as a majority of individual examples shifting toward entailment, particularly for constituent examples under W8A8.

## 6. FP32 device validation

The original FP32 HANS predictions were generated using Apple MPS, while the additional quantized evaluations were run on CPU. To check whether this device difference could affect the class-label comparisons, FP32 checkpoints were also evaluated on CPU. **All 90,000 three-class predicted labels** (30,000 HANS examples × three seeds) matched the previously saved MPS FP32 predictions.

This validates **predicted-label equivalence for these runs**, not bit-for-bit equality of logits or probabilities across devices.

## 7. Supplementary decision-threshold experiment

A separate exploratory experiment selected a per-seed entailment probability threshold on the **W8A8 MNLI matched validation split**, then applied the selected threshold to MNLI mismatched and HANS. The selected thresholds were 0.25, 0.35, and 0.24 for seeds 42, 123, and 456 respectively.

The rule predicts entailment if `P(entailment) >= threshold`; otherwise it selects whichever of neutral and contradiction has higher probability. This changes the decision rule; it is **not temperature scaling** and does not recalibrate the probability distribution itself.

Relative to the original W8A8 argmax rule, the mean MNLI mismatched accuracy changed by approximately **+0.08 pp**, HANS overall by **−2.37 pp**, and HANS lexical-overlap non-entailment by **−8.95 pp**. Because the threshold was selected on MNLI matched, this is an exploratory cross-dataset transfer test rather than a controlled comparison of quantization methods.

## 8. Reproduction and file guide

Run commands from the **repository root** with the Python environment and dependencies in `pyproject.toml`. The relevant checkpoints must already exist at `models/fp32/seed{seed}/final`. These checkpoints are not committed to Git.

### Quantized evaluations

For each seed in `42`, `123`, `456`:

```bash
python scripts/evaluate_weight_only_int8.py --seed 42
python scripts/evaluate_weight_only_mnli.py --seed 42 --split validation_matched
python scripts/evaluate_weight_only_mnli.py --seed 42 --split validation_mismatched
```

Repeat for seeds 123 and 456. The MNLI weight-only script refuses to overwrite existing prediction or summary files. HANS uses the evaluation file from the original HANS repository. These commands are for **reproducing the additional weight-only evaluations**; the original FP32 training and W8A8 inference pipelines are in the project's existing notebooks and outputs.

### CPU FP32 check

```bash
python scripts/validate_fp32_cpu.py --seed 42
```

Repeat for seeds 123 and 456. Compare the generated validation outputs with the original saved FP32 predictions as described above.

### Consolidated HANS analysis and figures

```bash
python scripts/analyze_quantization_effects.py
python scripts/plot_paper_figures.py
```

`analyze_quantization_effects.py` reads the existing FP32, W8A8, and weight-only HANS prediction CSVs, validates their alignment, and writes results to `results/analysis/`. The plotting script also requires **`results/analysis/mnli_accuracy_by_seed.csv`**, which was consolidated from the MNLI evaluation outputs in an ad hoc analysis step and is included with the saved results. A standalone MNLI consolidation script has **not yet been added**, so the current plotting pipeline is not fully end-to-end reproducible from raw prediction files alone.

### Supplementary threshold analysis

For each seed, first generate W8A8 MNLI prediction probabilities on both validation splits:

```bash
python scripts/generate_mnli_calibration.py --seed 42 --split validation_matched
python scripts/generate_mnli_calibration.py --seed 42 --split validation_mismatched
```

After generating both splits for all three seeds:

```bash
python scripts/select_calibration_thresholds.py
python scripts/evaluate_calibrated_mnli.py
python scripts/evaluate_calibrated_hans.py
```

The threshold evaluation scripts consume `results/calibration/selected_thresholds.json` and saved W8A8 predictions; they do not re-train the model.

### Key saved paths

| Path | Purpose |
|---|---|
| `results/hans/fp32_seed{seed}_predictions.csv` | FP32 HANS reference predictions |
| `results/hans/int8_seed{seed}_predictions.csv` | W8A8 HANS predictions |
| `results/hans/weight_only_int8/weight_only_int8_seed{seed}_predictions.csv` | Weight-only HANS predictions |
| `results/mnli/weight_only_int8/` | Weight-only MNLI prediction and summary files |
| `results/hans/cpu_fp32_validation/` | Additional CPU FP32 validation |
| `results/analysis/` | Per-seed and summary comparison CSVs |
| `results/calibration/` | Threshold selection and supplementary evaluations |
| `results/visualization/paper_figures/` | Final PNG/SVG figures and plot-data CSVs |

### Figures

- `figure1_accuracy_changes.png` / `.svg`: MNLI matched/mismatched, HANS overall, and lexical-overlap non-entailment accuracy changes relative to FP32.
- `figure2_hans_subgroup_changes.png` / `.svg`: changes across all six HANS heuristic/label subgroups.

The bars show **three-seed mean differences** and the whiskers show the **observed minimum and maximum seed differences**, **not confidence intervals**. Underlying figure data CSVs are saved alongside the figures.

## 9. Interpretation and limitations

1. **Three seeds:** The results describe the observed runs and are not enough to make strong statistical generalizations.
2. **Repeated HANS items:** Summed flip counts aggregate seed-example evaluations, not independent dataset samples.
3. **Baseline floor effects:** Subsequence and constituent non-entailment accuracy is already low in FP32.
4. **Different quantization implementations:** W8A8 and weight-only configurations differ in more than one implementation detail. Their difference does not isolate activation quantization as the sole cause.
5. **Behavioral, not mechanistic evidence:** HANS accuracy, flips, and probability margins measure output behavior. They do not directly reveal which internal features the model uses.
6. **Device comparison scope:** FP32 CPU/MPS class-label agreement does not imply exact numerical equivalence.
7. **Reproducibility scope:** Checkpoints are excluded from Git, and MNLI summary consolidation was not yet packaged as a script.

## 10. Project context

This work was carried out for the Deep Learning Beyond Accuracy (DLBA) seminar at North Carolina State University by Tejas Pavular Ramesh and Mohnish Ramachandran. The final course paper should cite the original MNLI, HANS, BERT, and quantization-method sources separately; this repository note is an experiment record, not a substitute for the paper's references.
