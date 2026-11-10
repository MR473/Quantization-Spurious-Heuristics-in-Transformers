# DLBA — research figure guide

All charts are generated from the precomputed CSVs in `results/visualization/`.
Each figure has a high-resolution PNG and editable SVG version.

## 01. Task-level changes across seeds

- **Interpretation:** MNLI is comparatively stable while HANS shows larger out-of-distribution differences.
- **Source:** `DLBA_seed_overview_3rows.csv`
- **PNG:** `01_task_level_accuracy.png`
- **SVG:** `01_task_level_accuracy.svg`

## 02. HANS heuristic × label heatmap

- **Interpretation:** Lexical-overlap non-entailment has the largest mean loss across three seeds.
- **Source:** `DLBA_crossseed_heuristic_6rows.csv`
- **PNG:** `02_heuristic_label_atlas.png`
- **SVG:** `02_heuristic_label_atlas.svg`

## 03. Seed-specific subgroup paired-interval forest

- **Interpretation:** The largest losses are consistently concentrated in lexical non-entailment; uncertainty is per seed.
- **Source:** `DLBA_heuristic_label_18rows.csv`
- **PNG:** `03_paired_subgroup_forest.png`
- **SVG:** `03_paired_subgroup_forest.svg`

## 04. Harmful versus beneficial prediction flips

- **Interpretation:** The lexical-overlap non-entailment group has a strongly asymmetric harmful flip balance.
- **Source:** `DLBA_crossseed_heuristic_6rows.csv`
- **PNG:** `04_correctness_flip_balance.png`
- **SVG:** `04_correctness_flip_balance.svg`

## 05. Full HANS subcase vulnerability ranking

- **Interpretation:** Some sentence structures deteriorate much more than the overall HANS average; signs vary across categories.
- **Source:** `DLBA_crossseed_subcase_30rows.csv`
- **PNG:** `05_all_hans_subcases.png`
- **SVG:** `05_all_hans_subcases.svg`

## 06. Entailment confidence shifts and harmful flips

- **Interpretation:** Shifts toward entailment coincide with a predominance of harmful flips on non-entailment examples.
- **Source:** `DLBA_entailment_probability_shift_bins.csv; DLBA_crossseed_heuristic_6rows.csv`
- **PNG:** `06_entailment_probability_shifts.png`
- **SVG:** `06_entailment_probability_shifts.svg`

## 07. Confidence–accuracy–flip relationship

- **Interpretation:** Lexical non-entailment exhibits an increased mean entailment probability and the largest harmful accuracy change.
- **Source:** `DLBA_crossseed_heuristic_6rows.csv`
- **PNG:** `07_confidence_accuracy_flip_bubbles.png`
- **SVG:** `07_confidence_accuracy_flip_bubbles.svg`

## 08. Memory reduction versus runtime penalty

- **Interpretation:** Dynamic INT8 reduces storage and peak CUDA allocation, but eager GPU throughput decreases strongly for seed 42.
- **Source:** `DLBA_runtime_cuda_seed42_2rows.csv`
- **PNG:** `08_cuda_eager_efficiency.png`
- **SVG:** `08_cuda_eager_efficiency.svg`

## Interpretation safeguards

- All percentage changes are signed **percentage points**, computed as INT8 minus FP32.
- Main HANS FP32 and INT8 prediction files were produced on different devices (GPU versus CPU); equivalence was **not verified**.
- Within-seed paired bootstrap intervals and BH-corrected tests do not imply generalization over all possible training seeds.
- Across-seed min–max intervals describe the three observed seeds and are **not confidence intervals**.
- The seed-42 CUDA eager efficiency benchmark must not be represented as CPU performance or a three-seed average.
- HANS behavior can reveal heuristic-sensitive errors; it does not directly establish an internal shortcut mechanism.
- The 18 confidence-shift bins have variable width; Figure 06 uses a **categorical** interval axis, not an equal-width histogram.