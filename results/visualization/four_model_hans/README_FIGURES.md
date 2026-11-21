# DLBA Experiment 3D — four-model HANS visuals

**Source:** `/home/mohnishr/workspace/DLBA/Quantization-Spurious-Heuristics-in-Transformers/results/analysis/four_model_comparison`

The figures use original Experiment 3D CSVs; no model inference is rerun.
Averages and standard deviations are computed over three trained seeds (42, 123, 456).
Whiskers show SD, *not* statistical confidence intervals.
Pooled flip counts are across seed–example pairs, not independent test sentences.
Flip definitions: harmful = FP32 correct and quantized incorrect; beneficial = FP32 incorrect and quantized correct.
Entailment margin = P(entailment) − max(P(neutral), P(contradiction).
OpenVINO INT8 here means NNCF weight compression + OpenVINO CPU inference; W8A8 was a distinct TorchAO implementation.
The optional runtime tradeoff plot excludes W8A8 because its original throughput results were collected on CUDA.

## Figures

- **01_overall_hans_accuracy** — Overall HANS accuracy; mean ± 1 SD across three fine-tuning seeds.
- **02_subgroup_accuracy_heatmap** — Six HANS subgroups; mean accuracy across seeds for all four models.
- **03_subgroup_accuracy_change** — Per-subgroup mean accuracy change in percentage points vs seed-matched FP32.
- **04_seed_consistency** — Overall and lexical non-entailment accuracy by seed.
- **05_harmful_beneficial_flips** — Harmful/beneficial flips vs FP32 in all six subgroups; three seed-runs pooled.
- **06_overall_flip_totals** — All-HANS harmful and beneficial flip counts pooled across three seeds.
- **07_entailment_margin_shifts** — Mean change in entailment margin; negative/positive values retain their signs.
- **08_positive_margin_fraction** — Across-seed mean percentage of subgroup examples with positive entailment-margin change.
- **09_prediction_agreement** — Three-class versus binary HANS agreement with FP32, averaged across seeds.
- **10_harmful_flip_overlap** — Pairwise harmful-flip Jaccard overlap; diagonal excluded.
- **11_cpu_speed_accuracy_tradeoff** — CPU-only comparison from Experiment 3 runtime benchmarks; all variants use seed-matched full HANS runs.
