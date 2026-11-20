# Quantization and Spurious Heuristics in Transformers

A research project investigating how post-training INT8 quantization affects the robustness of BERT-based natural language inference (NLI) models to spurious linguistic heuristics.

## Research Question

Does quantization increase a model's sensitivity to linguistic shortcuts, even when its standard benchmark accuracy remains largely unchanged?

We investigate this question by comparing full-precision and quantized versions of the same fine-tuned BERT models on MNLI and HANS.

## Experimental Setup

**Base model:** `google-bert/bert-base-uncased`

**Training dataset:** Multi-Genre Natural Language Inference (MNLI)

**Evaluation datasets:**
- MNLI validation matched: 9,815 examples
- MNLI validation mismatched: 9,832 examples
- HANS: 30,000 examples

**Random seeds:** 42, 123, 456

**Inference configurations:**

| Configuration | Description |
|---|---|
| FP32 | Full-precision fine-tuned BERT |
| W8A8 | Dynamic INT8 activation and INT8 weight quantization |
| Weight-only INT8 | INT8 weight quantization without activation quantization |

Quantization uses TorchAO. Both quantized configurations target 74 linear layers.

HANS evaluates three linguistic heuristics: lexical overlap, subsequence, and constituent. Each heuristic contains entailment and non-entailment examples.

## Main Results

Mean accuracy across three seeds:

| Evaluation | FP32 | W8A8 | Weight-only INT8 |
|---|---:|---:|---:|
| MNLI matched | 84.64% | 84.74% | 84.66% |
| MNLI mismatched | 84.85% | 84.89% | 84.88% |
| HANS overall | 57.90% | 57.03% | 57.70% |
| HANS lexical-overlap non-entailment | 35.76% | 31.55% | 34.83% |

Relative to FP32, W8A8 reduced HANS lexical-overlap non-entailment accuracy by 4.21 percentage points, while weight-only INT8 reduced it by 0.93 points.

Standard MNLI accuracy remained nearly unchanged under both quantization configurations.

These findings suggest that standard benchmark accuracy alone may not reveal changes in sensitivity to linguistic shortcuts.

They do not establish that quantization directly increases reliance on a particular internal feature.

## Figures

The final paper figures are available in:

`results/visualization/paper_figures/`

- `figure1_accuracy_changes.png`: MNLI and HANS accuracy changes relative to FP32.
- `figure2_hans_subgroup_changes.png`: Accuracy changes across six HANS heuristic/label subgroups.

SVG versions and underlying plotting data are also provided.

Bars represent mean changes across three seeds. Error bars show the observed minimum and maximum across seeds, not confidence intervals.

## Repository Structure

```text
.
├── notebooks/       # Training, evaluation, and exploratory analysis
├── scripts/         # Additional evaluations, validation, and analysis
├── results/
│   ├── analysis/    # Consolidated comparisons and summaries
│   ├── calibration/ # Decision-threshold experiments
│   ├── hans/        # HANS predictions and evaluations
│   ├── mnli/        # MNLI evaluations
│   └── visualization/
│       └── paper_figures/
├── docs/            # Detailed experimental documentation
├── pyproject.toml
└── README.md
```

## Environment

The project uses Python 3.12, PyTorch, Hugging Face Transformers and Datasets, TorchAO, pandas, NumPy, and Matplotlib.

Dependencies are declared in `pyproject.toml`.

The additional quantized evaluations were performed on CPU. The original FP32 HANS evaluations used Apple MPS.

CPU validation reproduced the same FP32 predicted class labels as the original MPS evaluations across all 90,000 HANS seed-example evaluations. This does not imply identical numerical probabilities.

## Reproducibility

Evaluation scripts are located in `scripts/`.

The quantized evaluations require the corresponding fine-tuned checkpoints under:

`models/fp32/seed{seed}/final`

Model checkpoints are excluded from Git.

For detailed script usage, data dependencies, validation checks, experimental results, and limitations, see:

[Experiments and Results](docs/EXPERIMENTS_AND_RESULTS.md)

## Limitations

- Results are based on three fine-tuning seeds.
- HANS examples are reused across seeds, so pooled flip counts are not independent samples.
- Some HANS non-entailment subgroups have low baseline accuracy, creating floor effects.
- The quantization configurations differ in implementation, so their performance difference cannot be attributed solely to activation quantization.
- Accuracy and prediction-margin changes do not directly establish the model's internal reasoning mechanism.

## Project

Deep Learning Beyond Accuracy (DLBA), North Carolina State University.

**Team:** Tejas Pavular Ramesh and Mohnish Ramachandran.
