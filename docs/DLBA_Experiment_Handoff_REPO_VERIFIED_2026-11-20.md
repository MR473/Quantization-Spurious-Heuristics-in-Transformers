# DLBA Experiment Handoff — Quantization, CPU Runtime, OpenVINO, and HANS Behavior

**Project:** Quantization and Spurious Heuristics in Transformers — Deep Learning Beyond Accuracy (DLBA)  
**Repository:** `MR473/Quantization-Spurious-Heuristics-in-Transformers`  
**Experimental branch:** `feature/int8-runtime-optimization` (based on `clean`)  
**Handoff date:** November 20, 2026  
**Repository inventory verified against:** the `tree` output supplied by the researcher (69 directories, 615 files).  
**Purpose:** An end-to-end technical account for a teammate: the research question, why each experiment was performed, what code and data were used, the confirmed results, how to interpret them, and where the artifacts are located.

> **Repository scope:** The file inventory below lists **only paths visible in the supplied repository tree**; this handoff Markdown is a **new deliverable**, not a pre-existing repository file. The tree lists filenames but not the contents of the final four-model analysis CSVs. Hidden dotfiles are not verified from this `tree` output, so this README does not assert the presence of particular `.gitignore` files.

> **Scope and evidence:** The numbers in this document come from the saved legacy accuracy/flip tables, prior project documentation, uploaded CPU-runtime archive, actual OpenVINO three-seed CSV/JSONs, and shared successful run logs. The **four-model HANS behavioral pipeline has been run and its original results have been reproduced and validated**, as confirmed by the researcher. The actual *new four-model subgroup/flip/margin summary CSV contents* were not included with the handoff materials available for writing this document; consequently, numerical OpenVINO **subgroup-specific flip and margin values are deliberately not fabricated**. Their authoritative, completed outputs are located under `results/analysis/four_model_comparison/`, as mapped below. This is a documentation/provenance distinction, not a claim that the experiment is unfinished.

---

## 1. Executive summary: what the experiments demonstrated

We studied the same fine-tuned **BERT-base-uncased** checkpoints under different quantization and inference implementations. Our scientific question is whether INT8 compression changes reliance on misleading linguistic heuristics; our systems question is whether we can make inference **materially faster**, not merely make the saved model smaller.

1. **Original TorchAO dynamic W8A8 on CUDA:** quantization compressed model state but greatly *reduced* throughput versus FP32 on the tested RTX 5080 Laptop GPU (**2,990.77 → 562.13 examples/s** in seed 42 eager inference; approximately **5.32× slower**). This was not a fair CPU-vs-GPU comparison: it was a within-CUDA comparison of FP32 and W8A8.
2. **TorchAO weight-only INT8 on CPU:** saved state shrank by **58.47%**, but full-HANS throughput was essentially unchanged at both the initial 4-thread setting (**FP32 81.98 vs weight-only 81.65 examples/s**) and the tuned 8-thread setting (**FP32 153.70 vs weight-only 155.15 examples/s**), averaged across seeds.
3. **Tuning helped both PyTorch models:** moving from 4 to 8 threads increased throughput approximately **1.87× for FP32** and **1.90× for TorchAO weight-only**. This was **CPU configuration tuning**, not a quantization-specific gain.
4. **Profiling identified a plausible TorchAO execution bottleneck:** the quantized path used substantially more copy/conversion operations (`aten::copy_` self CPU time **203.1 ms vs 44.8 ms**; `aten::_to_copy` calls **1,200 vs 16**) and exhibited nontrivial process-memory lifecycle effects. This supports a hypothesis about overhead, **not a causal proof** of why every run was slower/faster.
5. **OpenVINO + NNCF solved the primary CPU performance problem:** across all three seeds on full HANS, **OpenVINO INT8 averaged 391.70 examples/s**, versus **151.55 PyTorch FP32**, **150.80 TorchAO weight-only**, and **167.61 OpenVINO FP32** in the same Experiment 3 campaign. Mean seed-paired speedup: **2.58× vs PyTorch FP32**, **2.61× vs TorchAO weight-only**, **2.34× vs OpenVINO FP32**.
6. **The fast model is also much smaller:** OpenVINO INT8 IR = **105.49 MiB** versus OpenVINO FP32 IR = **418.03 MiB**, a **74.76% within-backend artifact reduction**. NNCF reported **77/77 eligible weight groups** compressed into symmetric per-channel INT8 representations, and **77 INT8 constants** were verified in the saved model.
7. **HANS overall accuracy remained close:** three-seed mean **FP32 57.899%**, **TorchAO weight-only 57.697%**, **OpenVINO INT8 57.580%**. OpenVINO's mean change from FP32 was **−0.319 percentage points (pp)**. It is a speed–accuracy tradeoff, not an identical model; fine-grained HANS subgroup and flip analyses have also been produced using per-example OpenVINO predictions.
8. **The original behavioral finding remains important:** W8A8 reduced *lexical-overlap / non-entailment* accuracy by **4.207 pp** relative to FP32; TorchAO weight-only reduced it by **0.933 pp**. Standard MNLI accuracy scarcely changed. This is why the research cannot stop at aggregate accuracy or throughput.

### Final full-HANS CPU results from Experiment 3

**Three independent fine-tuning seeds (42, 123, 456); 30,000 HANS examples per seed; 8 CPU threads; batch 32; 3 timed whole-dataset passes; one synchronous stream.** Each reported throughput is the median-of-passes value for a seed, then averaged across seeds.

| Implementation | Mean throughput (examples/s) | Seed SD (examples/s) | Mean HANS accuracy | Serialized artifact size |
|---|---:|---:|---:|---:|
| PyTorch FP32 | **151.55** | 2.23 | **57.899%** | **417.73 MiB** |
| PyTorch + TorchAO weight-only INT8 | **150.80** | 13.82 | **57.697%** | **173.48 MiB** |
| OpenVINO FP32 | **167.61** | 0.40 | **57.899%** | **418.03 MiB** |
| **OpenVINO + NNCF INT8_SYM weight compression** | **391.70** | 2.31 | **57.580%** | **105.49 MiB** |

**Important:** The 150.80 TorchAO mean here is from Experiment 3's own three-seed campaign. Experiment 1B independently measured **155.15** under similar 8-thread/batch-32 settings. Seed 123 was notably slower for TorchAO in Experiment 3 (134.84 ex/s, compared with 159.22 in Experiment 1B). Preserve both results with their experiment labels; **do not replace or silently average across campaigns**.

---

## 2. Model, data, and experimental controls

### 2.1 Shared neural model and checkpoints

- Architecture: `google-bert/bert-base-uncased`, sequence classification head, **three MNLI classes**.
- Training configuration recorded in the existing `docs/EXPERIMENTS_AND_RESULTS.md`: fine-tuned on MNLI for **3 epochs**, learning rate **2e-5**, batch size **32**, max sequence length **128**, using seeds **42, 123, 456**.
- All runtime experiments start from **existing saved FP32 checkpoints** at `models/fp32/seed{seed}/final/`. **There is no new training when switching to OpenVINO or changing quantization formats.**
- BERT-base has **12 encoder blocks**, each with attention projections Q/K/V/output and feed-forward intermediate/output projection (72 encoder Linear modules). The TorchAO implementation also quantizes the pooler and classifier for **74 Linear modules**.
- Separate seeds represent separate trained checkpoints, *not* repeated stochastic inference on one checkpoint.

### 2.2 Datasets and label mapping

| Dataset | Role | Examples used |
|---|---|---:|
| MNLI matched validation | Standard/in-distribution NLI accuracy | 9,815 |
| MNLI mismatched validation | Cross-genre NLI accuracy | 9,832 |
| HANS | Challenge set for spurious NLI heuristics | 30,000 per seed |

HANS consists of **lexical overlap, subsequence, and constituent** heuristics, each paired with gold **entailment** or **non-entailment**, for **six subgroups × 5,000 examples** each. Exactly the same HANS examples are evaluated under every seed and implementation.

Class mapping in this repository:

- `0`: entailment;
- `1`: neutral;
- `2`: contradiction;
- HANS binary prediction: class `0` → **entailment**; classes `1` and `2` → **non-entailment**.

This binary HANS conversion occurs **after the three-class argmax**. It does *not* normally threshold `P(entailment)` at `0.5`.

### 2.3 Implementations: what actually changes

| Variant | Compression/quantization | Runtime execution | Scope |
|---|---|---|---|
| PyTorch FP32 | No quantization | PyTorch eager CPU / original CUDA run | Reference model |
| TorchAO dynamic W8A8 | `Int8DynamicActivationInt8WeightConfig()` | PyTorch; original eager CUDA + other diagnostics | INT8 weights and dynamically quantized activations |
| TorchAO weight-only INT8 | `Int8WeightOnlyConfig()` | PyTorch eager CPU | Weight compression; activations not explicitly converted to INT8 by the configuration |
| OpenVINO FP32 | Converted FP32 graph (no NNCF quantization) | OpenVINO CPU plugin | Isolate backend difference within OpenVINO study |
| OpenVINO INT8_SYM | NNCF `compress_weights(..., mode=INT8_SYM)` | OpenVINO CPU plugin | Symmetric per-channel INT8 **weight compression** |

**Do not call these quantizers identical.** TorchAO counts 74 PyTorch Linear modules; NNCF reports 77 eligible *OpenVINO graph weight groups*. The counts reflect different graph/module representations and selection logic. Neither the `77` count nor the speedup proves that every activation or MatMul ran in native INT8 arithmetic. For Experiment 3, OpenVINO's `INFERENCE_PRECISION_HINT` was set to FP32 and activations were **not explicitly calibrated/quantized**.

### 2.4 Fair benchmarking boundaries

- Runtime performance work on **Pop!_OS/Linux x86_64 CPU** (Python **3.12.3**, PyTorch build recorded as **2.14.0+cu130**, even though CPU is explicitly selected). CUDA-build PyTorch does not mean the CPU experiment used CUDA.
- Exact same seed-specific checkpoints, HANS sentence pairs, tokenizer, max length 128, input order, and batch size within each comparison.
- **Timed scope:** model-forward inference over **pre-tokenized, pre-collated batches**; excludes tokenizer work, model loading, graph conversion, compression, and compilation.
- Benchmarks use **fresh processes** for each model implementation. Experiment 3 uses synchronous single-request execution, `NUM_STREAMS=1`, 8 threads and CPU plugin, rather than maximizing OpenVINO asynchronous server throughput.
- **Do not compare** 2,990 CUDA examples/s to ~392 CPU examples/s as a quantization comparison: hardware and quantization methods differ.
- **Process RSS** includes Python, framework buffers, allocators, data and model residency; it is *not* isolated parameter RAM or peak device memory.
- Means/standard deviations are computed across **three seeds**; they are not confidence intervals. Counts summed across seeds are **seed–example evaluations**, not independent new examples.

---

## 3. Original CUDA experiment — why we investigated other execution paths

**Question:** Does TorchAO dynamic INT8 W8A8 accelerate the MNLI-fine-tuned BERT model on the original GPU path?

**Setup:** seed 42, NVIDIA RTX 5080 Laptop GPU, PyTorch eager, dynamic activation/INT8 weight quantization, 30,000 HANS examples, batch size 32. The provided repository tree shows **completed eager-run output folders** for seed 42; no compiled-run output folder is listed in this snapshot. The directly comparable completed eager runs are reported below.

| Metric (CUDA seed 42) | FP32 eager | TorchAO W8A8 eager |
|---|---:|---:|
| Full-HANS throughput | **2,990.77 ex/s** | **562.13 ex/s** |
| Median whole-pass time | 10.03 s | 53.37 s |
| Batch-32 median forward latency | 10.28 ms | 58.71 ms |
| Serialized `state_dict` | 417.73 MiB | 173.48 MiB |
| Peak CUDA allocated during test | 475.52 MiB | 208.25 MiB |

**Interpretation:** the original W8A8 path reduced model size and measured GPU allocations, but inference was **~5.32× slower** (2990.77/562.13) than eager CUDA FP32. The plausible issue is the particular quantized execution path and kernel/conversion overhead, **not proof that all INT8 inference is slower**. Compiled variants and prediction-equivalence diagnostics are recorded separately in `results/runtime_analysis/seed42/` and aggregate files; do not mix their results with the eager comparison without documenting their configurations/status.

**Canonical files:**

- `results/runtime_analysis/all_benchmark_variants.csv` — one row per original CUDA variant and benchmark metadata.
- `results/runtime_analysis/all_throughput_repeats.csv`, `all_latency_samples.csv` — pass-level throughput and batch-latency evidence.
- `results/runtime_analysis/seed42/fp32_eager/`, `seed42/int8_eager/` — benchmark summaries, repeat timings, prediction metadata/CSV, profiler traces and statuses.
- `results/runtime_analysis/prediction_differences/` and `prediction_equivalence_checks.csv` — paired prediction diagnostics for the CUDA runtime variants.
- `results/runtime_analysis/runtime_throughput_comparison.png`, `runtime_batch32_latency.png`, `runtime_serialized_state.png`, `runtime_peak_cuda_allocated.png` — original visualization set.

---

## 4. Baseline behavior: why HANS matters more than aggregate MNLI accuracy

This study began with a **three-model** comparison: FP32, TorchAO dynamic W8A8, and TorchAO weight-only INT8, evaluated on the same checkpoints and aligned HANS examples. All values below are **means across three seeds**.

| Metric | FP32 | TorchAO W8A8 | TorchAO weight-only |
|---|---:|---:|---:|
| MNLI matched | **84.6358%** | **84.7444%** | **84.6561%** |
| MNLI mismatched | **84.8454%** | **84.8895%** | **84.8827%** |
| HANS overall | **57.8989%** | **57.0333%** | **57.6967%** |
| HANS lexical-overlap / non-entailment | **35.7600%** | **31.5533%** | **34.8267%** |

MNLI changes are tiny, yet **lexical-overlap non-entailment** drops **4.2067 pp for W8A8** and **0.9333 pp for weight-only**, relative to FP32. That subgroup contains counterexamples to the shortcut “high word overlap implies entailment.” Larger error rates there are **consistent with**, but do not prove, greater reliance on that heuristic.

### Changes across the six HANS subgroups (percentage points vs FP32)

| HANS subgroup | W8A8 | TorchAO weight-only |
|---|---:|---:|
| Lexical overlap — entailment | +0.5000 | +0.0400 |
| Lexical overlap — non-entailment | **−4.2067** | **−0.9333** |
| Subsequence — entailment | +0.1333 | +0.0067 |
| Subsequence — non-entailment | −0.9867 | −0.2333 |
| Constituent — entailment | +0.0867 | +0.0267 |
| Constituent — non-entailment | −0.7200 | −0.1200 |

**Floor effect:** FP32 already scored **5.52%** on subsequence non-entailment and **10.32%** on constituent non-entailment. Those subgroups have little room to decline, so a small percentage-point drop does not automatically imply a more robust mechanism.

### Earlier three-model flip analysis: lexical non-entailment

A **harmful flip** is `FP32 correct → quantized wrong`; a **beneficial flip** is `FP32 wrong → quantized correct`. Accuracy change = `(beneficial − harmful) / subgroup_size`.

| Seed | W8A8 harmful | W8A8 beneficial | Weight-only harmful | Weight-only beneficial |
|---|---:|---:|---:|---:|
| 42 | 211 | 32 | 45 | 14 |
| 123 | 304 | 11 | 81 | 6 |
| 456 | 171 | 12 | 40 | 6 |
| **Total (three seed-evaluations)** | **686** | **55** | **166** | **26** |

W8A8 therefore had a net **−631 correct** seed-example evaluations in this subgroup; TorchAO weight-only had **−140**. Of the harmful flips, **147** occurred under *both* quantizations, **539** only under W8A8, and **19** only under weight-only. Counts are pooled across the same **5,000 unique HANS items examined under three checkpoints**.

### Earlier entailment-margin analysis

For each example, define the **three-class entailment margin**:

```text
margin = P(entailment) - max(P(neutral), P(contradiction))
margin_shift = quantized_margin - fp32_margin
```

This is **not** `P(entailment) − P(neutral+contradiction)`. Positive margin shift means entailment became more competitive relative to the strongest individual alternative class.

| HANS non-entailment subgroup | W8A8 mean margin shift | TorchAO weight-only mean shift |
|---|---:|---:|
| Lexical overlap | **+0.0677** | **+0.0152** |
| Subsequence | +0.0192 | +0.0046 |
| Constituent | +0.0062 | +0.0018 |

Original three-model reports: `results/analysis/hans_accuracy_*`, `hans_flips_*`, `hans_margin_*`, and `hans_harmful_flip_overlap.csv`. Original paper figures: `results/visualization/paper_figures/figure1_accuracy_changes.*` and `figure2_hans_subgroup_changes.*`. These original files are **preserved**; four-model analysis is in a new directory.

**Cross-device reference check:** a separate FP32 CPU validation reproduced **all 90,000 saved three-class FP32 labels** from the original MPS evaluations (30k × three seeds). This supports label consistency, **not numerical equality of all FP32 logits/probabilities**.

**Supplementary threshold test (separate from quantization):** W8A8 thresholds selected on MNLI matched were **0.25, 0.35, 0.24** by seed. Relative to the original W8A8 argmax rule, mean MNLI mismatched changed **+0.08 pp**, HANS overall **−2.37 pp**, and lexical non-entailment **−8.95 pp**. It changes the *decision rule*, not model weights or underlying probabilities; files are under `results/calibration/`.

---

## 5. Initial CPU benchmark — FP32 vs TorchAO weight-only INT8 (4 threads)

**Why:** CUDA dynamic W8A8 was slower, but dynamic W8A8 and weight-only INT8 are different implementations. We therefore established a *same-device, same-workload CPU baseline* for the weight-only representation, instead of comparing unrelated GPU and CPU absolute speeds.

**Setup:** separate CPU processes, full 30k HANS per seed, three seeds, **4 CPU threads, batch size 32, 3 repeats**. A seed-42 3k-example preliminary pilot was also saved.

| Seed | PyTorch FP32 (ex/s) | TorchAO weight-only (ex/s) | Quantized vs FP32 |
|---|---:|---:|---:|
| 42 | 82.461 | 83.201 | +0.90% |
| 123 | 82.625 | 83.142 | +0.63% |
| 456 | 80.841 | 78.600 | −2.77% |
| **Mean** | **81.976** | **81.648** | **−0.40%** (ratio of means) |

**Result:** a **58.471% serialized `state_dict` reduction**, without a repeatable throughput improvement. At this model-setup checkpoint the weight-only worker showed roughly **+421 MiB more whole-process RSS**, motivating deeper memory analysis. Weight-only also gave slightly lower isolated batch latency in some runs, but that did **not** translate into a large full-dataset speedup.

**Output:** `results/runtime_analysis/fp32_vs_weight_only_cpu/` — `benchmark_results_n30000.csv`, `paired_comparison_n30000.csv`, per-seed JSONs, and four figure pairs (`01_serialized_size`, `02_process_rss`, `03_batch_latency`, `04_throughput`; `.png`/`.svg`). Pilot `n3000` files are stored separately in the same folder.

**Benchmark script:** `scripts/benchmark_fp32_weight_only_cpu.py`. Full original invocation:

```bash
uv run python scripts/benchmark_fp32_weight_only_cpu.py \
  --seeds 42 123 456 --num-examples 30000 \
  --repeats 3 --threads 4 --batch-size 32 \
  --out-dir results/runtime_analysis/fp32_vs_weight_only_cpu
```

---

## 6. Experiment 1A — CPU tuning grid

**Why:** If the slowdown was a suboptimal CPU execution configuration, thread count or batch size might improve both variants. We tested **both** FP32 and TorchAO weight-only under the same settings, using **seed 42, first 1,000 HANS items, two repeats**.

| CPU threads | Batch size | FP32 ex/s | Weight-only ex/s | Weight-only vs FP32 |
|---:|---:|---:|---:|---:|
| 4 | 32 | 76.543 | 70.344 | −8.10% |
| 2 | 32 | 39.510 | 40.057 | +1.38% |
| **8** | **32** | **143.704** | **149.865** | **+4.29%** |
| 4 | 16 | 76.864 | 79.421 | +3.33% |
| 4 | 64 | 75.125 | 71.588 | −4.71% |
| 8 | 64 | 145.695 | 133.290 | −8.51% |

**Decision:** advance **8 threads / batch 32** for a full-dataset test. It was the best weight-only configuration in this *tested grid*; not a claim of global optimality. The 1k tuning pilot is unsuitable for a definitive speed comparison.

**Output:** six non-overlapping folders under `results/runtime_analysis/tuning/`:

```text
t2_b32/  t4_b16/  t4_b32/  t4_b64/  t8_b32/  t8_b64/
```

Each contains per-variant JSON, aggregate CSV, and size/RSS/latency/throughput PNG+SVG figures.

---

## 7. Experiment 1B — full tuned CPU confirmation (8 threads)

**Why:** Small pilots can exaggerate or hide performance differences, so the chosen configuration must be tested across all 30k HANS items, three seeds and three repeats.

| Seed | PyTorch FP32 ex/s | TorchAO weight-only ex/s | Difference |
|---|---:|---:|---:|
| 42 | 155.935 | 159.550 | +2.32% |
| 123 | 152.755 | 159.217 | +4.23% |
| 456 | 152.421 | 146.691 | −3.76% |
| **Mean** | **153.704** | **155.153** | **+0.94%** (ratio of means) |

The switch **4 → 8 CPU threads** increased mean FP32 throughput from **81.98 → 153.70 ex/s (1.87×)** and weight-only throughput from **81.65 → 155.15 ex/s (1.90×)**. Both models improved dramatically; the **increment attributable to quantization itself remained small and inconsistent by seed**.

**Output:** `results/runtime_analysis/tuning/full_t8_b32/` with `benchmark_results_n30000.csv`, `paired_comparison_n30000.csv`, seed-specific JSONs, and matching figure pairs. It is a distinct campaign from Experiment 3's later 8-thread results.

**Run:**

```bash
uv run python scripts/benchmark_fp32_weight_only_cpu.py \
  --seeds 42 123 456 --num-examples 30000 \
  --repeats 3 --threads 8 --batch-size 32 \
  --out-dir results/runtime_analysis/tuning/full_t8_b32
```

---

## 8. Experiment 2 — CPU operator attribution and memory lifecycle

**Why:** TorchAO weight-only kept nearly the same throughput as FP32 despite dramatically reducing serialized weights. It also initially reported more process RSS. We asked **(a) what operators dominate the forward pass? (b) when do process-memory differences appear and disappear?**

**Setup:** seed 42, 8 threads, batch 32, **256 HANS examples**, PyTorch CPU profiler, independent worker processes. This is a **diagnostic run**, not a substitute for the full 30k performance results.

### 8.1 Whole-process RSS across execution stages (MiB)

| Stage | FP32 | Weight-only | Weight-only − FP32 |
|---|---:|---:|---:|
| Inputs prepared / before model load | 944.77 | 944.63 | −0.14 |
| Ready after model load/quantization | 1,000.57 | 1,422.47 | **+421.90** |
| After warm-up | 1,357.50 | 1,468.88 | +111.37 |
| After unprofiled inference | 1,388.68 | 1,495.32 | +106.64 |
| After garbage collection and `malloc_trim()` | 1,355.99 | 1,437.87 | **+81.88** |

**Correct interpretation:** most of the reduction from a **~422 MiB gap to ~107 MiB** was already observed **before** trimming, as the FP32 process's RSS increased during warm-up/inference and memory residency evolved. The final trim reduced the relative gap by about **24.77 MiB**, not ~340 MiB. RSS includes the runtime and allocator. We have **not demonstrated** that TorchAO weight-only has lower steady-state or peak RAM than FP32.

### 8.2 Profiled operations

| Operator/metric | FP32 | TorchAO weight-only |
|---|---:|---:|
| Leading matrix multiply operator | `aten::addmm` | `aten::mm` |
| Leading matrix multiply *self CPU time* | **3,118.1 ms** | **3,179.6 ms** |
| `aten::copy_` self CPU time | **44.8 ms** | **203.1 ms** |
| `aten::_to_copy` count | **16 calls** | **1,200 calls** |
| Total profiled self CPU time | **3,344.2 ms** | **3,636.2 ms** |

**Finding:** weight-only inference shows more conversion/copy operations and may fail to exploit a sufficiently optimized CPU integer-kernel path to compensate for that overhead. This is a **plausible explanation**, not a demonstrated exact breakdown of wall-clock slowdown. Profiled *self CPU time* differs from normal wall time; quantization doesn't automatically imply faster matrix multiplication.

**Output:** `results/runtime_analysis/experiment2_profiler/seed42_t8_b32_n256/`:

- `experiment2_comparison.json` — headline diagnostic differences;
- `experiment2_summary.csv` — per-model diagnostic metrics;
- `experiment2_memory_timeline.csv` — joint RSS history;
- `experiment2_operator_attribution.csv` — operator time comparison;
- `seed42_{fp32,weight_only_int8}_diagnostics.json` — model-specific metrics;
- `seed42_{fp32,weight_only_int8}_memory_stages.csv` and `_operators.csv` — raw per-model support;
- `01_memory_lifecycle`, `02_model_setup_rss`, `03_cpu_operator_attribution`, `04_cpu_operator_per_batch` in PNG and SVG — diagnostic figures.

**Script:** `scripts/experiment2_cpu_profiler_memory.py`.

---

## 9. Experiment 3 — replace the inference pipeline with OpenVINO + NNCF

### 9.1 Motivation and hypothesis

By Experiment 2, *weight compression* was successful, but the TorchAO CPU execution path was not giving a meaningful throughput benefit. Changing only thread counts was insufficient. Therefore, Experiment 3 changes **both** the graph representation/compression implementation and the inference runtime:

```text
Saved fine-tuned FP32 BERT (per seed)
    |
    +--> PyTorch FP32 on CPU                        [same-checkpoint reference]
    +--> TorchAO Int8WeightOnlyConfig on PyTorch CPU [old compressed baseline]
    +--> OpenVINO-converted FP32 graph on CPU        [backend control]
    +--> OpenVINO graph + NNCF INT8_SYM on CPU       [new optimized path]
```

The key two within-campaign controls are:

1. **PyTorch FP32 → OpenVINO FP32:** estimate the difference from the backend/graph-conversion change *without INT8 compression*.
2. **OpenVINO FP32 → OpenVINO INT8:** estimate the effect of NNCF weight compression **within OpenVINO** (including any resulting kernel-selection/fusion differences).

The cross-framework comparison (**TorchAO weight-only → OpenVINO INT8**) evaluates two **complete deployment implementations**, not NNCF's scaling algorithm in isolation.

### 9.2 How the OpenVINO quantization works

1. Load the existing FP32 BERT checkpoint and convert the model into OpenVINO's intermediate representation (IR) consisting of an XML graph and a binary weights file.
2. Apply `nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_SYM)`.
3. NNCF approximates eligible FP32 weight arrays by **signed 8-bit integers plus scaling information**, using **symmetric, per-channel weight compression**. Conceptually, `q ≈ round(w / scale)` (clipped to INT8 range), and the original magnitude is approximated by applying the scale during execution.
4. The export log reported **77/77 eligible weight groups**, all `int8_sym, per-channel`. Inspection of the compressed graph and reloaded saved IR confirmed **77 `i8` Constant nodes**.
5. OpenVINO's CPU compiler then executes the graph using its own optimized implementations. **No activation calibration dataset is used**, and no fine-tuning/retraining occurs.

Quantization errors can change predicted class probabilities and therefore labels. Weight compression does **not by itself guarantee** that every operation uses integer arithmetic at runtime.

### 9.3 Conversion-validation fix (important reproducibility note)

The first attempt converted BERT successfully and NNCF completed all 77 weight groups, but an overly strict post-compression verification failed with:

```text
RuntimeError: No INT8 constants detected: cannot claim weight compression
```

The cause was **type comparison in the verification code**: OpenVINO exposes element-type objects, which must be compared with `ov.Type.i8` / `ov.Type.u8`, not stringified to presumed `"i8"` / `"u8"`. The corrected script verified **77 INT8 Constant nodes**, saved the compressed IR and reloaded it to confirm serialization. That validation safeguard was **corrected**, not bypassed. The earlier `torch.jit.trace` deprecation/tracer warnings were not the cause of this error.

### 9.4 Initial 512-example pilot (seed 42)

**8 threads, batch 32, 2 repeats, 4 warm-up batches.** This pilot validated setup, correctness and relative performance, not final HANS accuracy.

| Model | Pilot throughput (ex/s) | Pilot HANS accuracy |
|---|---:|---:|
| PyTorch FP32 | 190.17 | 47.070% |
| TorchAO weight-only | 198.81 | 46.484% |
| OpenVINO FP32 | 215.36 | 47.070% |
| **OpenVINO INT8** | **485.02** | **44.922%** |

The first 512 examples **aren't representative of the balanced full HANS dataset**: the lower pilot accuracy and its −2.15 pp OpenVINO accuracy difference should not be extrapolated to all 30k items. The speed result motivated the full campaign.

### 9.5 Full-HANS, three-seed result

**30k HANS items × three seeds, 8 threads, batch 32, 3 repeats, 10 warm-up batches.** Conversion/compression are performed outside timed inference and cached by seed. All variants use independent workers.

| Seed | PyTorch FP32 | TorchAO weight-only | OpenVINO FP32 | **OpenVINO INT8** |
|---|---:|---:|---:|---:|
| **42** | 154.02 | 158.95 | 167.51 | **394.27** |
| **123** | 150.98 | 134.84 | 167.27 | **389.81** |
| **456** | 149.67 | 158.62 | 168.05 | **391.03** |
| **Mean** | **151.55** | **150.80** | **167.61** | **391.70** |

*Throughput units: HANS examples/second.*

| Seed | FP32 HANS accuracy | TorchAO weight-only | OpenVINO FP32 | **OpenVINO INT8** |
|---|---:|---:|---:|---:|
| 42 | 59.700% | 59.527% | 59.700% | **59.320%** |
| 123 | 60.150% | 59.840% | 60.150% | **59.783%** |
| 456 | 53.847% | 53.723% | 53.847% | **53.637%** |
| **Mean** | **57.899%** | **57.697%** | **57.899%** | **57.580%** |

**Mean of the three per-seed speedup ratios:**

| Comparison | Speedup |
|---|---:|
| OpenVINO FP32 / PyTorch FP32 | **1.11×** (approximately) |
| OpenVINO INT8 / PyTorch FP32 | **2.585×** |
| OpenVINO INT8 / TorchAO weight-only | **2.612×** |
| OpenVINO INT8 / OpenVINO FP32 | **2.337×** |

**Accuracy:** OpenVINO INT8 decreased overall HANS accuracy by **0.3189 percentage points** on average vs FP32, and by **0.1167 pp** versus TorchAO weight-only (mean difference of seed-paired accuracies). Mean three-class label agreement with saved FP32 predictions was **99.43%** for OpenVINO INT8 versus **100%** for OpenVINO FP32.

**Storage and process memory:**

| Metric (mean across seeds) | PyTorch FP32 | TorchAO weight-only | OpenVINO FP32 | **OpenVINO INT8** |
|---|---:|---:|---:|---:|
| Serialized artifact | 417.73 MiB | 173.48 MiB | 418.03 MiB | **105.49 MiB** |
| Process RSS, inference-ready | 1,444 MiB | 1,531 MiB | 1,714 MiB | **1,088 MiB** |
| Process RSS, after inference | 1,499 MiB | 1,596 MiB | 1,891 MiB | **1,343 MiB** |

From 418.03 → 105.49 MiB, **within-OpenVINO storage fell 74.76%**. From 173.48 → 105.49 MiB, OpenVINO's saved artifact was about **39.2% smaller** than TorchAO's, but file formats differ (`.xml/.bin` vs `.pt`) and this is a deployment artifact comparison rather than isolated quantizer efficiency. RSS values are whole-process snapshots in separate workers, **not peak tensor or model-only RAM**.

**Anomaly worth preserving in the analysis:** TorchAO seed 123 measured **134.84 ex/s** in Experiment 3 versus **159.22 ex/s** in Experiment 1B. OpenVINO INT8 was stable at **389.81–394.27 ex/s** across seeds; its speed advantage persists regardless of this TorchAO difference. Do not conflate the two campaigns.

### 9.6 Files and reproducibility

**Script:** `scripts/experiment3_openvino_cpu.py`.

Example full benchmark invocation (separately for seeds 42, 123, 456):

```bash
uv run python scripts/experiment3_openvino_cpu.py \
  --seed 42 --num-examples 30000 --threads 8 \
  --batch-size 32 --repeats 3 --warmup 10
```

**Cached models (do not commit large binaries):**

```text
models/experiment3_openvino/
  seed42/
    fp32.xml / fp32.bin             # converted FP32 OpenVINO IR
    int8_sym.xml / int8_sym.bin     # NNCF-compressed OpenVINO IR
    model_export_metadata.json      # checkpoint signature, export and compression verification
  seed123/                           # same layout
  seed456/                           # same layout
```

**Benchmark reports:**

```text
results/runtime_analysis/experiment3_openvino/
  seed42_t8_b32_n512/                # exploratory pilot
  seed42_t8_b32_n30000/              # full seed 42
  seed123_t8_b32_n30000/             # full seed 123
  seed456_t8_b32_n30000/             # full seed 456
  three_seed_results.csv             # combined 12-row, four-model × three-seed table
```

Inside each run directory:

- `benchmark_summary.csv` — four variants, throughput, latency, HANS accuracy, model size and process RSS;
- `torch_fp32.json`, `torchao_weight_only.json`, `openvino_fp32.json`, `openvino_int8_sym.json` — detailed individual-run metrics, full pass durations and checks;
- `experiment3_notes.json` — settings and interpretation caveats;
- `01_cpu_throughput`, `02_hans_accuracy`, `03_artifact_size`, `04_ready_process_rss` — paired `.png`/`.svg` plots.

**Overwrite behavior:** different seed/threads/batch-size/example-count combinations create separate directories. Rerunning the *same* combination can replace its reporting files; changing only warm-up or repeat count does not change the folder's name. Cached OpenVINO exports are reused unless explicitly rebuilt. In particular, **do not treat a benchmark result folder as immutable archival storage**.

---

## 10. Experiment 3C — OpenVINO per-example HANS predictions

**Why:** Experiment 3's runtime reports establish speed and overall accuracy, but aggregate numbers cannot tell us whether a small accuracy drop comes from a few harmful subgroup changes. For a scientific comparison, we need **the same individual HANS examples** and the full three-class probability distribution from the new OpenVINO model.

**Script:** `scripts/evaluate_openvino_int8.py`. It loads the existing `models/experiment3_openvino/seed{seed}/int8_sym.xml` cache, tokenizes the original HANS pairs consistently, runs OpenVINO CPU inference, softmax-normalizes the logits, and writes the familiar prediction schema. **No retraining, reconversion, recompression or TorchAO/FP32 rerun** is necessary.

Representative command (run for each seed):

```bash
uv run python scripts/evaluate_openvino_int8.py \
  --seed 42 --num-examples 30000 --threads 8 --batch-size 32
```

**Output for each seed:**

```text
results/hans/openvino_int8/
  openvino_int8_seed42_predictions.csv
  openvino_int8_seed42_summary.json
  openvino_int8_seed123_predictions.csv
  openvino_int8_seed123_summary.json
  openvino_int8_seed456_predictions.csv
  openvino_int8_seed456_summary.json
```

Required prediction columns include `pairID`, `premise`, `hypothesis`, `gold_label`, `heuristic`, `subcase`, `mnli_prediction_id`, `mnli_prediction`, `hans_prediction`, `correct`, `prob_entailment`, `prob_neutral`, `prob_contradiction`, and `prob_non_entailment`. Other metadata fields (such as `template`) are preserved from the reference if present.

**Validation built into the evaluation:**

- Full-run count is 30,000 examples per seed; unique `pairID` values and reference schema are required.
- The cached IR is checked against fine-tuned checkpoint signature metadata.
- Output probabilities must be finite, lie in [0, 1] and sum to 1; the class ID must agree with `argmax` and the HANS binary conversion.
- Accuracy and three-class agreement with the saved FP32 reference must match the previously saved Experiment 3 benchmark when the matching runtime JSON is present.
- The evaluator refuses to overwrite its existing per-seed result unless passed `--overwrite`.

**Why the `pairID` matters:** it is the HANS example identifier used to align model outputs to the *same premise–hypothesis pair*. We did not create new pair IDs as a post-processing trick. If the row order differs, alignment by `pairID` prevents spurious flip counts.

---

## 11. Experiment 3D — four-model HANS accuracy, flips, margins, and agreement

**Why:** The central DLBA question is not merely how fast OpenVINO runs, but *which decisions it changes*, especially on HANS heuristics. The existing three-model analysis was therefore extended **without rewriting its results** to cover:

1. **FP32 reference**;
2. **TorchAO W8A8**;
3. **TorchAO weight-only INT8**;
4. **OpenVINO NNCF INT8_SYM**.

**Script:** `scripts/analyze_four_model_hans.py` — reads per-example predictions only (no inference). Its completed, validated results are saved in `results/analysis/four_model_comparison/`.

The script enforces **30,000 rows/model/seed**, unique and matching `pairID` sets, matching gold labels/heuristics/subcases, valid three-class probabilities and predicted classes, and HANS binary/correctness consistency. It calculates every statistic against the *aligned seed-specific FP32 prediction*, not against unpaired samples.

### 11.1 Exact calculations

| Quantity | Definition | What it tells us |
|---|---|---|
| HANS subgroup accuracy | Correct binary HANS labels / subgroup count (5,000) | Which heuristic/label groups degrade |
| Harmful flips | FP32 correct **and** quantized incorrect | Newly introduced errors |
| Beneficial flips | FP32 incorrect **and** quantized correct | Newly corrected errors |
| Net accuracy change | `(beneficial − harmful) / n` | Exact difference in subgroup accuracy |
| Three-class prediction agreement | Fraction with equal MNLI class ID (0/1/2) | Whether the predicted NLI label changed |
| Binary HANS agreement | Fraction with same entailment/non-entailment prediction | Whether the HANS decision changed |
| Entailment margin | `P(E) − max(P(N), P(C))` | Competition of entailment vs strongest alternative |
| Mean/median margin shift | Quantized margin minus FP32 margin | Direction/magnitude of probability-space shift |
| Positive margin-shift fraction | Fraction where quantized margin > FP32 margin | How widespread the entailment-leaning change is |
| Pairwise harmful-flip overlap | Both harmful / only A harmful / only B harmful | Whether two quantizations damage the same example–seed pairs |

**Important distinction:** Changing from *neutral* to *contradiction* changes the **three-class** label but not the HANS **non-entailment** prediction. Also, a harmful flip changes correctness; it need not be the same thing as *any* three-class label change.

The code computes all six heuristic/label subgroups **and the overall set**, per seed; then reports three-seed averages/SDs and **pooled flip counts**. Pooled flip counts cover 90,000 **seed–example evaluations** overall (or 15,000 for one 5k subgroup), **not 90,000 unique HANS items**.

### 11.2 Confirmed overall behavioral context

| Model | Mean HANS accuracy | Change relative to FP32 |
|---|---:|---:|
| FP32 | **57.8989%** | reference |
| TorchAO W8A8 | **57.0333%** | **−0.8656 pp** |
| TorchAO weight-only INT8 | **57.6967%** | **−0.2022 pp** |
| **OpenVINO INT8** | **57.5800%** | **−0.3189 pp** |

These overall values are independently available from the previous three-model tables and Experiment 3 full-HANS runtime outputs. **For the exact OpenVINO *six-subgroup* losses, harmful/beneficial counts, pairwise overlaps, and margin statistics, refer directly to the completed Experiment 3D CSVs listed below** rather than guessing values from the overall accuracy difference. Their calculation definitions are identical to the legacy approach.

### 11.3 Experiment 3D output files — authoritative table index

All paths are relative to repository root:

| File in `results/analysis/four_model_comparison/` | Contents / use |
|---|---|
| `hans_accuracy_by_seed.csv` | Every model × seed × overall/six subgroup: `n` and accuracy |
| `hans_accuracy_summary.csv` | Mean and SD subgroup accuracies across seeds |
| `hans_flips_by_seed.csv` | Harmful, beneficial, and exact net-accuracy change for each quantized model/seed/group |
| `hans_flip_summary.csv` | Mean/SD flips and **pooled_harmful_seed_examples** / **pooled_beneficial_seed_examples** |
| `hans_margin_shifts_by_seed.csv` | Mean/median entailment-margin changes and positive-shift fraction |
| `hans_margin_summary.csv` | Across-seed mean and SD of margin statistics |
| `hans_prediction_agreement_by_seed.csv` | Per-seed three-class/binary agreement and changed prediction counts |
| `hans_prediction_agreement_summary.csv` | Across-seed agreement mean and SD |
| `hans_harmful_flip_pairwise_overlap.csv` | Pairwise harmful-flip overlap among W8A8, weight-only, and OpenVINO INT8, broken down by seed/subgroup |

The original `results/analysis/` three-model files stay untouched. Script protection: nonempty Experiment 3D output directories **are not overwritten** unless `--overwrite` is explicitly passed.

**Run using existing prediction files:**

```bash
uv run python scripts/analyze_four_model_hans.py
# If intentionally regenerating reports already in this output folder:
# uv run python scripts/analyze_four_model_hans.py --overwrite
```

The user has confirmed the analysis ran, original numbers reproduced, and the saved analysis is accurate.

---

## 12. Final visualization work: scientific, automatically saved figures

Two complementary visualization outputs were created. They **read existing CSV/JSONs** rather than running or altering models.

### 12.1 Comprehensive **runtime** notebook — 16 figures

**Notebook:** `notebooks/DLBA_Comprehensive_Runtime_Visualizations.ipynb`  
**Inputs:** `results/runtime_analysis/`  
**Saved outputs:** `results/visualization/runtime_comprehensive/figures/*.png` and `*.svg`; consolidated tables and a figure index are saved under `results/visualization/runtime_comprehensive/` as well. PNGs are high-resolution; SVG labels remain editable.

| Figure stem | Purpose |
|---|---|
| `01_cpu_throughput` | Four CPU variants across the three-seed full HANS experiment |
| `02_paired_speedups` | Paired per-seed speedup relative to baselines |
| `03_seed_consistency` | Variation in measured throughput by seed |
| `04_artifact_sizes` | Saved model sizes; compression compared within each format |
| `05_accuracy_vs_throughput` | Speed/accuracy tradeoff in the same CPU experiment |
| `06_size_vs_throughput` | Storage/speed efficiency frontier |
| `07_cpu_process_rss` | Separate ready/after-inference process RSS comparisons |
| `08_batch32_forward_latency` | Measured batch-forward latency |
| `09_cpu_thread_tuning_full` | 4-thread vs 8-thread full-HANS CPU impact |
| `10_tuning_heatmaps` | Experiment 1A thread × batch grid; untested cells not invented |
| `11_experiment2_memory_lifecycle` | Process RSS at sequential execution stages |
| `12_experiment2_cpu_operator_hotspots` | Main PyTorch CPU operator self times |
| `13_experiment2_copy_conversion` | Copy/conversion operation overhead |
| `14_original_cuda_w8a8_performance` | Original CUDA-only FP32 vs W8A8 speed comparison |
| `15_original_cuda_w8a8_memory` | Original CUDA-only model/memory comparison |
| `16_repeat_level_stability` | Repeat-level timing variation |

**Design choices:** same colors by variant across all runtime plots; CUDA W8A8 is **always plotted separately from CPU**. Repeated FP32 CPU measurements from Experiments 1B and 3 are **measurements of the same fine-tuned model**, not new FP32 model variants.

### 12.2 Four-model **HANS behavior** script — 11 figures

**Script:** `scripts/plot_four_model_hans.py`  
**Inputs:** `results/analysis/four_model_comparison/` and optionally `results/runtime_analysis/experiment3_openvino/three_seed_results.csv`  
**Output:** `results/visualization/four_model_hans/` (each figure `.png` at 300 DPI plus editable `.svg`, alongside `figure_index.csv`, `plot_summary.csv` and `README_FIGURES.md`).

| Figure stem | Purpose |
|---|---|
| `01_overall_hans_accuracy` | Four-model mean HANS accuracy ± seed SD |
| `02_subgroup_accuracy_heatmap` | Six HANS subgroup accuracies × four models |
| `03_subgroup_accuracy_change` | Subgroup delta in percentage points vs FP32 |
| `04_seed_consistency` | HANS accuracy across seeds, including lexical non-entailment |
| `05_harmful_beneficial_flips` | Directional harmful/beneficial flip bars for all three quantized variants |
| `06_overall_flip_totals` | Pooled full-HANS harmful/beneficial flips |
| `07_entailment_margin_shifts` | Mean three-class entailment-margin shift by subgroup |
| `08_positive_margin_fraction` | Fraction of examples whose entailment margin increased |
| `09_prediction_agreement` | Three-class vs binary HANS agreement with FP32 |
| `10_harmful_flip_overlap` | Pairwise overlap of new errors across the quantizers |
| `11_cpu_speed_accuracy_tradeoff` | Full-HANS CPU speed vs accuracy; original CUDA W8A8 excluded |

**Behavior-plot colors:** FP32 `#355C9A` (blue), W8A8 `#C94F62` (coral), TorchAO weight-only `#DB783A` (amber), OpenVINO FP32 `#8360B3` (purple), OpenVINO INT8 `#10947E` (teal). Harmful flips and beneficial flips use separate semantic red/green cues where relevant.

**Plot command:**

```bash
uv run python scripts/plot_four_model_hans.py
```

The plotting script validates the presence/shape of the four-model tables, checks that accuracy differences agree with net flip counts, and **does not overwrite analysis CSVs**. Re-running the plotting script regenerates figure files in its own output directory.

---

## 13. Complete repository artifact and responsibility map

Below, `{seed}` denotes 42/123/456; relative paths are rooted at `Quantization-Spurious-Heuristics-in-Transformers/`.

### 13.1 Source code and documentation

| Repository path | Responsibility | Produces / consumes |
|---|---|---|
| `README.md` | Main project overview and experiment scope | High-level repository entry point |
| `docs/EXPERIMENTS_AND_RESULTS.md` | Original FP32/W8A8/weight-only accuracy and behavior methodology | Explains legacy HANS/MNLI results |
| `docs/README_FP32_VS_WEIGHT_ONLY_INT8_CPU.md` | CPU benchmark-specific methods and interpretation | Experiments 1/1A/1B |
| `docs/README_EXPERIMENT_UPDATES_2026-11-20.md` | Snapshot written before OpenVINO results were available | **Historical only; superseded by this comprehensive handoff for Experiment 3** |
| `scripts/benchmark_fp32_weight_only_cpu.py` | PyTorch FP32 vs TorchAO weight-only, pilots, tuning, full passes | `results/runtime_analysis/fp32_vs_weight_only_cpu/`, `tuning/` |
| `scripts/experiment2_cpu_profiler_memory.py` | Independent-worker memory timeline and CPU operator profiling | `results/runtime_analysis/experiment2_profiler/` |
| `scripts/experiment3_openvino_cpu.py` | Export FP32 OpenVINO IR, NNCF INT8, four-way CPU comparison | `models/experiment3_openvino/`, `results/runtime_analysis/experiment3_openvino/` |
| `scripts/evaluate_openvino_int8.py` | Cached OpenVINO HANS per-example inference and softmax probabilities | `results/hans/openvino_int8/` |
| `scripts/analyze_quantization_effects.py` | Original three-model HANS behavioral comparison | `results/analysis/` |
| `scripts/analyze_four_model_hans.py` | Pair-aligned four-model accuracy, flips, margin and overlap | `results/analysis/four_model_comparison/` |
| `scripts/plot_paper_figures.py` | Original HANS/MNLI paper-style plots | `results/visualization/paper_figures/` |
| `scripts/plot_four_model_hans.py` | 11 four-model HANS behavior figures | `results/visualization/four_model_hans/` |
| `notebooks/DLBA_Comprehensive_Runtime_Visualizations.ipynb` | 16 old+new runtime figures with consistent visual conventions | `results/visualization/runtime_comprehensive/` |
| `scripts/evaluate_weight_only_int8.py` | Original TorchAO weight-only HANS evaluation | `results/hans/weight_only_int8/` |
| `scripts/evaluate_weight_only_mnli.py` | Original weight-only MNLI validation | `results/mnli/weight_only_int8/` |
| `scripts/validate_fp32_cpu.py` | CPU/MPS class-label equivalence test | `results/hans/cpu_fp32_validation/` |
| `pyproject.toml`, `uv.lock` | Reproducible Python dependency declarations | Include `openvino` and `nncf` after `uv add openvino nncf` |
| `requirements.txt` | Existing dependency-list file | Preserve alongside `pyproject.toml` and `uv.lock` |
| `scripts/analyze_mnli_accuracy.py` | Existing MNLI accuracy aggregation/analysis | `results/mnli/`, `results/analysis/mnli_accuracy_*` |
| `scripts/generate_mnli_calibration.py` | Generate MNLI data for supplementary threshold calibration | `results/calibration/` |
| `scripts/select_calibration_thresholds.py` | Choose the W8A8 decision thresholds from calibration results | `results/calibration/selected_thresholds.json` |
| `scripts/evaluate_calibrated_mnli.py` | Evaluate threshold-based W8A8 MNLI decisions | `results/calibration/` |
| `scripts/evaluate_calibrated_hans.py` | Evaluate threshold-based W8A8 HANS decisions | `results/calibration/hans_calibration_evaluation.csv` |

### 13.2 Input artifacts and evaluation data

| Repository path | Purpose / provenance |
|---|---|
| `models/fp32/seed{seed}/final/` | Original trained BERT checkpoint/tokenizer; the *sole* source of neural weights in all new runtime tests |
| `models/experiment3_openvino/seed{seed}/fp32.xml`, `.bin` | Converted OpenVINO FP32 graph and weights |
| `models/experiment3_openvino/seed{seed}/int8_sym.xml`, `.bin` | NNCF symmetric per-channel INT8-compressed graph and weights |
| `models/experiment3_openvino/seed{seed}/model_export_metadata.json` | Source signatures and compression-validation metadata; important for reproducibility |
| `results/hans/fp32_seed{seed}_predictions.csv` | Original reference per-example HANS labels/probabilities/`pairID` and text |
| `results/hans/int8_seed{seed}_predictions.csv` | Original TorchAO dynamic W8A8 per-example HANS results |
| `results/hans/weight_only_int8/weight_only_int8_seed{seed}_predictions.csv` | TorchAO weight-only per-example HANS results |
| `results/hans/openvino_int8/openvino_int8_seed{seed}_predictions.csv` | New OpenVINO INT8 per-example predictions and three-class probabilities |
| `results/hans/openvino_int8/openvino_int8_seed{seed}_summary.json` | Overall/per-subgroup metrics, model metadata, benchmark cross-check |
| `results/mnli/` | Original MNLI matched/mismatched model evaluations; do not confuse with new OpenVINO HANS benchmark |
| `results/calibration/` | Supplementary W8A8 threshold-selection analysis |

### 13.3 Other existing repository notebooks and supporting artifacts

The following files are **already present in the supplied `tree` output**. They are included to help locate historical/auxiliary work; they are not invented additional experiments or newly required files.

| Existing path | Role in the repository |
|---|---|
| `notebooks/fp32_baseline.ipynb` | Original FP32 model baseline/training notebook |
| `notebooks/quantization.ipynb` | TorchAO quantization evaluation notebook |
| `notebooks/hans_dataset_analysis.ipynb` | HANS exploration and data checks |
| `notebooks/mnli_dataset_analysis.ipynb` | MNLI exploration and preprocessing checks |
| `notebooks/hans_evaluation.ipynb` | Existing HANS evaluation workflow |
| `notebooks/paired_fp32_int8_analysis.ipynb` | Original paired FP32–INT8 comparison |
| `notebooks/paired_fp32_int8_analysis_with_visuals.ipynb` | Paired comparison with visual figures |
| `notebooks/subcase_analysis.ipynb` | Drill-down on HANS subcases |
| `notebooks/DLBA_Research_Visualization_Notebook.ipynb` | Earlier research-visualization version |
| `notebooks/DLBA_Research_Visualizations_Clear.ipynb` | Cleaned research-visualization version |
| `notebooks/DLBA_Comprehensive_Runtime_Visualizations.ipynb` | Current comprehensive runtime-figure notebook |
| `notebooks/runtime_analysis/00_preflight.ipynb` | Original runtime environment/preflight checks |
| `notebooks/runtime_analysis/01_isolated_eager.ipynb` | Isolated eager runtime measurements |
| `notebooks/runtime_analysis/02_cuda_profiler.ipynb` | CUDA operator/profiler inspection |
| `notebooks/runtime_analysis/04_prediction_validation.ipynb` | Check runtime prediction consistency |
| `notebooks/runtime_analysis/05_review_results.ipynb` | Review original runtime outputs |
| `notebooks/runtime_analysis/config.json` | Original notebook runtime configuration |
| `notebooks/runtime_analysis/runtime_common.py`, `worker.py`, `run_experiments.py`, `compare_predictions.py`, `summarize_results.py` | Original CUDA benchmarking/worker/comparison support code |
| `notebooks/runtime_analysis/README.md`, `requirements_additional.txt` | Documentation and supplemental dependencies for original runtime notebooks |
| `models/fp32/seed{seed}/checkpoint-{12272,24544,36816}/` | Training checkpoints, optimizer/scheduler/RNG state and metadata; distinct from the `final/` inference checkpoints |
| `models/fp32/seed{seed}/trainer_state.json` | Training state per seed |
| `fp32_final.zip`, `fp32_seed123_final.zip`, `fp32_seed456_final.zip` | Existing top-level archived checkpoint artifacts; do not confuse with the model directories |
| `results.zip`, `results/runtime_analysis.zip` | Existing archived result files; not the live analysis output directories |
| `results/hans/paired_fp32_int8/` | Earlier paired-comparison tables, significance tests, bootstrap summaries, and figures |
| `results/hans/subcase_analysis/` | Lexical non-entailment subcase accuracy, flips, probability shifts and plot |
| `results/visualization/figures/` | Original multi-figure research visualization set and figure guide |
| `results/visualization/figures_clean/` | Cleaned original research visualization set |
| `results/visualization/paper_figures/` | Paper-style HANS performance figures and their CSV source data |
| `results/visualization/DLBA_*`, `results/visualization/README_visualization.md` | Existing consolidated visualization data tables/manifest and their documentation |
| `results/runtime_analysis/seed42/{fp32_eager,int8_eager}/` | Original completed CUDA eager benchmark, predictions, profiles and traces |
| `results/runtime_analysis/prediction_differences/`, `prediction_equivalence_checks.csv`, `prediction_equivalence_status.csv` | Diagnostics of original runtime prediction equivalence |
| `results/runtime_analysis/pair_eligibility_status.csv`, `benchmark_inclusion_status.csv`, `all_latency_samples.csv` | Original benchmark eligibility/inclusion decisions and detailed timing samples |

For full per-file names inside result directories, use the repository's own `tree` listing; the tables above group repeated seed files and PNG/SVG pairs by their purpose rather than inventing extra files.

### 13.4 Result directory ownership

```text
results/
├── analysis/                          # Legacy 3-model analysis CSVs
│   └── four_model_comparison/          # Experiment 3D; nine authoritative CSVs
├── hans/
│   ├── fp32_seed{seed}_predictions.csv
│   ├── int8_seed{seed}_predictions.csv
│   ├── weight_only_int8/               # TorchAO per-example HANS predictions
│   ├── openvino_int8/                  # Experiment 3C predictions and summaries
│   └── cpu_fp32_validation/            # FP32 CPU/MPS label check
├── mnli/                               # Prior MNLI evaluations
├── calibration/                        # Supplementary decision-threshold analysis
├── runtime_analysis/
│   ├── seed42/                         # Original CUDA eager FP32 and eager INT8 outputs
│   ├── fp32_vs_weight_only_cpu/        # Initial 4-thread CPU benchmark
│   ├── tuning/
│   │   ├── t2_b32/
│   │   ├── t4_b16/
│   │   ├── t4_b32/
│   │   ├── t4_b64/
│   │   ├── t8_b32/
│   │   ├── t8_b64/                     # Experiment 1A, seed42 n=1000 pilot grid
│   │   └── full_t8_b32/               # Experiment 1B, 30k ×3 seeds ×3 repeats
│   ├── experiment2_profiler/
│   │   └── seed42_t8_b32_n256/        # Detailed memory and operator diagnostics
│   └── experiment3_openvino/
│       ├── seed42_t8_b32_n512/        # OpenVINO pilot
│       ├── seed42_t8_b32_n30000/
│       ├── seed123_t8_b32_n30000/
│       ├── seed456_t8_b32_n30000/    # Full OpenVINO benchmark
│       └── three_seed_results.csv     # Main 3-seed runtime comparison
└── visualization/
    ├── paper_figures/                 # Original 3-model HANS/MNLI plots
    ├── runtime_comprehensive/         # 16 full experimental-runtime figures
    └── four_model_hans/               # 11 four-model behavior figures
```

**Preservation and version-control rules:**

- Original three-model CSVs and prior CPU/CUDA reports are independent of Experiment 3C/3D; they are **not meant to be rewritten**.
- Small CSV/JSON summaries, chart images and scripts can be reviewed and committed to the `feature/int8-runtime-optimization` branch.
- Large `models/fp32/` checkpoints and `models/experiment3_openvino/` `.bin`/`.xml` caches are local large artifacts; check Git staging/ignore configuration before any push. This tree listing alone does **not** establish what Git currently tracks or ignores. Cached OpenVINO exports can be regenerated from FP32 checkpoints and the conversion script.
- The four-model analysis script refuses to overwrite an existing nonempty results directory unless `--overwrite` is explicitly used; the plotting scripts intentionally regenerate **plot files**, not per-example source predictions.
- All paths and numerical tables represent **completed runs**; they are not a request to re-execute them. Command snippets are for reproducibility and provenance.

---

## 14. How to interpret the findings correctly

### 14.1 Speed improvements are implementation-dependent

“INT8 is smaller” does not imply “INT8 will be faster.” TorchAO weight-only reduced serialized weights but failed to improve full-dataset PyTorch CPU throughput meaningfully. Its operator trace suggests conversion overhead. NNCF + OpenVINO achieved a large performance improvement **as a complete compressed inference stack**. The experiment does not isolate NNCF quantization math from the OpenVINO graph compiler and kernels.

### 14.2 The exact comparisons that are scientifically defensible

- **Within PyTorch CPU:** FP32 vs TorchAO weight-only, same seed/threads/batch.
- **Within OpenVINO CPU:** FP32 vs INT8_SYM, same seed/threads/batch, to assess compressed pipeline execution.
- **Cross-stack CPU:** TorchAO weight-only vs OpenVINO INT8, to compare practical deployment options rather than a single causal factor.
- **Within original CUDA:** eager FP32 vs W8A8, to explain why the first approach was unsuccessful on that GPU.
- **Behavior:** paired `pairID` comparisons against each seed's original FP32, never unpaired aggregate matching.

### 14.3 HANS is not explained by one overall accuracy number

Overall HANS accuracy can conceal group-specific deterioration because changes in different subgroups may offset each other. The **lexical-overlap/non-entailment** subgroup was particularly vulnerable under W8A8. The new four-model flip, agreement and margin tables provide the detail to evaluate OpenVINO behavior with exactly the same definitions. These data are evidence about **outputs and correlations**, not proof of model internals or a particular learned feature.

### 14.4 Device and memory limitations

- CUDA throughput and CPU throughput should remain **separate performance stories**.
- Absolute `.pt` vs OpenVINO IR file sizes depend partly on **serialization format**; within-framework compression ratios are stronger evidence.
- RSS changes with model loading, page faults, cached memory and allocator behavior. RSS snapshots should not be described as isolated weight memory or peak memory.
- A per-seed run reflects one fine-tuned checkpoint; three-seed SD describes **observed variation**, not a population confidence interval.
- HANS counts pooled across seeds reuse the same sentences. The effective data unit for paired inference comparisons is a **seed–example** pair, not a distinct independent observation.

---

## 15. Teammate orientation: recommended reading order

1. **Sections 1–2:** understand the shared model/checkpoints, HANS labels, and fair controls.
2. **Sections 3–4:** understand why earlier quantization motivated a more careful runtime and robustness investigation.
3. **Sections 5–8:** see that TorchAO weight-only compressed storage, CPU tuning helped both variants, and operator profiling suggested extra conversion work.
4. **Section 9:** follow the OpenVINO/NNCF model-export pipeline, verification fix, pilot and completed three-seed runtime evidence.
5. **Sections 10–11:** understand why per-example OpenVINO CSVs were generated and exactly how the four-model HANS/flip/margin statistics were computed.
6. **Sections 12–13:** locate every reproducible figure and canonical script/data artifact.
7. **Section 14:** understand which statements are findings and which are interpretations/limitations.

**Primary result file:** `results/runtime_analysis/experiment3_openvino/three_seed_results.csv`  
**Primary behavior summary directory:** `results/analysis/four_model_comparison/`  
**Primary behavior plots:** `results/visualization/four_model_hans/`  
**Primary old/new runtime plots:** `results/visualization/runtime_comprehensive/`  
**Original three-model research description:** `docs/EXPERIMENTS_AND_RESULTS.md`

---

**Placement:** Save this newly generated handoff as `docs/DLBA_Experiment_Handoff_2026-11-20.md`. The supplied repository tree does **not** currently contain this new document.

**Final research finding:** On our specific BERT/HANS CPU workload, **OpenVINO with NNCF symmetric per-channel INT8 weight compression** produced a repeatable **~2.58× mean paired speedup vs PyTorch FP32** and an OpenVINO artifact **~74.76% smaller** than OpenVINO FP32, with a **~0.32 pp mean HANS overall accuracy reduction**. A separate, paired four-model HANS analysis was completed and reproduced to study subgroup errors, harmful/beneficial flips, agreement and entailment margins. These are results of the tested execution implementations and checkpoints, not universal claims about INT8 or all transformers.
