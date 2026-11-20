# Experiment Updates and Results — Quantization & Spurious Heuristics in Transformers

**Research project:** Deep Learning Beyond Accuracy (DLBA) — NCSU  
**Updated:** November 20, 2026  
**Purpose:** Teammate handoff covering the new quantization analyses and all completed FP32 vs weight-only INT8 CPU efficiency experiments, including tuning and profiling.  
**Status:** Three-model behavioral evaluation, CPU FP32 check, supplementary threshold study, baseline CPU runtime benchmark, Experiment 1A (tuning), Experiment 1B (validation), and Experiment 2 (profiler) **completed**. Alternative backends (OpenVINO/ONNX Runtime) **not yet tested**.

## TL;DR — The main findings

1. **Standard MNLI accuracy barely changed**, but HANS exposed greater behavioral differences: mean lexical-overlap **non-entailment** accuracy changed by **−4.21 percentage points (pp) for W8A8** and **−0.93 pp for weight-only INT8**, relative to FP32.
2. **Weight-only INT8 preserved HANS performance better than W8A8**, but the methods use different implementations; we cannot isolate activation quantization as the sole cause.
3. **Weight-only INT8 reduced serialized model-state size by 58.47%.** This is an on-disk representation measurement, **not** proof of lower process RAM.
4. In an identical CPU benchmark, **weight-only INT8 and FP32 had roughly equal full-dataset throughput**: original 4-thread setting **81.65 vs 81.98 examples/s**, and tuned 8-thread setting **155.15 vs 153.70 examples/s**, respectively (three-seed means).
5. **Thread tuning, not quantization, produced the large speed improvement**: the 8-thread configuration raised throughput approximately **87.5% (FP32)** and **90.0% (weight-only)** relative to 4 threads.
6. **Experiment 2 explained much of the puzzling memory measurement and identified likely conversion overhead:** the inference-ready quantized worker initially reported ~**422 MiB** more process RSS; the gap declined to ~**107 MiB after inference** and ~**82 MiB after allocator trimming**. Profiler traces showed many more copy/conversion operations in the weight-only path. This does **not** establish a model-only RAM saving or a causal timing breakdown.

> **Bottom line:** With the tested TorchAO implementation, weight-only INT8 is an attractive **accuracy–storage** tradeoff. **A major CPU inference speedup has not been demonstrated.** A backend change (such as OpenVINO or ONNX Runtime) is a proposed follow-up, not a completed result.

---

## 1. Shared study setup

- **Model:** `google-bert/bert-base-uncased` fine-tuned for 3-class natural language inference (entailment, neutral, contradiction).
- **Fine-tuning:** 3 epochs, learning rate `2e-5`, batch size 32, maximum sequence length 128; independent checkpoints for **seeds 42, 123, 456**.
- **MNLI:** validation matched (**9,815** examples) and mismatched (**9,832** examples).
- **HANS:** **30,000** examples: lexical overlap, subsequence, constituent heuristics × entailment/non-entailment, **5,000 items per subgroup**.
- **Precision / inference configurations:**
  - **FP32:** unquantized model (baseline).
  - **W8A8:** TorchAO `Int8DynamicActivationInt8WeightConfig` — dynamically quantized activations and INT8 weights.
  - **Weight-only INT8:** TorchAO `Int8WeightOnlyConfig` — quantized weights, higher-precision activations.
- **Quantization coverage:** the two quantized configurations each target **74 linear layers**.
- **HANS decision rule:** model predicts one of 3 MNLI classes by argmax; class `0` → HANS *entailment*, classes `1`/`2` → *non-entailment*. This is **not** a binary `P(entailment) > 0.5` rule.
- **Hardware distinction:** the three-model behavior evaluation was recorded using Apple MPS for original FP32 predictions and CPU for quantized predictions. Additional FP32 CPU validation checked label equivalence. **The new runtime tuning/profiling was performed separately on Pop!_OS/Linux CPU using identical model-forward workloads**. Do not compare absolute CPU throughput with earlier CUDA/MPS results.

## 2. Updated three-model behavior / accuracy experiments

**All numbers below are arithmetic means across the three seeds.** Changes are relative to FP32 and expressed in *percentage points* (pp).

| Evaluation | FP32 accuracy | W8A8 accuracy | Weight-only accuracy | W8A8 − FP32 | Weight-only − FP32 |
|---|---:|---:|---:|---:|---:|
| MNLI matched | 84.6358% | 84.7444% | 84.6561% | +0.1087 pp | +0.0204 pp |
| MNLI mismatched | 84.8454% | 84.8895% | 84.8827% | +0.0441 pp | +0.0373 pp |
| HANS overall | 57.8989% | 57.0333% | 57.6967% | −0.8656 pp | −0.2022 pp |
| HANS lexical overlap / non-entailment | 35.7600% | 31.5533% | 34.8267% | **−4.2067 pp** | **−0.9333 pp** |

**Why it matters:** relying only on MNLI would make the three models look almost interchangeable; HANS reveals different failure patterns. The weight-only variant remains closer to FP32 on HANS, particularly on lexical-overlap counterexamples.

### Subgroup changes relative to FP32

| HANS heuristic | Gold label | W8A8 change | Weight-only change |
|---|---|---:|---:|
| Lexical overlap | Entailment | +0.5000 pp | +0.0400 pp |
| Lexical overlap | Non-entailment | **−4.2067 pp** | **−0.9333 pp** |
| Subsequence | Entailment | +0.1333 pp | +0.0067 pp |
| Subsequence | Non-entailment | −0.9867 pp | −0.2333 pp |
| Constituent | Entailment | +0.0867 pp | +0.0267 pp |
| Constituent | Non-entailment | −0.7200 pp | −0.1200 pp |

**Floor-effect caution:** FP32 already achieves only **5.52%** on subsequence non-entailment and **10.32%** on constituent non-entailment (versus **35.76%** for lexical overlap). Smaller absolute drops for these groups do not necessarily indicate weaker heuristic dependence: accuracy has less room to fall.

### Prediction-flip analysis: lexical-overlap non-entailment

*Harmful* = FP32 correct → quantized wrong. *Beneficial* = FP32 wrong → quantized correct. Aggregated counts below span **three seed-evaluations of the same 5,000 HANS items** (15,000 seed-item evaluations, **not 15,000 distinct examples**).

| Seed | W8A8 harmful | W8A8 beneficial | Weight-only harmful | Weight-only beneficial |
|---|---:|---:|---:|---:|
| 42 | 211 | 32 | 45 | 14 |
| 123 | 304 | 11 | 81 | 6 |
| 456 | 171 | 12 | 40 | 6 |
| **Total** | **686** | **55** | **166** | **26** |

- W8A8 lost a net **631 correct predictions** across the three evaluations; weight-only lost **140**.
- Harmful flip overlap: **147** harmful under **both** variants; **539** harmful only for W8A8; **19** only for weight-only. These are paired *seed-item* counts.

### Three-class entailment-margin analysis

For each example, define `margin = P(entailment) − max(P(neutral), P(contradiction))`. The reported **shift = quantized margin − FP32 margin**; a positive shift favors entailment relative to the most competitive alternative class. This is **not** the binary `P(E)−P(non-E)` margin.

On HANS *non-entailment* items:

| Heuristic | W8A8 mean shift | Weight-only mean shift | W8A8 shift positive | Weight-only shift positive |
|---|---:|---:|---:|---:|
| Lexical overlap | +0.0677 | +0.0152 | 82.32% | 75.09% |
| Subsequence | +0.0192 | +0.0046 | 53.82% | 53.91% |
| Constituent | +0.0062 | +0.0018 | 45.92% | 52.67% |

**Interpretation:** W8A8 shifts lexical-overlap non-entailment items toward entailment more strongly than weight-only, consistent with its larger loss of correct non-entailment predictions. This is **behavioral evidence**, not direct proof of internal shortcut-feature usage.

### Additional validations and exploratory decision-rule test

- **FP32 CPU vs MPS check:** **90,000/90,000 three-class predicted labels matched** across the 3 seeds. This validates class-label agreement, **not** numerical equality of logits or probabilities.
- **Threshold experiment (supplementary, W8A8 only):** entailment thresholds chosen from MNLI matched were **0.25, 0.35, 0.24** for seeds 42, 123, 456. Relative to W8A8's original argmax rule, MNLI mismatched changed **+0.08 pp**, HANS overall **−2.37 pp**, and lexical non-entailment **−8.95 pp**. This changes the **decision rule**, not the probability distribution or trained weights; it is **not** a comparison of quantization types.

Original behavior-study documentation: `docs/EXPERIMENTS_AND_RESULTS.md`. Main updated summary CSVs: `results/analysis/`.

---

## 3. Runtime benchmarking — initial CPU baseline (November 20)

**Question:** Does TorchAO *weight-only INT8* outperform FP32 on the **same CPU**, with identical inputs and inference setup?

**Procedure:** each model/seed runs in a separate Python process. Input HANS pairs are tokenized with max length 128 and **pre-collated** into identical batches. Measured throughput times **model forward calls only**, excluding tokenization, loading, quantization, and output processing. Per-seed throughput is the **median of 3 complete HANS passes**; reported means average the 3 seed medians. Batch latency is measured on one representative padded batch and should not be conflated with full-dataset throughput.

**Configuration:** PyTorch eager CPU, **4 intra-op threads**, **batch size 32**, 3 repeats, 30,000 HANS examples per seed. An initial **3,000-example seed-42 pilot** also passed (81.110 FP32 vs 83.614 weight-only examples/s).

Run from the repository root:

```bash
uv run python scripts/benchmark_fp32_weight_only_cpu.py \
  --seeds 42 123 456 --num-examples 30000 --repeats 3
```

| Seed | FP32 throughput (examples/s) | Weight-only throughput (examples/s) | Weight-only vs FP32 |
|---|---:|---:|---:|
| 42 | 82.461 | 83.201 | +0.90% |
| 123 | 82.625 | 83.142 | +0.63% |
| 456 | 80.841 | 78.600 | −2.77% |
| **Mean** | **81.976** | **81.648** | **−0.40%** (ratio of means) |

- **Serialized `state_dict` size:** weight-only **58.471% smaller** than FP32 for each seed.
- **Inference-ready process RSS** (weight-only − FP32): **+422.309**, **+420.254**, **+419.762 MiB** for seeds 42, 123, 456 (mean **+420.775 MiB**).
- Sampled-batch latency ratios (FP32 time / weight-only time): **1.027×, 1.045×, 1.053×**; **these are not equivalent to whole-dataset throughput speedups**.

**Finding:** major saved-state compression, **no consistent full-HANS CPU throughput advantage**, and an unexpectedly higher **process-level** RSS measurement that motivated Experiment 2. RSS includes Python, model loading, tokenized inputs, caching, and allocator behavior; it is **not** tensor-only or peak model RAM.

Saved to `results/runtime_analysis/fp32_vs_weight_only_cpu/` with `seed*_n30000.json`, `benchmark_results_n30000.csv`, `paired_comparison_n30000.csv`, and `01_serialized_size`, `02_process_rss`, `03_batch_latency`, `04_throughput` PNG/SVG figures. Original `n3000` pilot files remain alongside the `n30000` outputs.

---

## 4. Experiment 1A — CPU thread / batch tuning (completed)

**Goal:** determine whether inference settings, rather than quantization itself, limit throughput. This **pilot uses seed 42, 1,000 examples, 2 repeats**, and tests **both FP32 and weight-only INT8** for each configuration.

| CPU threads | Batch size | FP32 examples/s | Weight-only examples/s | Weight-only change vs FP32 |
|---:|---:|---:|---:|---:|
| 4 | 32 | 76.543 | 70.344 | −8.10% |
| 2 | 32 | 39.510 | 40.057 | +1.38% |
| **8** | **32** | **143.704** | **149.865** | **+4.29%** |
| 4 | 16 | 76.864 | 79.421 | +3.33% |
| 4 | 64 | 75.125 | 71.588 | −4.71% |
| 8 | 64 | 145.695 | 133.290 | −8.51% |

**Selected for full validation:** **8 threads, batch size 32**, because it had the highest measured weight-only throughput in this small tuning grid and showed a positive edge over FP32. Note that the 1,000-example tuning results are **not** directly interchangeable with the full 30,000-example measurements; some variation is expected.

Reproduction:

```bash
for config in "4 32" "2 32" "8 32" "4 16" "4 64" "8 64"; do
  read -r threads batch <<< "$config"
  uv run python scripts/benchmark_fp32_weight_only_cpu.py \
    --seeds 42 --num-examples 1000 --repeats 2 \
    --warmup 4 --latency-iterations 10 \
    --threads "$threads" --batch-size "$batch" \
    --out-dir "results/runtime_analysis/tuning/t${threads}_b${batch}"
done
```

Outputs: `results/runtime_analysis/tuning/t{threads}_b{batch}/` (separate JSON, CSV and PNG/SVG outputs for each configuration).

---

## 5. Experiment 1B — full tuned CPU validation (completed)

**Goal:** test whether Experiment 1A's apparent weight-only gain persists over **all 30,000 HANS examples**, **all 3 seeds**, with **3 repeated passes** and identical 8-thread, batch-32 settings.

```bash
uv run python scripts/benchmark_fp32_weight_only_cpu.py \
  --seeds 42 123 456 \
  --num-examples 30000 --repeats 3 \
  --threads 8 --batch-size 32 \
  --out-dir results/runtime_analysis/tuning/full_t8_b32
```

| Seed | FP32 throughput (examples/s) | Weight-only throughput (examples/s) | Weight-only change vs FP32 |
|---|---:|---:|---:|
| 42 | 155.935 | 159.550 | +2.32% |
| 123 | 152.755 | 159.217 | +4.23% |
| 456 | 152.421 | 146.691 | −3.76% |
| **Mean** | **153.704** | **155.153** | **+0.94%** (ratio of means) |

**Tuning impact relative to the original 4-thread/32-batch full benchmark:**

| Model | Four-thread mean throughput | Eight-thread mean throughput | Increase |
|---|---:|---:|---:|
| FP32 | 81.976 ex/s | 153.704 ex/s | **+87.5%** (1.87×) |
| Weight-only INT8 | 81.648 ex/s | 155.153 ex/s | **+90.0%** (1.90×) |

- All three seeds again show **58.471% smaller serialized state** for weight-only INT8.
- Tuned sampled-batch latency speedup (FP32/weight-only): **1.055×, 1.055×, 1.058×** across seeds (not whole-HANS speedups).
- The reported inference-ready weight-only **RSS excess persists** at **+423.148, +423.141, +423.016 MiB** across the three seeds.

**Finding:** moving from 4 to 8 threads produces an approximately **1.9× throughput gain for *both* models**. Weight-only is faster in 2 seeds, slower in the third, and only **~0.94% faster on the mean**. Thus **a meaningful, repeatable advantage attributable to weight-only INT8 is not established**.

Results and charts: `results/runtime_analysis/tuning/full_t8_b32/`.

---

## 6. Experiment 2 — CPU operator profiling and memory lifecycle (completed)

**Goal:** explain (a) why the quantized worker initially appeared to consume much more resident memory and (b) why compressed weights did not produce a major speedup.

**Setup:** seed **42**, **8 threads**, batch **32**, **256 HANS examples**; FP32 and weight-only each profiled in separate processes. Diagnostic workload intentionally smaller than the full throughput benchmark.

```bash
uv run python scripts/experiment2_cpu_profiler_memory.py \
  --seed 42 --threads 8 --batch-size 32 --num-examples 256
```

### 6.1 Observed memory lifecycle

All values below are **whole-process resident set size (RSS)** in MiB, not tensor-only memory.

| Checkpoint | FP32 RSS | Weight-only RSS | Weight-only − FP32 |
|---|---:|---:|---:|
| Inputs prepared, before loading model | 944.77 | 944.63 | −0.14 |
| Inference-ready, after model setup/quantization | 1,000.57 | 1,422.47 | **+421.90** |
| After warm-up | 1,357.50 | 1,468.88 | **+111.37** |
| After unprofiled inference | 1,388.68 | 1,495.32 | **+106.64** |
| After garbage collection and glibc `malloc_trim()` | 1,355.99 | 1,437.87 | **+81.88** |

**Key interpretation:** the large **+422 MiB** inference-ready difference is **not** a stable measurement of extra INT8 weight memory. The FP32 process's RSS grew substantially as inference accessed model data, bringing the gap down to ~**107 MiB before trimming**. Trimming then reduced the gap by only **~24.77 MiB**, to **~81.88 MiB**. **Do not claim that `malloc_trim()` alone reclaimed 340 MiB.** The memory lifecycle and allocator behavior, not just parameter precision, determine RSS. We still have **not** demonstrated reduced steady-state or peak RAM use by weight-only INT8.

The comparison JSON additionally reports **+422.031 MiB as the difference of setup RSS increments from the pre-load baseline**; this is slightly different from the **+421.90 MiB absolute RSS gap** shown in the table.

### 6.2 CPU operator attribution

The profiler's **self CPU time** (accumulated profiled operator time) identifies these main paths:

| Category/operator | FP32 | Weight-only INT8 |
|---|---:|---:|
| Dominant matrix multiplication operator | `aten::addmm` | `aten::mm` |
| Matrix multiplication self CPU time | 3,118.1 ms | 3,179.6 ms |
| `aten::copy_` self CPU time | 44.8 ms | **203.1 ms** |
| `aten::mul` self CPU time | Not a leading operator in uploaded summary | **71.0 ms** |
| `aten::_to_copy` calls | **16** | **1,200** |

**Interpretation:** the weight-only path performs substantially more copying/conversion and elementwise work, which plausibly offsets some benefits from compressed stored weights. However, **the profiler does not isolate the precise dequantization mechanism, hardware instruction path, or causal impact of each operator on unprofiled wall-clock latency**. Operator self times are **not** additive as if they were whole-dataset inference latency, and profiling changes execution characteristics.

The short unprofiled diagnostic's speed ratio was **1.0249× (weight-only/FP32)** on 256 examples. **Use Experiment 1B, not this short run, for the principal throughput statement.**

Saved under `results/runtime_analysis/experiment2_profiler/seed42_t8_b32_n256/`:

- `experiment2_comparison.json`, `experiment2_memory_timeline.csv`, `experiment2_operator_attribution.csv`;
- per-model diagnostic JSONs, memory-stage CSVs, operator CSVs;
- figures: `01_memory_lifecycle`, `02_model_setup_rss`, `03_cpu_operator_attribution`, `04_cpu_operator_per_batch` (PNG and SVG).

---

## 7. What we can and cannot conclude

| Question | Evidence-supported conclusion |
|---|---|
| Does weight-only INT8 preserve standard MNLI accuracy? | **Yes**, within ~0.04 pp of FP32 on both splits in the three-seed means. |
| Is it more accurate on HANS than W8A8? | **Yes** in the observed study, including overall HANS and lexical-overlap non-entailment. |
| Is activation quantization alone responsible? | **Not established**: the two quantization configurations have implementation differences. |
| Does weight-only INT8 reduce *serialized* model size? | **Yes: 58.471%**, as measured using `torch.save(model.state_dict())`. |
| Does it provide a major CPU throughput gain? | **No**: 8-thread full-HANS mean throughput difference is **+0.94%**, inconsistent across seeds. |
| Does CPU tuning improve throughput? | **Yes, for both models**: ~1.9× between the tested 4-thread and 8-thread settings. |
| Does weight-only INT8 reduce resident CPU process memory? | **Not shown**. The measured worker RSS remains higher, with strong dependence on model setup and memory-access stages. |
| Have we established a faster quantized backend? | **No**. OpenVINO and ONNX Runtime are **ideas for later experimentation**, not completed comparisons. |

**Scope notes:** The HANS items are reused across training seeds; pooled flip counts are *seed-item evaluations*, not independent unique observations. Three seeds do not support strong population-level statistical generalization. CPU/Apple MPS/CUDA results have different hardware and kernels and must **not** be treated as a direct three-way speed ranking. Runtime numbers cover model forward time, **not** tokenization + load + serving latency.

## 8. Repository map and deliverables

```text
.
├── README.md
├── docs/
│   └── EXPERIMENTS_AND_RESULTS.md         # Prior 3-model study methodology/results
├── scripts/
│   ├── benchmark_fp32_weight_only_cpu.py  # Baseline + tuning benchmark
│   ├── experiment2_cpu_profiler_memory.py # Diagnostic profiling + RSS timeline
│   ├── evaluate_weight_only_int8.py
│   ├── evaluate_weight_only_mnli.py
│   ├── analyze_quantization_effects.py
│   └── plot_paper_figures.py
└── results/
    ├── analysis/                         # Three-model MNLI/HANS comparison CSVs
    ├── calibration/                      # Supplementary threshold experiment
    ├── hans/                             # FP32/W8A8/weight-only prediction records
    ├── mnli/                             # Validation evaluations
    ├── visualization/paper_figures/      # Paper-oriented comparison plots
    └── runtime_analysis/
        ├── fp32_vs_weight_only_cpu/      # Original CPU pilot + 3-seed full benchmark
        ├── tuning/
        │   ├── t4_b32/ t2_b32/ t8_b32/
        │   ├── t4_b16/ t4_b64/ t8_b64/  # Experiment 1A
        │   └── full_t8_b32/             # Experiment 1B
        └── experiment2_profiler/
            └── seed42_t8_b32_n256/      # Experiment 2
```

**Recommended teammate reading order:** this file → `docs/EXPERIMENTS_AND_RESULTS.md` for model behavior → `results/runtime_analysis/tuning/full_t8_b32/paired_comparison_n30000.csv` for tuned speed → Experiment 2's memory timeline and operator CSVs.

### Reproducing the existing runs

Run all commands from the repository root with the project's Python environment (`uv run`) and locally available checkpoints at `models/fp32/seed{seed}/final`.

- **Do not rerun training.** All runtime experiments use the saved fine-tuned FP32 checkpoints.
- **Do not overwrite existing outputs.** The benchmark script normally reuses matching per-model JSONs; `--overwrite` deliberately replaces measurements. Experiment 2 refuses to run into a nonempty output directory unless `--overwrite` is used.
- **For experiment interpretation**, use the saved CSV/JSON rather than the chart labels alone. CPU profiling is diagnostic; the three-seed, full-HANS tuned benchmark is the official runtime comparison.

## 9. Proposed follow-up — *not performed*

**Experiment 3 (planned):** try an optimized alternative inference implementation, such as **OpenVINO + NNCF** or **ONNX Runtime**, while retaining current TorchAO data as baselines. A rigorous comparison would use the same CPU and HANS inputs for **PyTorch FP32**, **TorchAO weight-only INT8**, **alternative-backend FP32**, and **alternative-backend quantized** models; verify MNLI/HANS labels and accuracy because quantization implementations may differ.

**Do not report any OpenVINO/ONNX performance numbers yet**: no such benchmark has been run.

---

### Evidence provenance

This handoff consolidates the repository's `README.md` and `docs/EXPERIMENTS_AND_RESULTS.md`, provided analysis files, actual November 20 terminal logs for the baseline, Experiment 1A and Experiment 1B, plus uploaded Experiment 2 JSON and CSV diagnostics. **Baseline and tuning detailed per-model runtime JSON/CSV files were not uploaded to this conversation**; their values here follow the printed benchmark results and previously shared runtime README. No new model inference was performed to produce this document.
