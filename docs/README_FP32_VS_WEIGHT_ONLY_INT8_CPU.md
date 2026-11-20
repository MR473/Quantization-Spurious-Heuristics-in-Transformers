# CPU Runtime Benchmark — FP32 vs Weight-Only INT8

**Project:** Quantization and Spurious Heuristics in Transformers (DLBA)  
**Experiment date:** November 20, 2026  
**Status:** Completed for seeds 42, 123, and 456  
**Benchmark script:** `scripts/benchmark_fp32_weight_only_cpu.py`  
**Output directory:** `results/runtime_analysis/fp32_vs_weight_only_cpu/`

## 1. Motivation and research question

Our earlier evaluation compared three inference configurations of the same MNLI-fine-tuned BERT-base-uncased checkpoints: FP32, dynamic activation-and-weight INT8 (W8A8), and weight-only INT8. Weight-only INT8 retained substantially more of the original model's HANS accuracy than W8A8 while achieving similar MNLI accuracy. We therefore investigated the **runtime and memory cost of weight-only INT8 relative to its FP32 baseline**, rather than assuming that INT8 necessarily makes inference faster.

The question for this benchmark is:

> When evaluated under identical CPU inference conditions, does TorchAO weight-only INT8 reduce serialized model size, process memory usage, batch latency, and/or full-dataset inference time relative to FP32?

The existing accuracy comparisons are documented in `docs/EXPERIMENTS_AND_RESULTS.md`. **This benchmark is an efficiency experiment, not a new accuracy experiment.** No checkpoints were fine-tuned, and no HANS predictions were revalidated for accuracy here.

## 2. Benchmark configurations

| Setting | FP32 | Weight-only INT8 |
|---|---|---|
| Checkpoint | `models/fp32/seed{seed}/final` | Same FP32 checkpoint, quantized at runtime |
| Base model | MNLI-fine-tuned BERT-base-uncased | Same |
| Precision | Original FP32 model | TorchAO `Int8WeightOnlyConfig()` |
| Quantized linear layers | 0 | 74 (asserted by the script) |
| Device and execution | CPU, PyTorch eager | CPU, PyTorch eager |
| Fine-tuning seeds | 42, 123, 456 | 42, 123, 456 |
| Evaluation input | HANS premise/hypothesis pairs | Identical pairs, order, and pre-collated batches |
| Batch size | 32 | 32 |
| Max token length | 128 | 128 |
| CPU threads / interop | 4 / 1 | 4 / 1 |
| Warm-up forwards | 8 | 8 |
| Batch latency iterations | 40 | 40 |
| Full HANS passes per seed | 3 | 3 |

The experiment was run on the user's Pop!_OS/Linux CPU environment. The benchmark JSON records the host's Python, PyTorch, architecture, and platform information. Exact processor specifications are **not established by the shared terminal log** and should be taken from the local host rather than inferred.

### Measurement protocol

1. Load the saved FP32 HANS reference CSV for a seed, using the `premise`, `hypothesis`, and `pairID` fields as inputs.
2. Tokenize the selected examples identically for both configurations (`max_length=128`, truncation enabled) and pre-collate all batches.
3. Launch **a separate Python worker process** for each configuration to reduce direct cross-configuration memory contamination.
4. Load the seed's FP32 checkpoint. For the weight-only worker, apply `quantize_(model, Int8WeightOnlyConfig())`, then verify 74 quantized linear layers.
5. Record process RSS before loading the model and again after the model is inference-ready (after `gc.collect()`).
6. Warm up with a representative batch, time 40 forward passes on that same batch, and take the median batch-forward latency.
7. Run **three complete passes** over the selected HANS inputs, timing model forwards only; report median full-pass time and median examples per second.
8. Serialize `model.state_dict()` into a temporary `.pt` file and record its file size in MiB.

**Timing scope:** model forward calls only. Tokenization, batch collation, dataset loading, model loading, quantization setup, serialization, and accuracy postprocessing are **not** included in the reported throughput. Each pass uses 938 batches for the full 30,000 examples.

**Memory scope:** Linux **resident set size (RSS)** for the *whole worker process* at the inference-ready checkpoint. This includes PyTorch, Python, tokenized inputs, allocator behavior, and model/quantization setup; it is **not** a direct tensor-memory accounting or a peak-inference-memory measurement.

## 3. Experiments performed

### 3.1 Smoke/pilot benchmark — 3,000 HANS examples

We first ran seed 42 on a fixed 3,000-example HANS subset (random state 2026, sorted back into reference order) to verify that loading, quantization, measurement, saving, and plotting completed successfully.

```bash
uv run python scripts/benchmark_fp32_weight_only_cpu.py \
  --seeds 42 \
  --num-examples 3000 \
  --repeats 3
```

| Metric | FP32 | Weight-only INT8 |
|---|---:|---:|
| Median throughput | 81.110 examples/s | 83.614 examples/s |
| Median full-pass time (from log) | ~36.99 s | ~35.88 s |
| Relative batch-latency ratio (FP32 / INT8) | — | 1.045× |
| Serialized model-size reduction | — | 58.471% |
| Difference in inference-ready process RSS (INT8 − FP32) | — | +421.375 MiB |

The 3,000-example run was a **pipeline check**, not the final estimate.

### 3.2 Full benchmark — 30,000 HANS examples × three seeds

We then repeated the comparison for each independently fine-tuned checkpoint on the complete HANS evaluation set:

```bash
uv run python scripts/benchmark_fp32_weight_only_cpu.py \
  --seeds 42 123 456 \
  --num-examples 30000 \
  --repeats 3
```

This produces **six isolated model/seed worker results**, each based on three timed full-data passes. The same 30,000 HANS examples are reused across seeds; they are **not** independent datasets.

## 4. Full-benchmark results

### 4.1 Throughput and latency

All throughput numbers below are the **median of three full passes for a single seed**; the final row is the arithmetic mean of those three seed-level medians.

| Seed | FP32 throughput (examples/s) | Weight-only throughput (examples/s) | INT8 throughput change | Batch latency speedup (FP32 / INT8) |
|---|---:|---:|---:|---:|
| 42 | 82.461 | 83.201 | +0.90% | 1.027× |
| 123 | 82.625 | 83.142 | +0.63% | 1.045× |
| 456 | 80.841 | 78.600 | −2.77% | 1.053× |
| **Mean of seed results** | **81.976** | **81.648** | **−0.40%**¹ | **1.042×**² |

¹ Calculated from the ratio of the two mean throughput values.  
² Arithmetic mean of the three reported batch-latency speedup ratios; equivalent to an approximately 4% improvement on the *single sampled batch*, not a 4% whole-dataset speedup.

**Interpretation:** Weight-only INT8 has **approximately unchanged full-HANS CPU throughput**. It was slightly faster for seeds 42 and 123, but slower for seed 456. The single-batch latency measurement favored weight-only INT8 in all seeds; that microbenchmark does **not** guarantee faster end-to-end processing over batches with different padded sequence lengths.

For transparency, the median times from the full three-pass logs were approximately:

| Seed | FP32 median full-pass time | Weight-only median full-pass time |
|---|---:|---:|
| 42 | 363.808 s | 360.571 s |
| 123 | 363.088 s | 360.829 s |
| 456 | 371.101 s | 381.680 s |

### 4.2 Serialized size and memory

| Seed | Serialized state reduction with weight-only INT8 | Inference-ready process RSS change (INT8 − FP32) |
|---|---:|---:|
| 42 | 58.471% | +422.309 MiB |
| 123 | 58.471% | +420.254 MiB |
| 456 | 58.471% | +419.762 MiB |
| **Mean** | **58.471% smaller** | **+420.775 MiB** |

**Serialized model size:** Saving the quantized `state_dict` resulted in a file that was approximately **58.47% smaller** than the FP32 state. This is a clear model-storage benefit under this serialization method; it does not by itself measure CPU RAM consumption or total deployment footprint.

**Process memory:** Despite the smaller serialized state, the benchmark reported an average **420.8 MiB higher inference-ready process RSS** for weight-only INT8 than for FP32. This is an observation about *whole-process resident memory*, not proof that the quantized weight tensors occupy more memory. The script loads the FP32 checkpoint and transforms it in-place, and temporary/retained allocations or quantization implementation overhead may affect the measurement. We did not isolate those causes in this experiment.

Consequently, **we have demonstrated storage reduction, but have not demonstrated runtime RAM reduction** with this implementation and measurement protocol. We also did not measure peak memory during a complete forward pass.

## 5. Relationship to the project's model-quality findings

The motivation for measuring weight-only INT8 is its previously observed behavior on MNLI and HANS. The following values are **from the earlier three-model accuracy experiments**, not from today's runtime script:

| Evaluation (three-seed mean) | FP32 | W8A8 | Weight-only INT8 |
|---|---:|---:|---:|
| MNLI matched | 84.64% | 84.74% | 84.66% |
| MNLI mismatched | 84.85% | 84.89% | 84.88% |
| HANS overall | 57.90% | 57.03% | 57.70% |
| HANS lexical-overlap, non-entailment | 35.76% | 31.55% | 34.83% |

Weight-only INT8 better preserved HANS accuracy than W8A8 under those evaluations. Today's CPU experiment adds the following *separate* finding: its saved model state was much smaller than FP32, while CPU forward throughput remained roughly unchanged. **The runtime benchmark did not include W8A8 on the same CPU**, so it should not be treated as a three-way speed or RAM comparison. Earlier GPU/MPS timings are not directly comparable.

## 6. Saved outputs and visualizations

All outputs are written beneath:

```text
results/runtime_analysis/fp32_vs_weight_only_cpu/
├── seed42_fp32_n3000.json
├── seed42_weight_only_int8_n3000.json
├── benchmark_results_n3000.csv
├── paired_comparison_n3000.csv
├── seed42_fp32_n30000.json
├── seed42_weight_only_int8_n30000.json
├── seed123_fp32_n30000.json
├── seed123_weight_only_int8_n30000.json
├── seed456_fp32_n30000.json
├── seed456_weight_only_int8_n30000.json
├── benchmark_results_n30000.csv
├── paired_comparison_n30000.csv
├── 01_serialized_size.png / .svg
├── 02_process_rss.png / .svg
├── 03_batch_latency.png / .svg
└── 04_throughput.png / .svg
```

The benchmark script **regenerates the four figure names** whenever the orchestrator finishes. Since the complete 30,000-example benchmark was run after the pilot, these image files represent the most recent **full benchmark**, not the earlier 3,000-example pilot. The `n3000` and `n30000` JSON/CSV files coexist.

| Figure | Interpretation |
|---|---|
| `01_serialized_size` | Mean saved `state_dict` size across three seeds (smaller is better) |
| `02_process_rss` | Mean whole-worker RSS measured when the model is ready; **not** peak/isolated model memory |
| `03_batch_latency` | Mean of per-seed median forward latency on one representative batch (lower is better) |
| `04_throughput` | Mean of per-seed median full-HANS throughput (higher is better) |

Error bars, when present, represent the **observed minimum-to-maximum across three seeds**, not confidence intervals. The underlying JSON files contain all recorded full-pass times and batch-latency samples.

## 7. Limitations and follow-up questions

- **Only one hardware/software setup:** Results depend on CPU, memory subsystem, TorchAO, PyTorch, threading, and kernel implementations.
- **Only three fine-tuning seeds:** These characterize the tested checkpoints; observed seed variability is not a broad performance confidence interval.
- **Sequential execution:** The benchmark runs FP32 then weight-only INT8 for each seed in separate processes. Machine load, clock speeds, thermal state, or time-varying conditions could affect small differences in throughput.
- **One sampled latency batch:** The batch-32 microbenchmark uses a middle pre-collated batch; it cannot stand in for the distribution of varying padded sequence lengths across HANS.
- **Model-forward scope:** Times exclude tokenization, data preparation, disk I/O, loading, quantization initialization, and output postprocessing.
- **RSS is not isolated model RAM:** It includes application overhead, cached memory, and allocator behavior. A cleaner follow-up would compare steady-state and peak RSS after direct loading of an already quantized checkpoint in a fresh process, recording memory throughout setup and inference.
- **Storage metric definition:** The file size is produced by `torch.save(model.state_dict())`, not necessarily the size of an optimized deployment artifact.
- **Not a W8A8 speed comparison:** No FP32/W8A8/weight-only three-way CPU performance result is established here.

The `torch.utils._pytree` `register_constant()` deprecation warnings appeared during worker startup but did **not** prevent successful completion. They do not constitute a measured performance finding.

## 8. Reproducibility

From the **repository root**, ensure your Python environment supports the project dependencies and that these files exist:

- `models/fp32/seed42/final`, `models/fp32/seed123/final`, `models/fp32/seed456/final`
- `results/hans/fp32_seed42_predictions.csv`, `results/hans/fp32_seed123_predictions.csv`, `results/hans/fp32_seed456_predictions.csv`
- `scripts/benchmark_fp32_weight_only_cpu.py`

Run the same two commands given in §3. If JSONs already exist, the benchmark script reuses them when the stored configuration and host platform match. **Do not pass `--overwrite` unless you intentionally want to rerun and replace measured files.**

To inspect final outputs:

```bash
cat results/runtime_analysis/fp32_vs_weight_only_cpu/paired_comparison_n30000.csv
ls results/runtime_analysis/fp32_vs_weight_only_cpu/*.{png,svg}
```

For reproducibility beyond these high-level tables, inspect the per-seed JSONs for the software versions, precise latency distributions, and full-pass timing repetitions.

## 9. Conclusion

> In a same-CPU, model-forward-only comparison across three MNLI-fine-tuned BERT checkpoints and 30,000 HANS examples per checkpoint, TorchAO weight-only INT8 reduced serialized model-state size by **58.47%** while preserving **approximately the same throughput** as FP32 (81.65 vs 81.98 examples/s on average). It was modestly faster on a sampled batch, but whole-dataset speed improvements were **not consistent** across seeds. Measured whole-process RSS was higher by approximately **421 MiB** under the quantized setup, so lower resident RAM consumption has **not** been demonstrated.

Taken together with the existing HANS results, weight-only INT8 represents a promising **accuracy–storage tradeoff**, but the current evidence does not support claiming a general CPU speed or runtime-memory advantage.

---

**Data provenance:** All benchmark figures and numbers above are transcribed or directly calculated from the November 20, 2026 terminal output shared for the 3,000- and 30,000-example runs. Raw benchmark JSON/CSV files remain on the user's local machine and were not provided for independent numerical audit in this document.
