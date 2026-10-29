# Runtime analysis review — seed 42

Source: supplied runtime_analysis.zip and the separately uploaded fp32_seed456_predictions.csv (used only for fixed HANS gold labels/heuristic metadata, matched by pairID). This gold-label mapping was sanity checked because seed-42 FP32 runtime accuracy equals the known 59.70%. Original seed-42 INT8 binary predictions are reconstructed from the difference CSV relative to INT8 eager; original seed-42 predictions are not in the supplied ZIP.

## Core results

* FP32 eager: 2990.8 examples/s, 10.031 seconds.
* Original full INT8 eager: 562.1 examples/s, 53.368 seconds.
* FP32 compiled: 3202.9 examples/s, 9.366 seconds.
* INT8 compiled with FP32 classifier: 13482.3 examples/s, 2.225 seconds.
* Compiled INT8 variant throughput / compiled FP32: 4.209x.
* Compiled INT8 variant throughput / original INT8 eager: 23.984x (NOT a controlled compilation-only comparison: classifier precision differs).
* Serialized INT8 state savings: 58.470% (eager).
* Peak allocated CUDA savings: 51.396% (compiled).

## Behavioral interpretation

* Original FP32 reference vs FP32 eager: 0 HANS flips; FP32 eager vs compiled: 0 flips.
* Original INT8 reference vs INT8 eager GPU: 375/30,000 binary flips; probability MAE 0.008235.
* Original INT8 reference vs INT8 compiled FP32 head: 369/30,000 binary flips; probability MAE 0.008190.
* INT8 eager GPU vs INT8 compiled FP32 head: 356/30,000 binary flips; probability MAE 0.008106.
* HANS overall accuracy: FP32 59.700%, original INT8 reconstructed 58.927%, INT8 eager GPU 58.897%, compiled INT8 FP32 head 58.890%.
* Lexical-overlap non-entailment accuracy: FP32 42.80%, original INT8 38.90%, INT8 eager GPU 39.08%, compiled INT8 FP32 head 39.20%.

## Critical methodological cautions

1. The compiled INT8 FP32-head model is NOT bitwise/output equivalent to the original quantized model. Report it as a distinct model configuration; aggregate HANS trends nevertheless match closely at seed 42.
2. Original quantization.ipynb leaves the quantized model and inputs on CPU. The runtime worker benchmarks INT8 on CUDA. CPU/GPU differences are a plausible contributor to original-vs-runtime prediction disagreement, but are not causally isolated by these results.
3. The compiled INT8 and eager INT8 variants differ in BOTH head precision and compilation. Add an int8_eager_fp32_head variant if you want to isolate compilation from head precision.
4. Only one fine-tuning seed appears in this archive. Repeat on other seeds before making general claims.
5. All variants repeat the final 16 HANS examples to physically run 30,016 inputs; reported throughput divides 30,000 by elapsed time. Same protocol for every variant.
6. Batch size 1 is absent. These results assess batch-32 bulk inference, not single-request latency.
7. Compiled first timed passes contain additional shape-specialization overhead. The median of passes 2 and 3 represents steady state better; record cold-start separately. Exact model loading and compile wall time require a separate measurement.
8. The archive contains only results, not final config.json or current scripts. Run metadata records compile_mode=max-autotune-no-cudagraphs and compile_dynamic=true; this can be reported but exact source-code reproducibility cannot be independently confirmed from ZIP alone.
9. Profiler shows significant INT8 eager host/device overhead: 21,320 `Memcpy DtoH` events plus 21,321 `aten::_local_scalar_dense` calls in eight batches, compared with eight `Memcpy DtoH` events for FP32 eager (see profiler CSV). This suggests overhead, but not proof of complete causal explanation of 5.3x slowdown.
