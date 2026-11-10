# Runtime analysis — eager FP32 versus dynamic INT8 (TorchAO)

## Scope

- **Main behavioral study:** MNLI-fine-tuned BERT, FP32 vs dynamic W8A8 INT8,
  CPU eager inference, three seeds (42, 123, 456), evaluated on HANS and MNLI.
- **Runtime study:** FP32 eager vs dynamic INT8 eager on CUDA, with isolated
  fresh Python processes, batch size 32 and repeated throughput measurements.
  **Currently benchmarked for seed 42 only.** CPU and CUDA predictions are not
  assumed to be numerically identical.
- The compiler optimization experiments were moved to an external local archive
  and are not included in this repository's final performance comparison.

## Files

- `00_preflight.ipynb`: verify checkpoints, Python packages and CUDA device.
- `01_isolated_eager.ipynb`: isolated memory, latency, throughput measurements.
- `02_cuda_profiler.ipynb`: explain performance through operator timing.
- `04_prediction_validation.ipynb`: compare saved CPU reference and eager CUDA predictions.
- `05_review_results.ipynb`: export runtime figures and aggregate summaries.
- `config.json`: defines the execution and dataset configuration.
- `runtime_common.py`, `worker.py`, `run_experiments.py`,
  `compare_predictions.py`, `summarize_results.py`: supporting scripts.

## Run from the repository root

```bash
python notebooks/runtime_analysis/run_experiments.py --seeds 42 --task benchmark --variants fp32_eager int8_eager
python notebooks/runtime_analysis/run_experiments.py --seeds 42 --task predictions --variants fp32_eager int8_eager
python notebooks/runtime_analysis/compare_predictions.py --seeds 42
python notebooks/runtime_analysis/summarize_results.py
```

Already-saved measurements can be re-summarized without rebenchmarking:

```bash
python notebooks/runtime_analysis/compare_predictions.py --seeds 42
python notebooks/runtime_analysis/summarize_results.py
```

## Primary metrics and interpretation

- `serialized_state_mib`: serialized model-state storage, **not** full package size.
- `cuda_model_resident_delta_mib`: additional allocated model tensor memory.
- `median_peak_cuda_allocated_mib`: peak allocated tensors during inference.
- `median_throughput_examples_per_second`: full HANS forward throughput;
  includes collation and host-to-device transfers but not tokenization.
- Batch-32 forward latency: forward-pass timing, separately from throughput.

Main experiment outputs are in `results/runtime_analysis/seed42/fp32_eager/`
 and `results/runtime_analysis/seed42/int8_eager/`.
The aggregate result tables are `all_benchmark_variants.csv` and
`paired_fp32_int8_comparison.csv` in `results/runtime_analysis/`.

Do not remove any trained checkpoint or canonical HANS prediction CSV as part
of runtime-folder cleanup. Large `profiler_chrome_trace.json` files should be
kept locally (if needed) and ignored by Git.
