# Runtime Analysis — Independent FP32 vs Dynamic INT8 Benchmark

This entire directory is **new and independent**. It does not edit, rename, or overwrite your existing project notebooks or checkpoints.

Place the full folder at:

```text
Quantization-Spurious-Heuristics-in-Transformers/
├── models/
│   └── fp32/
│       ├── seed42/final/
│       ├── seed123/final/
│       └── seed456/final/
├── results/
│   └── hans/
│       ├── fp32_seed42_predictions.csv
│       └── int8_seed42_predictions.csv
└── notebooks/
    └── runtime_analysis/              <-- COPY THIS WHOLE FOLDER HERE
        ├── README.md
        ├── config.json                 <-- ONLY REQUIRED CONFIG EDIT
        ├── 00_preflight.ipynb
        ├── 01_isolated_eager.ipynb
        ├── 02_cuda_profiler.ipynb
        ├── 03_compiled_experiment.ipynb
        ├── 04_prediction_validation.ipynb
        ├── 05_review_results.ipynb
        ├── runtime_common.py
        ├── worker.py
        ├── run_experiments.py
        ├── summarize_results.py
        ├── compare_predictions.py
        ├── requirements_additional.txt
        └── reference/
            ├── quantization_original.ipynb
            ├── quantization_efficiency_analysis_revised_original.ipynb
            └── previous_results/        <-- copies of old seed CSV baselines
```

## One-time setup

1. **Copy the whole `runtime_analysis/` folder into `notebooks/`.** Do not move files out individually: worker scripts are required.
2. Open `00_preflight.ipynb` using **the same Python kernel** as your working `quantization.ipynb` (PyTorch 2.14.0+cu130, TorchAO 0.18.0 as previously used).
3. **Only edit `config.json` if the directory names differ from those shown above**. Paths are resolved from the source file location, **not the Jupyter working directory**.
4. Prefer existing HANS predictions at `results/hans/fp32_seed{seed}_predictions.csv`. If missing, the code fetches the official raw TSV from the same URL as your `quantization.ipynb`. **It does not use `load_dataset("hans")`**.
5. No training and no saved INT8 checkpoint are needed: a fresh worker loads the original FP32 checkpoint, applies `quantize_(..., Int8DynamicActivationInt8WeightConfig())` when appropriate, then runs measurements.

**Do not `pip install` or upgrade PyTorch or TorchAO without need.** GPU versions are environment-sensitive. If imports are missing, see `requirements_additional.txt`.

## Execute in this order

**Step 1 — Preflight**: Run `00_preflight.ipynb` to validate paths, GPU, and versions. This does not benchmark.

**Step 2 — Eager baseline**: Run `01_isolated_eager.ipynb` for **seed 42 only**. It runs `fp32_eager` and `int8_eager` in independent subprocesses. Each does three complete HANS passes by default; each pass uses 30,000 examples / batch 32. Single-request latency uses a separate batch-size-1 measurement.

**Step 3 — Diagnose**: Run `02_cuda_profiler.ipynb` on seed 42. It profiles 8 batches per variant to find expensive operators/kernels, preserving Chrome traces. Inspect `profiler_operators.csv`. Profiling is deliberately *not* part of benchmark timing.

**Step 4 — Optional optimization experiment**: Run `03_compiled_experiment.ipynb` for seed 42. It tests `fp32_compiled` and `int8_compiled` **with the same torch.compile settings**. The TorchAO GPU tutorial recommends compilation, but success and acceleration are hardware/version/model dependent. First compilation may consume significant resources. **Compilation failures are recorded, never silently replaced with eager**. Both FP32 and INT8 must work in the same execution mode to measure a precision-specific speed effect.

**Step 5 — Verify model behavior**: Run `04_prediction_validation.ipynb`. For the full 30,000 examples, save new model probabilities and pair with existing `results/hans/*_predictions.csv` by `pairID`. If compiled modes worked, enable them too. It checks whether CPU-vs-CUDA or compiled-vs-eager execution changed model predictions or probabilities. Existing HANS predictions may have been computed on CPU; differences should be quantified, not presumed absent.

**Step 6 — Visualize and export**: Run `05_review_results.ipynb`. It reads existing CSV and JSON files, creates aggregate comparisons, and saves a table/plots to `results/runtime_analysis/`.

Then repeat Steps 2–5 with seeds 123 and 456 **only after seed 42 validates**. This avoids repeating a problematic compilation/profiling path three times.

## Only configuration to edit

`config.json`:

```json
{
  "seeds": [42, 123, 456],
  "model_dir_template": "models/fp32/seed{seed}/final",
  "results_dir": "results/runtime_analysis",
  "hans_predictions_template": "results/hans/fp32_seed{seed}_predictions.csv",
  "reference_int8_predictions_template": "results/hans/int8_seed{seed}_predictions.csv",
  "device": "cuda",
  "max_length": 128,
  "batch_sizes": [1, 32],
  "throughput_batch_size": 32,
  "benchmark_repeats": 3,
  "max_examples": null
}
```

The shipped `config.json` includes additional optional settings. **Keep `max_examples: null` for report-quality results.** To debug quickly, change it to `1000`; then restore `null` and regenerate predictions/benchmarks before final comparisons.

If your checkpoint directories differ, modify `model_dir_template`; the standalone analysis will not move or edit your checkpoints.

## What is measured / how to interpret

| Metric | How measured | Publication status |
|---|---|---|
| Serialized state size | `torch.save(model.state_dict())` into BytesIO | **Primary** (state footprint, not full application package) |
| Resident model CUDA allocated delta | `memory_allocated` after model transfer minus before model transfer in isolated worker | **Primary memory estimate** (includes PyTorch tensor storage; not total GPU device use) |
| Peak CUDA allocated tensor memory | `max_memory_allocated` reset after warm-up | **Primary inference memory**; may include temporary buffers/inputs |
| Peak CUDA reserved memory | `max_memory_reserved` | **Supporting** allocator diagnostic, not model tensor memory |
| Batch 1 median and p95 latency | Wall-clock synchronized model-forward timing on pretransferred inputs (plus separate CUDA event diagnostics) | **Primary responsiveness** |
| Batch 32 examples/s | `num_examples / elapsed wall time`, synchronizing only once before and after whole inference pass | **Primary throughput** (collation + device transfer included; tokenization excluded) |
| Full HANS execution time | same full pass as throughput | **Supporting** (largely redundant with throughput) |
| Quantization setup time | conversion time of dynamic W8A8 | **Supporting** startup cost |
| Whole-process RSS | collected at startup/load and during run | **Diagnostic only**, not primary GPU inference metric |

**The latency and throughput timing scopes intentionally differ:** latency isolates the GPU forward pass, whereas throughput includes DataLoader collation and transfers. State that difference explicitly in the report. Both models use exactly the same timing methodology within each metric.

**Important:** To compare INT8 impact, use `fp32_eager` vs `int8_eager`, or `fp32_compiled` vs `int8_compiled`. Do **not** compare `fp32_eager` to `int8_compiled` as a quantization-only benefit, since both the execution strategy and precision changed.

Compiled inference uses `dynamic=true` and `mode="max-autotune"` by default to explore GPU TorchAO's optimized execution path; it can fail or be expensive for BERT on this particular hardware. The correct result in that case is a documented *unsupported/failed optimization trial*, not an invented compiled speedup.

## Output directory (automatically created)

```text
results/runtime_analysis/
├── seed42/
│   ├── fp32_eager/
│   │   ├── benchmark_summary.json
│   │   ├── throughput_repeats.csv
│   │   ├── latency_samples.csv
│   │   ├── profiler_operators.csv               [when profiled]
│   │   ├── profiler_chrome_trace.json            [when profiled]
│   │   ├── predictions.csv                       [when generated]
│   │   ├── benchmark_status.json
│   │   └── ...
│   ├── int8_eager/
│   ├── fp32_compiled/                            [optional]
│   └── int8_compiled/                            [optional]
├── seed123/
├── seed456/
├── all_benchmark_variants.csv
├── all_throughput_repeats.csv
├── all_latency_samples.csv
├── paired_fp32_int8_comparison.csv
└── prediction_equivalence_checks.csv
```

Prior outputs are copied into `notebooks/runtime_analysis/reference/previous_results/`. The new experiment does **not** overwrite your old `results/efficiency/` data.

## Terminal commands (if you prefer terminal)

Run these from anywhere, with the same activated Python environment as your working quantization notebook:

```bash
python notebooks/runtime_analysis/run_experiments.py --seeds 42 --task benchmark --variants fp32_eager int8_eager
python notebooks/runtime_analysis/run_experiments.py --seeds 42 --task profile --variants fp32_eager int8_eager
python notebooks/runtime_analysis/run_experiments.py --seeds 42 --task benchmark --variants fp32_compiled int8_compiled
python notebooks/runtime_analysis/run_experiments.py --seeds 42 --task predictions --variants fp32_eager int8_eager
python notebooks/runtime_analysis/compare_predictions.py --seeds 42
python notebooks/runtime_analysis/summarize_results.py
```

## References (implementation choices)

- TorchAO dynamic INT8 W8A8: https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int8DynamicActivationInt8WeightConfig.html
- TorchAO CUDA compilation discussion: https://docs.pytorch.org/tutorials/unstable/gpu_quantization_torchao_tutorial.html
- PyTorch profiler: https://docs.pytorch.org/docs/stable/profiler.html

## Validation limitation

This package was structurally validated in the creation environment; the creator did **not** have access to your local FP32 checkpoints or CUDA/TorchAO installation for an end-to-end GPU run. Use the preflight, then seed 42 eager, and inspect any detailed `*_status.json` errors before proceeding.
