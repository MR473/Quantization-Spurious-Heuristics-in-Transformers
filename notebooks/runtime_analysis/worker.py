#!/usr/bin/env python3
"""One isolated process per (seed, precision, execution mode, task).

Do NOT import or run this inside a notebook kernel. run_experiments.py launches
this program using subprocess.run so that model memory is isolated.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from contextlib import nullcontext

import numpy as np
import pandas as pd

from runtime_common import (
    VARIANTS, config_read, current_cuda_allocated_mib,
    current_cuda_reserved_mib, device_batch, input_examples, load_model,
    loader, output_dir, pack_environment, rss_mib, synchronize, tokenize,
    write_json,
)


def prepare(seed, variant):
    import torch
    from transformers import AutoTokenizer
    cfg = config_read()
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config.json requests CUDA, but torch.cuda.is_available() is False")
        torch.cuda.init()
    cuda_baseline = current_cuda_allocated_mib(device)
    rss_baseline = rss_mib()
    data, source = input_examples(cfg, seed)
    model, tokenizer, setup_s, state_mib = load_model(cfg, seed, variant, device)
    synchronize(device)
    model_alloc_mib = current_cuda_allocated_mib(device) - cuda_baseline
    mem_after_loading = rss_mib()
    env = pack_environment(cfg, seed, variant, source, device)
    encoded = tokenize(cfg, data, tokenizer)
    return {
        "cfg": cfg, "seed": seed, "variant": variant, "device": device,
        "data": data, "encoded": encoded, "model": model, "tokenizer": tokenizer,
        "source": source, "env": env, "quantization_seconds": setup_s,
        "serialized_state_mib": state_mib, "cuda_before_loading_mib": cuda_baseline,
        "cuda_model_resident_delta_mib": model_alloc_mib,
        "rss_before_loading_mib": rss_baseline,
        "rss_after_loading_mib": mem_after_loading,
    }


def forward(model, batch):
    with __import__("torch").inference_mode():
        return model(**batch).logits


def pad_incomplete_batch(batch, target_size):
    """Repeat examples only within an incomplete batch to reach target_size.

    This works around a compiled CUDA INT8 GEMM path that rejects matrix
    dimensions <= 16. We preserve every real input and its original order.
    The caller MUST count only original examples and, for predictions, discard
    the logits corresponding to repeated padding examples.

    Apply the identical policy to FP32 and INT8 for fair timing comparisons.
    """
    import torch

    n_real = int(batch["input_ids"].shape[0])
    target_size = int(target_size)
    if not 0 < n_real <= target_size:
        raise ValueError(f"Invalid batch size: {n_real}; target={target_size}")
    if n_real == target_size:
        return batch, n_real

    # Repeating existing examples preserves attention masks, token_type_ids,
    # input_ids and already-collated sequence length in all model-input tensors.
    repeat_indices = torch.arange(
        target_size - n_real, device=batch["input_ids"].device
    ) % n_real
    padded = {
        key: torch.cat([tensor, tensor.index_select(0, repeat_indices)], dim=0)
        for key, tensor in batch.items()
    }
    return padded, n_real


def warmup(ctx):
    cfg, device, model = ctx["cfg"], ctx["device"], ctx["model"]
    dl = loader(cfg, ctx["encoded"], ctx["tokenizer"], cfg["throughput_batch_size"])
    count = 0
    start_time = time.perf_counter()
    for batch in dl:
        _ = forward(model, device_batch(batch, device))
        count += 1
        if count >= int(cfg["warmup_batches"]):
            break
    synchronize(device)
    ctx["warmup_seconds"] = time.perf_counter() - start_time
    return count


def latency_samples(ctx, batch_size):
    """Forward-only latency for resident inputs: wall-clock AND CUDA events.

    Per-request wall time includes CPU dispatch + GPU execution/synchronization;
    CUDA event duration isolates device execution. Neither includes H2D transfer.
    Synchronizing each sample is valid for latency, never for throughput.
    """
    import torch
    cfg, device, model = ctx["cfg"], ctx["device"], ctx["model"]
    data_n = len(ctx["data"])
    want = min(int(cfg["latency_batches_per_size"]) * batch_size, data_n)
    # Spread samples across HANS; avoid benchmarking only the first heuristic.
    indices = np.linspace(0, data_n - 1, want, dtype=int)
    subset = {key: [values[int(i)] for i in indices] for key, values in ctx["encoded"].items()}
    dl = loader(cfg, subset, ctx["tokenizer"], batch_size)
    prepared = [device_batch(batch, device) for batch in dl]
    synchronize(device)
    latencies = []
    for batch in prepared:
        n = int(batch["input_ids"].shape[0])
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record()
            _ = forward(model, batch)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            gpu_event_ms = start.elapsed_time(end)
        else:
            wall_start = time.perf_counter()
            _ = forward(model, batch)
            synchronize(device)
            wall_ms = (time.perf_counter() - wall_start) * 1000
            gpu_event_ms = float("nan")
        latencies.append({
            "seed": ctx["seed"], "variant": ctx["variant"],
            "batch_size": batch_size, "batch_n": n,
            "forward_ms": wall_ms,
            "forward_gpu_event_ms": gpu_event_ms,
        })
    del prepared, dl, subset
    gc.collect()
    return latencies


def throughput_repeats(ctx):
    """Timed HANS pass: no per-batch GPU synchronization.

    Timed region includes CPU collation, host->device copies and model forward;
    excludes dataset read and tokenization. GPU sync once at each boundary.
    """
    import torch
    cfg, device, model = ctx["cfg"], ctx["device"], ctx["model"]
    bs = int(cfg["throughput_batch_size"])
    dl = loader(cfg, ctx["encoded"], ctx["tokenizer"], bs)
    runs = []
    for repeat in range(int(cfg["benchmark_repeats"])):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        before_rss = rss_mib()
        before_alloc = current_cuda_allocated_mib(device)
        before_reserved = current_cuda_reserved_mib(device)
        synchronize(device)
        start = time.perf_counter()
        count = 0
        extra_padding_examples = 0
        with torch.inference_mode():
            for batch in dl:
                batch_gpu = device_batch(batch, device)
                batch_gpu, n_real = pad_incomplete_batch(batch_gpu, bs)
                _ = model(**batch_gpu).logits
                count += n_real
                extra_padding_examples += int(batch_gpu["input_ids"].shape[0]) - n_real
        assert count == len(ctx["data"]), (count, len(ctx["data"]))
        synchronize(device)
        seconds = time.perf_counter() - start
        after_rss = rss_mib()
        del batch_gpu, _
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else float("nan")
        runs.append({
            "seed": ctx["seed"], "variant": ctx["variant"], "repeat": repeat + 1,
            "batch_size": bs, "examples": count, "seconds": seconds,
            "padded_duplicate_examples": extra_padding_examples,
            "physical_examples_processed": count + extra_padding_examples,
            "throughput_examples_per_second": count / seconds,
            "pre_inference_cuda_allocated_mib": before_alloc,
            "peak_inference_cuda_allocated_mib": peak_alloc,
            "peak_extra_cuda_allocated_mib": max(0, peak_alloc - before_alloc),
            "pre_inference_cuda_reserved_mib": before_reserved,
            "peak_inference_cuda_reserved_mib": peak_reserved,
            "process_rss_before_mib": before_rss,
            "process_rss_after_mib": after_rss,
        })
        print(f"  {ctx['variant']} repeat {repeat+1}: {count} examples, {seconds:.3f}s, {count/seconds:.1f}/s", flush=True)
    return pd.DataFrame(runs)


def benchmark(ctx):
    output = output_dir(ctx["cfg"], ctx["seed"], ctx["variant"])
    warmup(ctx)
    samples = []
    for bs in ctx["cfg"]["batch_sizes"]:
        samples.extend(latency_samples(ctx, int(bs)))
    # Release any GPU cached input batches from latency before peak tracking.
    gc.collect()
    if ctx["device"].type == "cuda":
        import torch
        torch.cuda.empty_cache()
    repeats = throughput_repeats(ctx)
    latency = pd.DataFrame(samples)
    repeats.to_csv(output / "throughput_repeats.csv", index=False)
    latency.to_csv(output / "latency_samples.csv", index=False)

    summary = dict(ctx["env"])
    summary.update({
        "status": "success", "num_hans_examples": len(ctx["data"]),
        "partial_batch_policy": "repeat examples within incomplete final batch to reach throughput_batch_size; count only real examples; apply to ALL variants",
        "padded_duplicate_examples_per_pass": int(repeats["padded_duplicate_examples"].iloc[0]),
        "serialized_state_mib": ctx["serialized_state_mib"],
        "runtime_quantization_setup_seconds": ctx["quantization_seconds"],
        "warmup_seconds_including_any_compilation": ctx.get("warmup_seconds", float("nan")),
        "cuda_model_resident_delta_mib": ctx["cuda_model_resident_delta_mib"],
        "rss_before_loading_mib": ctx["rss_before_loading_mib"],
        "rss_after_loading_mib": ctx["rss_after_loading_mib"],
        "median_full_hans_seconds": float(repeats["seconds"].median()),
        "median_throughput_examples_per_second": float(repeats["throughput_examples_per_second"].median()),
        "median_peak_cuda_allocated_mib": float(repeats["peak_inference_cuda_allocated_mib"].median()),
        "median_peak_extra_cuda_allocated_mib": float(repeats["peak_extra_cuda_allocated_mib"].median()),
        "median_peak_cuda_reserved_mib": float(repeats["peak_inference_cuda_reserved_mib"].median()),
    })
    for bs in ctx["cfg"]["batch_sizes"]:
        subset = latency.loc[latency["batch_size"] == int(bs), "forward_ms"]
        summary[f"batch{bs}_median_forward_latency_ms"] = float(subset.median())
        summary[f"batch{bs}_p95_forward_latency_ms"] = float(subset.quantile(0.95))
        summary[f"batch{bs}_latency_batches"] = len(subset)
        gpu_event = latency.loc[latency["batch_size"] == int(bs), "forward_gpu_event_ms"].dropna()
        summary[f"batch{bs}_median_gpu_event_ms"] = float(gpu_event.median()) if len(gpu_event) else float("nan")
    write_json(output / "benchmark_summary.json", summary)
    print(f"Saved: {output / 'benchmark_summary.json'}")


def run_predictions(ctx):
    import torch
    output = output_dir(ctx["cfg"], ctx["seed"], ctx["variant"])
    warmup(ctx)
    dl = loader(ctx["cfg"], ctx["encoded"], ctx["tokenizer"], ctx["cfg"]["throughput_batch_size"])
    probs = []
    synchronize(ctx["device"])
    with torch.inference_mode():
        for batch in dl:
            batch_gpu = device_batch(batch, ctx["device"])
            batch_gpu, n_real = pad_incomplete_batch(
                batch_gpu, int(ctx["cfg"]["throughput_batch_size"])
            )
            logits = ctx["model"](**batch_gpu).logits
            probs.append(torch.softmax(logits[:n_real].float(), dim=-1).cpu().numpy())
    logits_probs = np.concatenate(probs, axis=0)
    if logits_probs.shape != (len(ctx["data"]), 3):
        raise ValueError(f"Unexpected logits softmax shape: {logits_probs.shape}")
    labels = logits_probs.argmax(axis=1)
    pred = pd.DataFrame({
        "pairID": ctx["data"]["pairID"].astype(str),
        "mnli_prediction_id": labels.astype(int),
        "hans_prediction": np.where(labels == 0, "entailment", "non-entailment"),
        "prob_entailment": logits_probs[:, 0],
        "prob_neutral": logits_probs[:, 1],
        "prob_contradiction": logits_probs[:, 2],
    })
    pred.to_csv(output / "predictions.csv", index=False)
    write_json(output / "prediction_metadata.json", {
        **ctx["env"], "examples": len(pred),
        "partial_batch_policy": "repeat within incomplete batch; discard duplicated logits; retain all original examples",
    })
    print(f"Saved: {output / 'predictions.csv'}")


def profile(ctx):
    import torch
    output = output_dir(ctx["cfg"], ctx["seed"], ctx["variant"])
    warmup(ctx)
    dl = loader(ctx["cfg"], ctx["encoded"], ctx["tokenizer"], ctx["cfg"]["throughput_batch_size"])
    activities = [torch.profiler.ProfilerActivity.CPU]
    if ctx["device"].type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    n_batches = int(ctx["cfg"]["profiling_batches"])
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.inference_mode():
            for i, batch in enumerate(dl):
                if i >= n_batches:
                    break
                _ = ctx["model"](**device_batch(batch, ctx["device"])).logits
                prof.step()
    synchronize(ctx["device"])
    rows = []
    for event in prof.key_averages():
        rows.append({
            "operator": event.key,
            "count": int(event.count),
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "self_device_time_us": float(getattr(event, "self_device_time_total", 0.0)),
            "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
            "device_memory_bytes": int(getattr(event, "device_memory_usage", 0)),
        })
    data = pd.DataFrame(rows).sort_values("self_device_time_us", ascending=False)
    data.to_csv(output / "profiler_operators.csv", index=False)
    prof.export_chrome_trace(str(output / "profiler_chrome_trace.json"))
    write_json(output / "profile_metadata.json", {**ctx["env"], "profile_batches": n_batches})
    print("Profiler summary (device self-time; operator breakdown):")
    print(data.head(15).to_string(index=False))
    print(f"Saved: {output / 'profiler_operators.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--task", choices=("benchmark", "profile", "predictions"), required=True)
    args = parser.parse_args()
    cfg = config_read()
    target = output_dir(cfg, args.seed, args.variant)
    try:
        ctx = prepare(args.seed, args.variant)
        if args.task == "benchmark":
            benchmark(ctx)
        elif args.task == "profile":
            profile(ctx)
        else:
            run_predictions(ctx)
        write_json(target / f"{args.task}_status.json", {"status": "success", "seed": args.seed, "variant": args.variant})
    except Exception as exc:
        write_json(target / f"{args.task}_status.json", {
            "status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
            "seed": args.seed, "variant": args.variant,
        })
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
