#!/usr/bin/env python3
"""Safely launch independent Python workers from terminal OR a notebook cell.

Examples:
 python notebooks/runtime_analysis/run_experiments.py --seeds 42 --variants fp32_eager int8_eager --task benchmark
 python notebooks/runtime_analysis/run_experiments.py --seeds 42 --variants fp32_compiled int8_compiled --task benchmark
 python notebooks/runtime_analysis/run_experiments.py --seeds 42 --variants fp32_eager int8_eager --task profile
 python notebooks/runtime_analysis/run_experiments.py --seeds 42 --variants fp32_eager int8_eager --task predictions
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from runtime_common import ANALYSIS_DIR, VARIANTS, config_read, project_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--variants", nargs="+", choices=VARIANTS,
                   default=["fp32_eager", "int8_eager"])
    p.add_argument("--task", choices=["benchmark", "profile", "predictions"], default="benchmark")
    p.add_argument("--continue-on-error", action="store_true", help="Collect failures rather than stopping")
    a = p.parse_args()
    cfg = config_read()
    bad = []
    print("CUDA variants run SEQUENTIALLY in a FRESH PROCESS each.", flush=True)
    print("Working directory:", project_path("."), flush=True)
    for seed in a.seeds:
        for variant in a.variants:
            cmd = [sys.executable, str(ANALYSIS_DIR / "worker.py"), "--seed", str(seed),
                   "--variant", variant, "--task", a.task]
            print(f"\nRunning seed={seed}, variant={variant}, task={a.task}\n", flush=True)
            start = time.perf_counter()
            res = subprocess.run(cmd, cwd=project_path("."), check=False)
            print(f"Finished exit code {res.returncode} in {time.perf_counter() - start:.1f}s", flush=True)
            if res.returncode:
                bad.append((seed, variant, a.task))
                if not a.continue_on_error:
                    raise SystemExit(f"Worker failed. Inspect results/runtime_analysis/seed{seed}/{variant}/{a.task}_status.json")
    if bad:
        raise SystemExit(f"Failed jobs: {bad}")
    print("\nAll requested subprocess jobs completed successfully.")


if __name__ == "__main__":
    main()
