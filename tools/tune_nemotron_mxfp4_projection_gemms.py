#!/usr/bin/env python3
"""Sweep exact Nemotron MXFP4 input-projection GEMM configurations."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _configure(fp4_root: Path) -> None:
    from run_nvpaper_nemotron_h_8b_matrix import _mxfp4_all_linear_ssd_env

    env = _mxfp4_all_linear_ssd_env()
    env.update(
        {
            "FP4_MATMUL_ROOT": str(fp4_root),
            "FP4_MATMUL_GEMM_ROOT": str(fp4_root),
            "FP4_MXFP4_ROOT": str(fp4_root),
        }
    )
    os.environ.update(env)


def _measure(call, *, warmup: int, iterations: int) -> list[float]:
    import torch

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _sweep_orientation(
    *,
    name: str,
    a_fp4,
    a_sc,
    b_fp4,
    b_sc,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    import torch

    from low_bits_training.quantization.mxfp4_backend import (
        mxfp4_gemm,
        mxfp4_gemm_config,
    )

    output = torch.empty(
        a_fp4.shape[0],
        b_fp4.shape[0],
        device=a_fp4.device,
        dtype=torch.bfloat16,
    )
    reference = torch.empty_like(output)
    mxfp4_gemm(a_fp4, a_sc, b_fp4, b_sc, reference)
    torch.cuda.synchronize()

    candidates = []
    for config_id in range(11):
        mxfp4_gemm_config(
            a_fp4,
            a_sc,
            b_fp4,
            b_sc,
            output,
            config_id=config_id,
        )
        torch.cuda.synchronize()
        exact = bool(torch.equal(output, reference))
        samples = _measure(
            lambda config_id=config_id: mxfp4_gemm_config(
                a_fp4,
                a_sc,
                b_fp4,
                b_sc,
                output,
                config_id=config_id,
            ),
            warmup=warmup,
            iterations=iterations,
        )
        candidates.append(
            {
                "config_id": config_id,
                "exact": exact,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "samples_ms": samples,
            }
        )

    baseline_samples = _measure(
        lambda: mxfp4_gemm(
            a_fp4,
            a_sc,
            b_fp4,
            b_sc,
            output,
        ),
        warmup=warmup,
        iterations=iterations,
    )
    baseline_median = statistics.median(baseline_samples)
    valid = [candidate for candidate in candidates if candidate["exact"]]
    winner = min(valid, key=lambda candidate: candidate["median_ms"])
    return {
        "name": name,
        "shape": [
            int(a_fp4.shape[0]),
            int(b_fp4.shape[0]),
            int(a_fp4.shape[1] * 2),
        ],
        "baseline_median_ms": baseline_median,
        "baseline_samples_ms": baseline_samples,
        "winner": winner,
        "winner_speedup": baseline_median / winner["median_ms"],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp4-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _configure(args.fp4_root.resolve())

    import torch

    from low_bits_training.quantization.mxfp4_fused_linear import (
        _quantize_row_col_bf16,
        _quantize_row_col_bf16_padded,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    x = torch.randn(
        args.rows, 4096, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        18560, 4096, device="cuda", dtype=torch.bfloat16
    )
    dy = torch.randn(
        args.rows, 18560, device="cuda", dtype=torch.bfloat16
    )
    x_q = _quantize_row_col_bf16(x, role="activation")
    weight_q = _quantize_row_col_bf16_padded(weight, 18688, 4096)
    dy_q = _quantize_row_col_bf16_padded(dy, args.rows, 18688)

    results = [
        _sweep_orientation(
            name="forward",
            a_fp4=x_q.row_fp4,
            a_sc=x_q.row_sc,
            b_fp4=weight_q.row_fp4,
            b_sc=weight_q.row_sc,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        _sweep_orientation(
            name="dgrad",
            a_fp4=dy_q.row_fp4,
            a_sc=dy_q.row_sc,
            b_fp4=weight_q.col_fp4,
            b_sc=weight_q.col_sc,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
        _sweep_orientation(
            name="wgrad",
            a_fp4=dy_q.col_fp4,
            a_sc=dy_q.col_sc,
            b_fp4=x_q.col_fp4,
            b_sc=x_q.col_sc,
            warmup=args.warmup,
            iterations=args.iterations,
        ),
    ]
    rendered = json.dumps(results, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
