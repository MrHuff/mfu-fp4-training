#!/usr/bin/env python3
"""Benchmark fused versus materialized Nemotron RMSNorm NVFP4 projection."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _configure(backend: str, fp4_root: Path) -> None:
    from run_nvblog_llama3_8b_matrix import (
        _localcta_v4_highwater_env,
        _tk_v5_swiglu_env,
    )

    env = (
        _localcta_v4_highwater_env()
        if backend == "localcta"
        else _tk_v5_swiglu_env()
    )
    env.update(
        {
            "FP4_MATMUL_ROOT": str(fp4_root),
            "FP4_MATMUL_GEMM_ROOT": str(fp4_root),
            "FP4_MXFP4_ROOT": str(fp4_root),
        }
    )
    os.environ.update(env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("v5", "localcta"), required=True)
    parser.add_argument("--fp4-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _configure(args.backend, args.fp4_root.resolve())

    import torch
    import torch.nn as nn

    from low_bits_training.quantization.fused_te_linear import (
        NVFP4RMSNormLinearTK,
        SimpleFP4Linear,
    )

    class MaterializedRMSLinear(nn.Module):
        def __init__(self, rows: int):
            super().__init__()
            del rows
            self.norm_weight = nn.Parameter(
                torch.ones(4096, device="cuda", dtype=torch.float32)
            )
            self.linear = SimpleFP4Linear(
                4096,
                18560,
                bias=False,
                device="cuda",
                dtype=torch.bfloat16,
            )

        def forward(self, value):
            value_f = value.float()
            inv_rms = torch.rsqrt(
                value_f.square().mean(dim=-1, keepdim=True) + 1e-5
            )
            normed = (
                value_f * inv_rms * self.norm_weight.float()
            ).to(torch.bfloat16)
            return self.linear(normed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    x = torch.randn(
        args.rows,
        4096,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    dy = (
        torch.randn(
            args.rows,
            18560,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)
    fused = NVFP4RMSNormLinearTK(
        4096,
        18560,
        eps=1e-5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    materialized = MaterializedRMSLinear(args.rows)
    with torch.no_grad():
        materialized.linear.weight.copy_(fused.weight)
        materialized.norm_weight.copy_(fused.norm_weight.float())

    def measure(module: nn.Module) -> list[float]:
        durations = []
        for index in range(args.warmup + args.iterations):
            x.grad = None
            for parameter in module.parameters():
                parameter.grad = None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = module(x)
            torch.autograd.backward(output, dy)
            end.record()
            end.synchronize()
            if index >= args.warmup:
                durations.append(float(start.elapsed_time(end)))
        return durations

    materialized_a1 = measure(materialized)
    fused_b = measure(fused)
    materialized_a2 = measure(materialized)
    a1_median = statistics.median(materialized_a1)
    b_median = statistics.median(fused_b)
    a2_median = statistics.median(materialized_a2)
    control_midpoint = 0.5 * (a1_median + a2_median)
    result = {
        "backend": args.backend,
        "shape": [args.rows, 4096, 18560],
        "materialized_a1_ms": a1_median,
        "fused_ms": b_median,
        "materialized_a2_ms": a2_median,
        "materialized_midpoint_ms": control_midpoint,
        "speedup": control_midpoint / b_median,
        "reduction_percent": 100.0 * (control_midpoint - b_median)
        / control_midpoint,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
