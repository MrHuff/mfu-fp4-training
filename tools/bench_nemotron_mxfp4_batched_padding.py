#!/usr/bin/env python3
"""Validate and benchmark batched native MXFP4 Nemotron padding."""

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


def _cosine(lhs, rhs) -> float:
    import torch

    return float(
        torch.nn.functional.cosine_similarity(
            lhs.float().reshape(1, -1),
            rhs.float().reshape(1, -1),
        ).item()
    )


def _relative_l2(lhs, rhs) -> float:
    return float(
        ((lhs.float() - rhs.float()).norm() / rhs.float().norm().clamp_min(1e-12)).item()
    )


def _run_once(layer, x, dy, *, native: bool):
    import torch

    os.environ["MXFP4_USE_BATCHED_NEMOTRON_PADDING"] = "1" if native else "0"
    layer.zero_grad(set_to_none=True)
    x.grad = None
    output = layer(x)
    torch.autograd.backward(output, dy)
    torch.cuda.synchronize()
    return {
        "output": output.detach().clone(),
        "output_stride": list(output.stride()),
        "dx": x.grad.detach().clone(),
        "dw": layer.weight.grad.detach().clone(),
        "dgamma": layer.norm_weight.grad.detach().clone(),
    }


def _validate(layer, *, rows: int, seed: int) -> dict[str, object]:
    import torch

    torch.manual_seed(seed)
    x_value = torch.randn(
        rows, 4096, device="cuda", dtype=torch.bfloat16
    )
    dy = (
        torch.randn(
            rows, 18560, device="cuda", dtype=torch.float32
        )
        * 0.1
    ).to(torch.bfloat16)
    x_control = x_value.detach().clone().requires_grad_(True)
    control = _run_once(layer, x_control, dy, native=False)
    x_native = x_value.detach().clone().requires_grad_(True)
    native = _run_once(layer, x_native, dy, native=True)

    with torch.no_grad():
        x_f = x_value.float()
        gamma_f = layer.norm_weight.detach().float()
        weight_f = layer.weight.detach().float()
        dy_f = dy.float()
        inv_rms = torch.rsqrt(
            x_f.square().mean(dim=-1, keepdim=True) + layer.eps
        )
        normed_f = x_f * inv_rms * gamma_f
        output_ref = normed_f @ weight_f.t()
        d_normed_f = dy_f @ weight_f
        projected_f = d_normed_f * gamma_f
        dot_f = (projected_f * x_f).mean(dim=-1, keepdim=True)
        dx_ref = (
            inv_rms * projected_f
            - inv_rms.pow(3) * x_f * dot_f
        )
        dw_ref = dy_f.t() @ normed_f
        dgamma_ref = (d_normed_f * x_f * inv_rms).sum(dim=0)

    result = {
        "shape": [rows, 4096, 18560],
        "control_output_stride": control["output_stride"],
        "native_output_stride": native["output_stride"],
        "native_vs_control": {
            name: {
                "cosine": _cosine(native[name], control[name]),
                "relative_l2": _relative_l2(native[name], control[name]),
            }
            for name in ("output", "dx", "dw", "dgamma")
        },
        "native_vs_fp32": {
            "output_cosine": _cosine(native["output"], output_ref),
            "output_tail_cosine": _cosine(
                native["output"][:, -128:], output_ref[:, -128:]
            ),
            "dx_cosine": _cosine(native["dx"], dx_ref),
            "dw_cosine": _cosine(native["dw"], dw_ref),
            "dw_tail_cosine": _cosine(
                native["dw"][-128:], dw_ref[-128:]
            ),
            "dgamma_cosine": _cosine(native["dgamma"], dgamma_ref),
        },
        "finite": all(
            bool(torch.isfinite(native[name]).all().item())
            for name in ("output", "dx", "dw", "dgamma")
        ),
        "output_tail_nonzero_fraction": float(
            native["output"][:, -128:].ne(0).float().mean().item()
        ),
        "dw_tail_nonzero_fraction": float(
            native["dw"][-128:].ne(0).float().mean().item()
        ),
    }
    result["passed"] = bool(
        result["native_output_stride"] == [18688, 1]
        and result["finite"]
        and all(
            item["cosine"] >= 0.999
            for item in result["native_vs_control"].values()
        )
        and all(
            value >= 0.98
            for value in result["native_vs_fp32"].values()
        )
        and result["output_tail_nonzero_fraction"] >= 0.99
        and result["dw_tail_nonzero_fraction"] >= 0.99
    )
    return result


def _benchmark(layer, *, rows: int, warmup: int, iterations: int, seed: int):
    import torch

    torch.manual_seed(seed + 1)
    x = torch.randn(
        rows,
        4096,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    dy = (
        torch.randn(
            rows,
            18560,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.1
    ).to(torch.bfloat16)

    def measure(native: bool) -> list[float]:
        os.environ["MXFP4_USE_BATCHED_NEMOTRON_PADDING"] = (
            "1" if native else "0"
        )
        durations = []
        for index in range(warmup + iterations):
            layer.zero_grad(set_to_none=True)
            x.grad = None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = layer(x)
            torch.autograd.backward(output, dy)
            end.record()
            end.synchronize()
            if index >= warmup:
                durations.append(float(start.elapsed_time(end)))
        return durations

    control_a1 = measure(False)
    native_b = measure(True)
    control_a2 = measure(False)
    a1_median = statistics.median(control_a1)
    b_median = statistics.median(native_b)
    a2_median = statistics.median(control_a2)
    control_midpoint = 0.5 * (a1_median + a2_median)
    return {
        "shape": [rows, 4096, 18560],
        "control_a1_ms": a1_median,
        "native_ms": b_median,
        "control_a2_ms": a2_median,
        "control_midpoint_ms": control_midpoint,
        "speedup": control_midpoint / b_median,
        "reduction_percent": (
            100.0 * (control_midpoint - b_median) / control_midpoint
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp4-root", type=Path, required=True)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--benchmark-rows", type=int, default=24576)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _configure(args.fp4_root.resolve())

    import torch

    from low_bits_training.quantization.mxfp4_fused_linear import (
        MXFP4RMSNormLinearTK,
    )

    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    layer = MXFP4RMSNormLinearTK(
        4096,
        18560,
        eps=1e-5,
        device="cuda",
        dtype=torch.bfloat16,
    )
    result = {
        "validation": _validate(
            layer,
            rows=args.validation_rows,
            seed=args.seed,
        ),
        "benchmark": _benchmark(
            layer,
            rows=args.benchmark_rows,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not result["validation"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
