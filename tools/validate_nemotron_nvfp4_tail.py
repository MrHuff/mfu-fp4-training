#!/usr/bin/env python3
"""Validate the native NVFP4 256-tile tail used by Nemotron-H Mamba in_proj."""

from __future__ import annotations

import argparse
import json
import os
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


def _stats(actual, expected) -> dict[str, float | bool]:
    import torch

    actual_f = actual.float()
    expected_f = expected.float()
    finite = bool(torch.isfinite(actual_f).all().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_f.flatten(), expected_f.flatten(), dim=0
        ).item()
    )
    rel_l2 = float(
        (actual_f - expected_f).norm().div(expected_f.norm().clamp_min(1e-12)).item()
    )
    return {"finite": finite, "cosine": cosine, "rel_l2": rel_l2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("v5", "localcta"), required=True)
    parser.add_argument("--fp4-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--fused-rms", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    _configure(args.backend, args.fp4_root.resolve())

    import torch

    from low_bits_training.quantization.fused_te_linear import (
        NVFP4RMSNormLinearTK,
        SimpleFP4Linear,
    )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows, in_features, out_features = 256, 4096, 18560
    padded_out_features = 18688
    x = (torch.randn(rows, in_features, device=device) * 0.5).to(torch.bfloat16)
    weight = (
        torch.randn(out_features, in_features, device=device) * 0.02
    ).to(torch.bfloat16)
    dy = (
        torch.randn(rows, out_features, device=device) * 0.1
    ).to(torch.bfloat16)
    epsilon = 1e-5
    norm_weight = (
        1.0 + torch.randn(in_features, device=device) * 0.05
    ).to(torch.bfloat16)

    if args.fused_rms:
        layer = NVFP4RMSNormLinearTK(
            in_features,
            out_features,
            eps=epsilon,
            device=device,
            dtype=torch.bfloat16,
        )
    else:
        layer = SimpleFP4Linear(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
    with torch.no_grad():
        layer.weight.copy_(weight)
        if args.fused_rms:
            layer.norm_weight.copy_(norm_weight)
    x_fp4 = x.detach().clone().requires_grad_(True)
    y_fp4 = layer(x_fp4)
    torch.cuda.synchronize(device)
    y_fp4.backward(dy)
    torch.cuda.synchronize(device)

    with torch.no_grad():
        x_f = x.float()
        weight_f = weight.float()
        dy_f = dy.float()
        if args.fused_rms:
            gamma_f = norm_weight.float()
            inv_rms = torch.rsqrt(
                x_f.square().mean(dim=-1, keepdim=True) + epsilon
            )
            normed_f = x_f * inv_rms * gamma_f
            d_normed_f = dy_f @ weight_f
            projected_f = d_normed_f * gamma_f
            dot_f = (projected_f * x_f).mean(dim=-1, keepdim=True)
            dx_ref = (
                inv_rms * projected_f
                - inv_rms.pow(3) * x_f * dot_f
            )
            dgamma_ref = (d_normed_f * x_f * inv_rms).sum(dim=0)
            y_ref = normed_f @ weight_f.t()
            dw_ref = dy_f.t() @ normed_f
        else:
            y_ref = x_f @ weight_f.t()
            dx_ref = dy_f @ weight_f
            dw_ref = dy_f.t() @ x_f

    result = {
        "backend": args.backend,
        "fused_rms": args.fused_rms,
        "shape": [rows, in_features, out_features],
        "padded_out_features": padded_out_features,
        "output_stride": list(y_fp4.stride()),
        "forward": _stats(y_fp4, y_ref),
        "forward_tail128": _stats(y_fp4[:, -128:], y_ref[:, -128:]),
        "dgrad": _stats(x_fp4.grad, dx_ref),
        "wgrad": _stats(layer.weight.grad, dw_ref),
        "wgrad_tail128": _stats(layer.weight.grad[-128:], dw_ref[-128:]),
        "forward_tail_nonzero_fraction": float(
            y_fp4[:, -128:].ne(0).float().mean().item()
        ),
        "wgrad_tail_nonzero_fraction": float(
            layer.weight.grad[-128:].ne(0).float().mean().item()
        ),
    }
    if args.fused_rms:
        result["norm_weight_grad"] = _stats(
            layer.norm_weight.grad, dgamma_ref
        )
    checks = [
        result["output_stride"] == [padded_out_features, 1],
        result["forward"]["finite"],
        result["forward_tail128"]["finite"],
        result["dgrad"]["finite"],
        result["wgrad"]["finite"],
        result["wgrad_tail128"]["finite"],
        result["forward"]["cosine"] >= 0.98,
        result["forward_tail128"]["cosine"] >= 0.98,
        result["dgrad"]["cosine"] >= 0.98,
        result["wgrad"]["cosine"] >= 0.98,
        result["wgrad_tail128"]["cosine"] >= 0.98,
        result["forward_tail_nonzero_fraction"] >= 0.99,
        result["wgrad_tail_nonzero_fraction"] >= 0.99,
    ]
    if args.fused_rms:
        checks.extend(
            [
                result["norm_weight_grad"]["finite"],
                result["norm_weight_grad"]["cosine"] >= 0.98,
            ]
        )
    result["passed"] = all(checks)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
