#!/usr/bin/env python3
"""Isolated MXFP4 split2 SiLU-derivative producer benchmark.

This targets the Megatron Bridge Llama 3 8B TP=2 FFN backward shard:
M=32768, H=7168 by default.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Callable

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def bench(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return float(statistics.median(samples))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    denom = float(af.norm() * bf.norm())
    if denom == 0.0:
        return float("nan")
    return float((af @ bf) / denom)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--h", type=int, default=7168)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    os.environ.setdefault("MXFP4_BACKEND_VERSION", "v4")
    os.environ.setdefault("FP4_MXFP4_ROOT", "/opt/mfu/EXTERNAL_PATH")
    os.environ.setdefault("FP4_CCE_TK_ROOT", "/opt/mfu/EXTERNAL_PATH")

    from low_bits_training.quantization.fused_te_linear import _get_te_fused
    from low_bits_training.quantization.mxfp4_backend import (
        mxfp4_fused_silu_deriv_quantize_split2_row_and_col,
        mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols,
        mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace,
        mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace,
        mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace,
        mxfp4_quantize_split2_col_only_launch_inplace,
        mxfp4_quantize_split2_row_and_col_launch_inplace,
    )

    torch.cuda.set_device(args.device_index)
    device = torch.device(f"cuda:{args.device_index}")
    M = args.m
    H = args.h
    if M % 128 != 0 or H % 128 != 0:
        raise ValueError("M and H must be divisible by 128")

    torch.manual_seed(args.seed)
    dh = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    h3 = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    h1 = torch.randn(M, H, device=device, dtype=torch.bfloat16)

    dh1 = torch.empty_like(dh)
    dh3 = torch.empty_like(dh)
    row_fp4 = torch.empty((M, H), device=device, dtype=torch.float4_e2m1fn_x2)
    row_sc = torch.empty((M // 128, (2 * H) // 128, 32, 16), device=device, dtype=torch.uint8)
    row_fp4_legacy = torch.empty((2, M, H // 2), device=device, dtype=torch.float4_e2m1fn_x2)
    row_sc_legacy = torch.empty((2, M // 128, H // 128, 32, 16), device=device, dtype=torch.uint8)
    col_fp4_legacy = torch.empty((2, H, M // 2), device=device, dtype=torch.float4_e2m1fn_x2)
    col_sc_legacy = torch.empty((2, H // 128, M // 128, 32, 16), device=device, dtype=torch.uint8)
    col_fp4 = torch.empty((2 * H, M // 2), device=device, dtype=torch.float4_e2m1fn_x2)
    col_sc = torch.empty(((2 * H) // 128, M // 128, 32, 16), device=device, dtype=torch.uint8)
    col0_fp4 = torch.empty((H, M // 2), device=device, dtype=torch.float4_e2m1fn_x2)
    col1_fp4 = torch.empty((H, M // 2), device=device, dtype=torch.float4_e2m1fn_x2)
    col0_sc = torch.empty((H // 128, M // 128, 32, 16), device=device, dtype=torch.uint8)
    col1_sc = torch.empty((H // 128, M // 128, 32, 16), device=device, dtype=torch.uint8)

    te_fused = _get_te_fused()

    def te_deriv() -> None:
        if hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
            te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(dh, h3, h1, dh1, dh3)
        elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out"):
            amax1 = torch.empty(1, dtype=torch.float32, device=device)
            amax2 = torch.empty(1, dtype=torch.float32, device=device)
            te_fused.fused_silu_deriv_dual_mul_bf16_out(dh, h3, h1, dh1, dh3, amax1, amax2)
        else:
            out1, out3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1)
            dh1.copy_(out1)
            dh3.copy_(out3)

    def fused_splitcols_inplace() -> None:
        mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols_launch_inplace(
            dh, h3, h1, row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc, 1
        )

    def fused_splitcols_alloc() -> None:
        mxfp4_fused_silu_deriv_quantize_split2_row_and_col_splitcols(dh, h3, h1, 1)

    def fused_legacy_alloc() -> None:
        mxfp4_fused_silu_deriv_quantize_split2_row_and_col(dh, h3, h1, 1)

    def te_then_quant_splitcols() -> None:
        te_deriv()
        mxfp4_quantize_split2_row_and_col_launch_inplace(
            dh1, dh3, row_fp4, row_sc, col_fp4, col_sc, 1
        )

    def row_bf16_then_col() -> None:
        mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace(
            dh, h3, h1, dh1, dh3, row_fp4, row_sc, 1
        )
        mxfp4_quantize_split2_col_only_launch_inplace(dh1, dh3, col_fp4, col_sc, 1)

    def row_bf16_tile_then_col() -> None:
        mxfp4_fused_silu_deriv_quantize_split2_row_bf16_tile_launch_inplace(
            dh, h3, h1, dh1, dh3, row_fp4, row_sc, 1
        )
        mxfp4_quantize_split2_col_only_launch_inplace(dh1, dh3, col_fp4, col_sc, 1)

    # Correctness smoke: compare BF16 derivative producer to the row-BF16 path.
    te_deriv()
    ref_dh1 = dh1.detach().clone()
    ref_dh3 = dh3.detach().clone()
    mxfp4_fused_silu_deriv_quantize_split2_row_bf16_launch_inplace(
        dh, h3, h1, dh1, dh3, row_fp4, row_sc, 1
    )
    torch.cuda.synchronize()

    results = {
        "device_index": args.device_index,
        "M": M,
        "H": H,
        "finite_row_bf16": bool(torch.isfinite(dh1).all().item() and torch.isfinite(dh3).all().item()),
        "row_bf16_cos_dh1": cosine(dh1, ref_dh1),
        "row_bf16_cos_dh3": cosine(dh3, ref_dh3),
        "timings_ms": {
            "te_deriv_only": bench(te_deriv, args.warmup, args.iters),
            "te_then_quant_splitcols": bench(te_then_quant_splitcols, args.warmup, args.iters),
            "fused_splitcols_inplace": bench(fused_splitcols_inplace, args.warmup, args.iters),
            "fused_splitcols_alloc": bench(fused_splitcols_alloc, args.warmup, args.iters),
            "fused_legacy_alloc": bench(fused_legacy_alloc, args.warmup, args.iters),
            "row_bf16_then_col": bench(row_bf16_then_col, args.warmup, args.iters),
            "row_bf16_tile_then_col": bench(row_bf16_tile_then_col, args.warmup, args.iters),
        },
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
