#!/usr/bin/env python3
"""Exact-shape MXFP4 W2-dgrad SiLU producer diagnostic.

This targets the Megatron Bridge Llama 3 8B TP=2 FFN backward shard:
M=32768, K=4096, H=7168 by default.
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


def byte_diff_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().view(torch.uint8).reshape(-1)
    bv = b.detach().view(torch.uint8).reshape(-1)
    return float((av != bv).float().mean().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--h", type=int, default=7168)
    parser.add_argument("--config-id", type=int, default=4)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--device-index", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--producer-repeats", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--fresh-outputs", action="store_true")
    parser.add_argument("--producer-only-bench", action="store_true")
    parser.add_argument("--from-sigmoid", action="store_true")
    parser.add_argument("--row-bf16-producer", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MXFP4_BACKEND_VERSION", "v4")
    os.environ.setdefault("FP4_MXFP4_ROOT", "/opt/mfu/EXTERNAL_PATH")
    os.environ.setdefault("FP4_CCE_TK_ROOT", "/opt/mfu/EXTERNAL_PATH")

    from low_bits_training.quantization.fused_te_linear import _get_te_fused
    from low_bits_training.quantization.mxfp4_backend import (
        mxfp4_gemm,
        mxfp4_gemm_silu_dgrad_from_sigmoid_row_bf16_quant,
        mxfp4_gemm_silu_dgrad_from_sigmoid_quant,
        mxfp4_gemm_silu_dgrad_quant,
        mxfp4_quantize_split2_col_only_launch_inplace,
        mxfp4_quantize_row_and_col,
        mxfp4_quantize_split2_row_and_col_launch_inplace,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    device_index = local_rank if args.device_index is None else args.device_index
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    M, K, H = args.m, args.k, args.h
    if M % 256 != 0 or K % 256 != 0 or H % 256 != 0:
        raise ValueError("M, K, and H must be divisible by 256")

    torch.manual_seed(args.seed)
    dY = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(K, H, device=device, dtype=torch.bfloat16)
    h3 = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    h1 = torch.randn(M, H, device=device, dtype=torch.bfloat16)
    sig_h1 = torch.sigmoid(h1.float()).to(torch.bfloat16) if args.from_sigmoid else None

    dY_row_fp4, dY_row_sc, _, _ = mxfp4_quantize_row_and_col(dY, args.mode)
    _, _, w2_col_fp4, w2_col_sc = mxfp4_quantize_row_and_col(w2, args.mode)

    dh = torch.empty(M, H, device=device, dtype=torch.bfloat16)
    dh1 = torch.empty_like(dh)
    dh3 = torch.empty_like(dh)
    def make_outputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty((M, H), device=device, dtype=torch.float4_e2m1fn_x2),
            torch.empty((M // 128, (2 * H) // 128, 32, 16), device=device, dtype=torch.uint8),
            torch.empty((H, M // 2), device=device, dtype=torch.float4_e2m1fn_x2),
            torch.empty((H // 128, M // 128, 32, 16), device=device, dtype=torch.uint8),
            torch.empty((H, M // 2), device=device, dtype=torch.float4_e2m1fn_x2),
            torch.empty((H // 128, M // 128, 32, 16), device=device, dtype=torch.uint8),
        )

    row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc = make_outputs()

    row_ref = torch.empty_like(row_fp4)
    row_sc_ref = torch.empty_like(row_sc)
    col_ref = torch.empty((2 * H, M // 2), device=device, dtype=torch.float4_e2m1fn_x2)
    col_sc_ref = torch.empty((2 * H // 128, M // 128, 32, 16), device=device, dtype=torch.uint8)
    row_bf16_col_fp4 = torch.empty_like(col_ref)
    row_bf16_col_sc = torch.empty_like(col_sc_ref)

    te_fused = _get_te_fused()

    def te_deriv() -> None:
        if args.from_sigmoid and hasattr(te_fused, "fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax"):
            te_fused.fused_silu_deriv_dual_mul_from_sigmoid_bf16_out_no_amax(
                dh, h3, h1, sig_h1, dh1, dh3
            )
        elif args.from_sigmoid and hasattr(te_fused, "fused_silu_deriv_dual_mul_from_sigmoid_bf16_out"):
            amax1 = torch.empty(1, dtype=torch.float32, device=device)
            amax2 = torch.empty(1, dtype=torch.float32, device=device)
            te_fused.fused_silu_deriv_dual_mul_from_sigmoid_bf16_out(
                dh, h3, h1, sig_h1, dh1, dh3, amax1, amax2
            )
        elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
            te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(dh, h3, h1, dh1, dh3)
        elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out"):
            amax1 = torch.empty(1, dtype=torch.float32, device=device)
            amax2 = torch.empty(1, dtype=torch.float32, device=device)
            te_fused.fused_silu_deriv_dual_mul_bf16_out(dh, h3, h1, dh1, dh3, amax1, amax2)
        else:
            out1, out3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, h3, h1)
            dh1.copy_(out1)
            dh3.copy_(out3)

    def separate_path() -> None:
        mxfp4_gemm(dY_row_fp4, dY_row_sc, w2_col_fp4, w2_col_sc, dh)
        te_deriv()
        mxfp4_quantize_split2_row_and_col_launch_inplace(
            dh1, dh3, row_ref, row_sc_ref, col_ref, col_sc_ref, args.mode
        )

    def producer_path(
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> None:
        out_row_fp4, out_row_sc, out_col0_fp4, out_col0_sc, out_col1_fp4, out_col1_sc = (
            make_outputs() if outputs is None and args.fresh_outputs else (
                outputs if outputs is not None else (row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc)
            )
        )
        if args.from_sigmoid:
            ok = mxfp4_gemm_silu_dgrad_from_sigmoid_quant(
                dY_row_fp4,
                dY_row_sc,
                w2_col_fp4,
                w2_col_sc,
                h3,
                h1,
                sig_h1,
                out_row_fp4,
                out_row_sc,
                out_col0_fp4,
                out_col0_sc,
                out_col1_fp4,
                out_col1_sc,
                config_id=args.config_id,
                mode=args.mode,
            )
        else:
            ok = mxfp4_gemm_silu_dgrad_quant(
                dY_row_fp4,
                dY_row_sc,
                w2_col_fp4,
                w2_col_sc,
                h3,
                h1,
                out_row_fp4,
                out_row_sc,
                out_col0_fp4,
                out_col0_sc,
                out_col1_fp4,
                out_col1_sc,
                config_id=args.config_id,
                mode=args.mode,
            )
        if not ok:
            raise RuntimeError(
                "mxfp4_gemm_silu_dgrad_from_sigmoid_quant backend is unavailable"
                if args.from_sigmoid
                else "mxfp4_gemm_silu_dgrad_quant backend is unavailable"
            )

    def row_bf16_producer_path(
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
    ) -> None:
        if not args.from_sigmoid:
            raise RuntimeError("--row-bf16-producer requires --from-sigmoid")
        out_row_fp4, out_row_sc, _, _, _, _ = (
            make_outputs() if outputs is None and args.fresh_outputs else (
                outputs if outputs is not None else (row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc)
            )
        )
        ok = mxfp4_gemm_silu_dgrad_from_sigmoid_row_bf16_quant(
            dY_row_fp4,
            dY_row_sc,
            w2_col_fp4,
            w2_col_sc,
            h3,
            h1,
            sig_h1,
            dh1,
            dh3,
            out_row_fp4,
            out_row_sc,
            config_id=args.config_id,
            mode=args.mode,
        )
        if not ok:
            raise RuntimeError("mxfp4_gemm_silu_dgrad_from_sigmoid_row_bf16_quant backend is unavailable")
        mxfp4_quantize_split2_col_only_launch_inplace(dh1, dh3, row_bf16_col_fp4, row_bf16_col_sc)

    if args.producer_only_bench:
        timings = []
        torch.cuda.synchronize()
        for _ in range(args.iters):
            outputs = make_outputs() if args.fresh_outputs else (row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if args.row_bf16_producer:
                row_bf16_producer_path(outputs)
            else:
                producer_path(outputs)
            end.record()
            torch.cuda.synchronize()
            timings.append(float(start.elapsed_time(end)))
        print(json.dumps({
            "rank": rank,
            "local_rank": local_rank,
            "device_index": device_index,
            "M": M,
            "K": K,
            "H": H,
            "config_id": args.config_id,
            "mode": args.mode,
            "from_sigmoid": args.from_sigmoid,
            "producer_only_ms": timings,
        }, indent=2, sort_keys=True))
        return 0

    separate_path()
    if args.row_bf16_producer:
        row_bf16_producer_path((row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc))
    else:
        producer_path((row_fp4, row_sc, col0_fp4, col0_sc, col1_fp4, col1_sc))
    torch.cuda.synchronize()

    col0_ref = col_ref.narrow(0, 0, H)
    col1_ref = col_ref.narrow(0, H, H)
    col0_sc_ref = col_sc_ref.narrow(0, 0, H // 128)
    col1_sc_ref = col_sc_ref.narrow(0, H // 128, H // 128)
    col0_out = row_bf16_col_fp4.narrow(0, 0, H) if args.row_bf16_producer else col0_fp4
    col1_out = row_bf16_col_fp4.narrow(0, H, H) if args.row_bf16_producer else col1_fp4
    col0_sc_out = row_bf16_col_sc.narrow(0, 0, H // 128) if args.row_bf16_producer else col0_sc
    col1_sc_out = row_bf16_col_sc.narrow(0, H // 128, H // 128) if args.row_bf16_producer else col1_sc

    results = {
        "rank": rank,
        "local_rank": local_rank,
        "device_index": device_index,
        "M": M,
        "K": K,
        "H": H,
        "config_id": args.config_id,
        "mode": args.mode,
        "from_sigmoid": args.from_sigmoid,
        "fresh_outputs": args.fresh_outputs,
        "finite_dh": bool(torch.isfinite(dh).all().item()),
        "diff_rate": {
            "row_fp4": byte_diff_rate(row_fp4, row_ref),
            "row_sc": byte_diff_rate(row_sc, row_sc_ref),
            "col0_fp4": byte_diff_rate(col0_out, col0_ref),
            "col1_fp4": byte_diff_rate(col1_out, col1_ref),
            "col0_sc": byte_diff_rate(col0_sc_out, col0_sc_ref),
            "col1_sc": byte_diff_rate(col1_sc_out, col1_sc_ref),
        },
    }
    if args.producer_repeats > 0:
        completed = 0
        for _ in range(args.producer_repeats):
            if args.row_bf16_producer:
                row_bf16_producer_path()
            else:
                producer_path()
            torch.cuda.synchronize()
            completed += 1
            if args.progress_every > 0 and completed % args.progress_every == 0:
                print(json.dumps({"producer_repeats_completed": completed}), flush=True)
        results["producer_repeats_completed"] = completed
    if not args.no_bench:
        results["timings_ms"] = {
            "separate_gemm_te_quant": bench(separate_path, args.warmup, args.iters),
            "producer": bench(row_bf16_producer_path if args.row_bf16_producer else producer_path, args.warmup, args.iters),
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
