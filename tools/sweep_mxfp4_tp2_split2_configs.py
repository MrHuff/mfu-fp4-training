#!/usr/bin/env python3
"""Sweep MXFP4 split2 one-pass dgrad configs on the Bridge TP=2 FFN shape."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP4_ROOT = Path("/opt/mfu/EXTERNAL_PATH")
DEFAULT_GEMM_ROOT = Path("/opt/mfu/EXTERNAL_PATH")


def _setup_paths(fp4_root: Path, gemm_root: Path) -> None:
    os.environ.setdefault("FP4_MATMUL_ROOT", str(fp4_root))
    os.environ.setdefault("FP4_MXFP4_ROOT", str(fp4_root))
    os.environ.setdefault("FP4_CCE_TK_ROOT", str(fp4_root))
    os.environ.setdefault("FP4_MATMUL_GEMM_ROOT", str(gemm_root))
    sys.path.insert(0, str(REPO_ROOT))


def _bench(fn, warmup: int, iters: int) -> float:
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
    return statistics.median(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=7168, help="Logical FFN shard width per split.")
    parser.add_argument("--out", type=int, default=4096, help="Output hidden dimension.")
    parser.add_argument("--configs", default="1,3,5")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--fp4-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument("--gemm-root", type=Path, default=DEFAULT_GEMM_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_paths(args.fp4_root, args.gemm_root)

    from low_bits_training.quantization.mxfp4_backend import (  # noqa: WPS433
        mxfp4_split2_dgrad_strided_onepass_gemm,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda")
    m = args.m
    h = args.hidden
    n = args.out
    if m % 256 != 0 or h % 256 != 0 or n % 256 != 0:
        raise ValueError("m, hidden, and out must be divisible by 256")

    packed_h = h // 2
    h_sc = h // 128
    configs = [int(x) for x in args.configs.split(",") if x.strip()]

    row_fp4 = torch.empty((m, h), device=device, dtype=torch.float4_e2m1fn_x2)
    row_sc_full = torch.empty((m // 128, 2 * h_sc, 32, 16), device=device, dtype=torch.uint8)
    b_fp4 = [
        torch.empty((n, packed_h), device=device, dtype=torch.float4_e2m1fn_x2),
        torch.empty((n, packed_h), device=device, dtype=torch.float4_e2m1fn_x2),
    ]
    b_sc = [
        torch.empty((n // 128, h_sc, 32, 16), device=device, dtype=torch.uint8),
        torch.empty((n // 128, h_sc, 32, 16), device=device, dtype=torch.uint8),
    ]
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    a_sc = [
        row_sc_full.narrow(1, 0, h_sc),
        row_sc_full.narrow(1, h_sc, h_sc),
    ]
    offsets = [0, packed_h]
    widths = [packed_h, packed_h]
    flops = 2.0 * m * n * h * 2

    print("shape,config,median_ms,tflops,status", flush=True)
    for config in configs:
        try:
            fn = lambda config=config: mxfp4_split2_dgrad_strided_onepass_gemm(
                row_fp4,
                a_sc,
                offsets,
                widths,
                b_fp4,
                b_sc,
                out,
                config,
            )
            median_ms = _bench(fn, args.warmup, args.iters)
            tflops = flops / (median_ms * 1e-3) * 1e-12
            print(f"tp2_split2_dgrad,{config},{median_ms:.6f},{tflops:.2f},ok", flush=True)
        except Exception as exc:  # noqa: BLE001
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            msg = f"{type(exc).__name__}:{str(exc).replace(',', ';')[:160]}"
            print(f"tp2_split2_dgrad,{config},nan,nan,{msg}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
