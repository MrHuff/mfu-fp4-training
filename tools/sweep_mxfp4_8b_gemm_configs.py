#!/usr/bin/env python3
"""Sweep MXFP4 dense GEMM config IDs on Llama-3 8B hot shapes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP4_ROOT = Path("/opt/mfu/EXTERNAL_PATH")
DEFAULT_GEMM_ROOT = Path("/opt/mfu/EXTERNAL_PATH")


@dataclass(frozen=True)
class Shape:
    name: str
    m: int
    n: int
    k: int
    residual: bool = False


def _setup_paths(fp4_root: Path, gemm_root: Path) -> None:
    os.environ.setdefault("FP4_MATMUL_ROOT", str(fp4_root))
    os.environ.setdefault("FP4_MXFP4_ROOT", str(fp4_root))
    os.environ.setdefault("FP4_MATMUL_GEMM_ROOT", str(gemm_root))
    sys.path.insert(0, str(REPO_ROOT))


def _alloc_shape(shape: Shape, device: torch.device):
    a = torch.empty(shape.m, shape.k // 2, dtype=torch.float4_e2m1fn_x2, device=device)
    a_sc = torch.empty(shape.m // 128, shape.k // 128, 32, 16, dtype=torch.uint8, device=device)
    b = torch.empty(shape.n, shape.k // 2, dtype=torch.float4_e2m1fn_x2, device=device)
    b_sc = torch.empty(shape.n // 128, shape.k // 128, 32, 16, dtype=torch.uint8, device=device)
    out = torch.empty(shape.m, shape.n, dtype=torch.bfloat16, device=device)
    residual = torch.empty_like(out) if shape.residual else None
    return a, a_sc, b, b_sc, out, residual


def _bench(fn, args, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _default_shapes() -> list[Shape]:
    # Llama-3 8B blog shape: local batch 4, seq 8192, dim 4096,
    # SwiGLU hidden dim 14336.
    return [
        Shape("qkv_fwd", 32768, 6144, 4096),
        Shape("wo_fwd", 32768, 4096, 4096),
        Shape("ffn_w13_fwd", 32768, 14336, 4096),
        Shape("ffn_w2_fwd_residual", 32768, 4096, 14336, residual=True),
        Shape("qkv_wgrad", 6144, 4096, 32768),
        Shape("wo_wgrad", 4096, 4096, 32768),
        Shape("ffn_w2_wgrad", 4096, 14336, 32768),
        Shape("ffn_w13_wgrad_part", 14336, 4096, 32768),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--configs", default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--shape", action="append", default=[], help="Only run shape name; repeatable.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--fp4-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument("--gemm-root", type=Path, default=DEFAULT_GEMM_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_paths(args.fp4_root, args.gemm_root)
    from low_bits_training.quantization.mxfp4_backend import (  # noqa: WPS433
        mxfp4_gemm_config,
        mxfp4_gemm_residual_config,
    )

    configs = [int(x) for x in args.configs.split(",") if x.strip()]
    wanted = set(args.shape)
    shapes = [s for s in _default_shapes() if not wanted or s.name in wanted]
    device = torch.device("cuda")
    rows: list[dict[str, object]] = []
    print("shape,m,n,k,residual,config,avg_ms,tflops,status")
    for shape in shapes:
        tensors = _alloc_shape(shape, device)
        a, a_sc, b, b_sc, out, residual = tensors
        flops = 2.0 * shape.m * shape.n * shape.k
        for config in configs:
            try:
                if shape.residual:
                    assert residual is not None
                    fn = lambda A, As, B, Bs, R, O: mxfp4_gemm_residual_config(  # noqa: E731
                        A, As, B, Bs, R, O, config_id=config
                    )
                    bench_args = (a, a_sc, b, b_sc, residual, out)
                else:
                    fn = lambda A, As, B, Bs, O: mxfp4_gemm_config(  # noqa: E731
                        A, As, B, Bs, O, config_id=config
                    )
                    bench_args = (a, a_sc, b, b_sc, out)
                avg_ms = _bench(fn, bench_args, args.warmup, args.iters)
                tflops = flops / (avg_ms * 1e-3) * 1e-12
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                torch.cuda.synchronize()
                avg_ms = float("nan")
                tflops = float("nan")
                status = type(exc).__name__ + ":" + str(exc).replace(",", ";")[:160]
            row = {
                "shape": shape.name,
                "m": shape.m,
                "n": shape.n,
                "k": shape.k,
                "residual": shape.residual,
                "config": config,
                "avg_ms": avg_ms,
                "tflops": tflops,
                "status": status,
            }
            rows.append(row)
            print(
                f"{shape.name},{shape.m},{shape.n},{shape.k},{int(shape.residual)},"
                f"{config},{avg_ms:.6f},{tflops:.2f},{status}",
                flush=True,
            )
        del tensors
        torch.cuda.empty_cache()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
