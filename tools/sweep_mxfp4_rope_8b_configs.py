#!/usr/bin/env python3
"""Sweep the exact Llama-3 8B MXFP4 Q and K/V RoPE launch configs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = Path("/opt/mfu/EXTERNAL_PATH")


def _setup_paths(runtime_root: Path) -> None:
    root = str(runtime_root)
    os.environ.setdefault("FP4_MATMUL_ROOT", root)
    os.environ.setdefault("FP4_MXFP4_ROOT", root)
    os.environ.setdefault("FP4_MATMUL_GEMM_ROOT", root)
    sys.path.insert(0, str(REPO_ROOT))


def _rope_table(seq_len: int, pair_dim: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (
        500_000.0
        ** (torch.arange(pair_dim, dtype=torch.float32, device=device) / pair_dim)
    )
    angles = torch.outer(positions, inv_freq)
    return torch.stack((angles.cos(), angles.sin()), dim=-1).contiguous()


def _bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _difference(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
) -> tuple[bool, float]:
    bitwise = all(torch.equal(x, y) for x, y in zip(actual, expected))
    max_abs = max(
        (x.float() - y.float()).abs().max().item()
        for x, y in zip(actual, expected)
    )
    return bitwise, max_abs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--configs", default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--m", type=int, default=32_768)
    parser.add_argument("--k", type=int, default=4_096)
    parser.add_argument("--q-dim", type=int, default=4_096)
    parser.add_argument("--kv-dim", type=int, default=1_024)
    parser.add_argument("--seq-len", type=int, default=8_192)
    parser.add_argument("--rope-pairs", type=int, default=64)
    parser.add_argument("--csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _setup_paths(args.runtime_root)
    from low_bits_training.quantization.mxfp4_backend import (  # noqa: WPS433
        mxfp4_batched_gemm_rope_live64_config,
        mxfp4_gemm_rope_live64_config,
        mxfp4_quantize_for_gemm,
    )

    configs = [int(value) for value in args.configs.split(",") if value.strip()]
    device = torch.device("cuda")
    torch.manual_seed(42)

    x = torch.randn(args.m, args.k, dtype=torch.bfloat16, device=device)
    q_weight = torch.randn(args.q_dim, args.k, dtype=torch.bfloat16, device=device)
    k_weight = torch.randn(args.kv_dim, args.k, dtype=torch.bfloat16, device=device)
    v_weight = torch.randn(args.kv_dim, args.k, dtype=torch.bfloat16, device=device)
    x_fp4, x_sc = mxfp4_quantize_for_gemm(x, 1)
    q_fp4, q_sc = mxfp4_quantize_for_gemm(q_weight, 1)
    k_fp4, k_sc = mxfp4_quantize_for_gemm(k_weight, 1)
    v_fp4, v_sc = mxfp4_quantize_for_gemm(v_weight, 1)
    del x, q_weight, k_weight, v_weight

    rope_cs = _rope_table(args.seq_len, args.rope_pairs, device)
    rope_empty = torch.empty(0, dtype=torch.float32, device=device)
    q_out = torch.empty(args.m, args.q_dim, dtype=torch.bfloat16, device=device)
    k_out = torch.empty(args.m, args.kv_dim, dtype=torch.bfloat16, device=device)
    v_out = torch.empty_like(k_out)

    def launch_q(config: int) -> list[torch.Tensor]:
        mxfp4_gemm_rope_live64_config(
            x_fp4,
            x_sc,
            q_fp4,
            q_sc,
            rope_cs,
            args.seq_len,
            q_out,
            config_id=config,
        )
        return [q_out]

    def launch_kv(config: int) -> list[torch.Tensor]:
        mxfp4_batched_gemm_rope_live64_config(
            [x_fp4, x_fp4],
            [x_sc, x_sc],
            [k_fp4, v_fp4],
            [k_sc, v_sc],
            [rope_cs, rope_empty],
            [args.seq_len, 0],
            [k_out, v_out],
            config_id=config,
        )
        return [k_out, v_out]

    references: dict[str, list[torch.Tensor]] = {}
    for name, launch in (("q_rope", launch_q), ("kv_rope", launch_kv)):
        references[name] = [tensor.clone() for tensor in launch(0)]
    torch.cuda.synchronize()

    rows: list[dict[str, object]] = []
    print("path,config,avg_ms,bitwise_config0,max_abs_config0,status", flush=True)
    for name, launch in (("q_rope", launch_q), ("kv_rope", launch_kv)):
        for config in configs:
            try:
                outputs = launch(config)
                torch.cuda.synchronize()
                bitwise, max_abs = _difference(outputs, references[name])
                avg_ms = _bench(lambda: launch(config), args.warmup, args.iters)
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                avg_ms = float("nan")
                bitwise = False
                max_abs = float("nan")
                status = type(exc).__name__ + ":" + str(exc).replace(",", ";")[:160]
            row = {
                "path": name,
                "config": config,
                "avg_ms": avg_ms,
                "bitwise_config0": bitwise,
                "max_abs_config0": max_abs,
                "status": status,
            }
            rows.append(row)
            print(
                f"{name},{config},{avg_ms:.6f},{int(bitwise)},"
                f"{max_abs:.9g},{status}",
                flush=True,
            )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
