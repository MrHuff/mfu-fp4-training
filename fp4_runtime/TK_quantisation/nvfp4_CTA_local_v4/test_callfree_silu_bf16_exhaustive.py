#!/usr/bin/env python3
"""Exhaustive BF16-domain gate for the paired-W2 call-free SiLU carrier."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mismatch_bytes(lhs: torch.Tensor, rhs: torch.Tensor) -> int:
    return int((lhs.view(torch.uint8) != rhs.view(torch.uint8)).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--te-extension", required=True, type=Path)
    parser.add_argument("--chunk-cols", type=int, default=1024)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.chunk_cols <= 0 or args.chunk_cols % 64 != 0:
        raise ValueError("--chunk-cols must be a positive multiple of 64")
    if 65536 % args.chunk_cols != 0:
        raise ValueError("--chunk-cols must divide 65536")
    if args.num_shards <= 0 or 65536 % args.num_shards != 0:
        raise ValueError("--num-shards must be a positive divisor of 65536")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    shard_cols = 65536 // args.num_shards
    if shard_cols % args.chunk_cols != 0:
        raise ValueError("--chunk-cols must divide each shard's column count")
    shard_start = args.shard_index * shard_cols
    shard_stop = shard_start + shard_cols

    module = _load(args.extension.resolve(), "_tk_quant_localcta_v4")
    te_module = _load(
        args.te_extension.resolve(), "te_fused_rmsnorm_ext_linear"
    )
    values = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.uint16)
        .view(torch.bfloat16)
    )

    checked_pairs = 0
    for start in range(shard_start, shard_stop, args.chunk_cols):
        h1 = values[:, None].expand(65536, args.chunk_cols).contiguous()
        h3 = values[start : start + args.chunk_cols][None, :].expand(
            65536, args.chunk_cols
        ).contiguous()

        precise_fast, precise_fast_amax = (
            module.tk_localcta_test_w2_transform_bf16_exact(
                h1, h3, True, False
            )
        )
        callfree_fast, callfree_fast_amax = (
            module.tk_localcta_test_w2_transform_bf16_exact(
                h1, h3, True, True
            )
        )
        precise_slow, precise_slow_amax = (
            module.tk_localcta_test_w2_transform_bf16_exact(
                h1, h3, False, False
            )
        )
        callfree_slow, callfree_slow_amax = (
            module.tk_localcta_test_w2_transform_bf16_exact(
                h1, h3, False, True
            )
        )
        te_fast = torch.empty_like(h1)
        te_module.fused_silu_mul_bf16_out_no_amax(h1, h3, te_fast)
        torch.cuda.synchronize()

        checks = {
            "callfree_fast_vs_precise": _mismatch_bytes(
                callfree_fast, precise_fast
            ),
            "callfree_fast_vs_te": _mismatch_bytes(callfree_fast, te_fast),
            "callfree_slow_vs_precise": _mismatch_bytes(
                callfree_slow, precise_slow
            ),
            "callfree_fast_amax": _mismatch_bytes(
                callfree_fast_amax, precise_fast_amax
            ),
            "callfree_slow_amax": _mismatch_bytes(
                callfree_slow_amax, precise_slow_amax
            ),
        }
        failures = {name: count for name, count in checks.items() if count}
        if failures:
            raise AssertionError(
                f"BF16 exhaustive gate failed at h3 bits "
                f"[{start}, {start + args.chunk_cols}): {failures}"
            )
        checked_pairs += 65536 * args.chunk_cols
        print(
            f"BF16 exhaustive progress: h3_end={start + args.chunk_cols} "
            f"pairs={checked_pairs}",
            flush=True,
        )

    expected_pairs = 65536 * shard_cols
    if checked_pairs != expected_pairs:
        raise AssertionError(f"unexpected checked-pair count: {checked_pairs}")
    print(
        "call-free SiLU exhaustive BF16 gate passed: "
        f"shard={args.shard_index}/{args.num_shards} "
        f"h3=[{shard_start},{shard_stop}) pairs={checked_pairs} "
        "fast/slow/TE/amax exact"
    )


if __name__ == "__main__":
    main()
