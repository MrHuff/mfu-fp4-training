#!/usr/bin/env python3
"""Diagnose logical-to-swizzled localCTA row-scale placement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from analyze_localcta_swiglu_sr_bias import QUANT_RELATIVE_PATH, _load_extension


def _unswizzle_row_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Return hardware-swizzled E4M3 scales as logical [row, col/16]."""
    flat = scales.reshape(-1)
    row = torch.arange(rows, device=scales.device).reshape(-1, 1)
    k_block = torch.arange(cols // 16, device=scales.device).reshape(1, -1)
    k_block_groups = cols // 64
    block_base = ((row // 128) * k_block_groups + k_block // 4) * 512
    local = (row % 32) * 16 + ((row // 32) % 4) * 4 + k_block % 4
    return flat[(block_base + local).long()].float()


def _pattern(
    rows: int,
    cols: int,
    axis: str,
    amplitude: float,
    device: torch.device,
) -> torch.Tensor:
    row = torch.arange(rows, device=device).reshape(-1, 1)
    k_block = torch.arange(cols // 16, device=device).reshape(1, -1)
    if axis == "k":
        exponent = k_block % 8
    elif axis == "row":
        exponent = row % 8
    elif axis == "tile":
        tile_row = row // 128
        tile_col = k_block // 8
        amplitude = amplitude * torch.pow(
            2.0, -((tile_row * 3 + tile_col) % 4).float()
        )
        exponent = (k_block % 8 + tile_row + tile_col) % 8
    else:
        exponent = (row % 4) * 2 + (k_block % 2)
    exponent = exponent.expand(rows, cols // 16)
    block_values = amplitude * torch.pow(2.0, -exponent.float())
    return block_values.repeat_interleave(16, dim=1).to(torch.bfloat16)


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    os.environ["USE_TK_LOCALCTA_SCALE_NUM"] = str(args.scale_num)
    os.environ["USE_TK_LOCALCTA_V4_FAST_DATA_SR"] = "1"
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    quant = _load_extension(args.runtime_root / QUANT_RELATIVE_PATH)
    quant.tk_localcta_set_global_scale_num(args.scale_num)

    dh = _pattern(args.rows, args.cols, args.axis, args.amplitude, device)
    h1 = torch.ones_like(dh)
    h3 = torch.zeros_like(dh)
    dh1 = torch.empty_like(dh)
    dh3 = torch.empty_like(dh)
    buffers = quant.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
        args.rows, args.cols, device
    )
    def quantize():
        return quant.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace(
            dh,
            h3,
            h1,
            dh1,
            dh3,
            *buffers,
            False,
            True,
            False,
            args.seed,
            0,
            "row",
        )

    result = quantize()
    row_fp4, row_sc, _, _, row_sg, _ = result[6:12]
    torch.cuda.synchronize(device)

    logical_scales = _unswizzle_row_scales(row_sc, args.rows, args.cols)
    sg = row_sg.reshape(args.rows // 128, args.cols // 128)
    sg_per_block = (
        sg.repeat_interleave(128, dim=0)
        .repeat_interleave(8, dim=1)
    )
    ceiling = logical_scales * sg_per_block * 6.0
    exact_blocks = dh3.float().abs().reshape(args.rows, args.cols // 16, 16).amax(dim=2)
    ratio = ceiling / exact_blocks.clamp_min(1.0e-30)

    max_payload = torch.empty_like(row_fp4)
    max_payload.view(torch.uint8).fill_(0x77)
    reconstructed_ceiling = quant.tk_localcta_reconstruct_row(
        max_payload, row_sc, row_sg
    ).float().abs().reshape(args.rows, args.cols // 16, 16).amax(dim=2)

    sample_rows = [0, 1, 7, 8, 31, 32, 63, 64, 95, 96, 127, 128]
    sample_rows = [row for row in sample_rows if row < args.rows]
    report = {
        "shape": [args.rows, args.cols],
        "axis": args.axis,
        "scale_num": args.scale_num,
        "amplitude": args.amplitude,
        "ratio": {
            "min": float(ratio.min()),
            "p01": float(torch.quantile(ratio, 0.01)),
            "p50": float(torch.quantile(ratio, 0.50)),
            "p99": float(torch.quantile(ratio, 0.99)),
            "max": float(ratio.max()),
            "below_0_9_fraction": float((ratio < 0.9).float().mean()),
        },
        "reconstruct_manual_ceiling_max_abs_diff": float(
            (reconstructed_ceiling - ceiling).abs().max()
        ),
        "sample_rows": {
            str(row): {
                "exact": exact_blocks[row, :16].tolist(),
                "scale": logical_scales[row, :16].tolist(),
                "ceiling": ceiling[row, :16].tolist(),
                "ratio": ratio[row, :16].tolist(),
            }
            for row in sample_rows
        },
    }
    if args.benchmark_iters:
        for _ in range(args.benchmark_warmup):
            quantize()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.benchmark_iters):
            quantize()
        end.record()
        end.synchronize()
        report["quantize_ms"] = start.elapsed_time(end) / args.benchmark_iters
    if args.summary_only:
        report.pop("sample_rows")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=Path("/tmp/localcta-prod-35ac612-20260816"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=256)
    parser.add_argument(
        "--axis", choices=("row", "k", "tile", "both"), default="k"
    )
    parser.add_argument("--scale-num", type=float, default=448.0)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.rows % 256 or args.cols % 256:
        parser.error("rows and cols must be multiples of 256")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
