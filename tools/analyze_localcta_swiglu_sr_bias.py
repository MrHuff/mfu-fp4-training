#!/usr/bin/env python3
"""Attribute localCTA SwiGLU dgrad bias on a captured FFN batch.

The mean RMSNorm-gamma gradient is linear in the quantized SwiGLU derivative.
This probe builds that linear sensitivity once, then measures stochastic FP4
rounding error without involving GEMM kernels, collectives, or the optimizer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_RUNTIME = Path("/tmp/localcta-prod-35ac612-20260816")
QUANT_RELATIVE_PATH = Path(
    "TK_quantisation/nvfp4_CTA_local_v4/"
    "_tk_quant_localcta_v4.cpython-312-aarch64-linux-gnu.so"
)
BIN_EDGES = (0.0, 1.0 / 64, 1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2, 0.75, 1.01)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--io-state",
        type=Path,
        default=Path("/tmp/localcta-layer31-ffn-io-step8000-dolma-b4s8192.pt"),
    )
    parser.add_argument(
        "--ffn-state",
        type=Path,
        default=Path("/tmp/localcta-layer31-ffn-step8000.pt"),
    )
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32768)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subsequence-base", type=int, default=0)
    parser.add_argument("--subsequence-stride", type=int, default=1_000_000_000)
    parser.add_argument("--scale-num", type=float, default=448.0)
    parser.add_argument("--fast-data-sr", choices=("0", "1"), default="1")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or args.rows % 256:
        parser.error("--rows must be a positive multiple of 256")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    return args


def _load_extension(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    name = "_tk_quant_localcta_v4"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load localCTA extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    error = candidate - reference
    ref_sq = float(reference.square().sum(dtype=torch.float64))
    error_sq = float(error.square().sum(dtype=torch.float64))
    dot = float((candidate * reference).sum(dtype=torch.float64))
    candidate_sq = float(candidate.square().sum(dtype=torch.float64))
    return {
        "relative_l2": math.sqrt(error_sq / max(ref_sq, 1.0e-300)),
        "cosine": dot / max(math.sqrt(ref_sq * candidate_sq), 1.0e-300),
        "rms_ratio": math.sqrt(candidate_sq / max(ref_sq, 1.0e-300)),
        "mean": float(candidate.mean()),
        "reference_mean": float(reference.mean()),
        "zero_fraction": float((candidate == 0).float().mean()),
    }


@torch.no_grad()
def _prepare_reference(args: argparse.Namespace, device: torch.device):
    io_state = torch.load(args.io_state, map_location="cpu", weights_only=True)
    state = torch.load(args.ffn_state, map_location="cpu", weights_only=True)
    x = io_state["input"][: args.rows].to(device=device, dtype=torch.bfloat16).contiguous()
    upstream = io_state["upstream"][: args.rows].to(
        device=device, dtype=torch.bfloat16
    ).contiguous()
    gamma = state["norm_weight"].to(device=device, dtype=torch.bfloat16).contiguous()
    w1 = state["w1_weight"].to(device=device, dtype=torch.bfloat16).contiguous()
    w3 = state["w3_weight"].to(device=device, dtype=torch.bfloat16).contiguous()
    w2 = state["w2_weight"].to(device=device, dtype=torch.bfloat16).contiguous()

    base = x.float()
    base.mul_(torch.rsqrt(base.square().mean(dim=1, keepdim=True) + 1.0e-5))
    normalized = (base * gamma.float()).to(torch.bfloat16)
    h1 = F.linear(normalized, w1)
    h3 = F.linear(normalized, w3)
    dh = torch.mm(upstream, w2)
    sigmoid = torch.sigmoid(h1.float())
    dh3 = (dh.float() * h1.float() * sigmoid).to(torch.bfloat16)
    dh1 = (
        dh.float()
        * h3.float()
        * sigmoid
        * (1.0 + h1.float() * (1.0 - sigmoid))
    ).to(torch.bfloat16)
    del sigmoid, normalized, x, upstream, gamma, w1, w2

    # dgamma.mean() = sum(dh3 * sensitivity), including the 1/K mean.
    sensitivity = (torch.mm(base.to(torch.bfloat16), w3.t()) / base.size(1)).to(
        torch.bfloat16
    )
    exact_contribution = float(
        (dh3.float() * sensitivity.float()).sum(dtype=torch.float64)
    )
    return dh, h3, h1, dh1, dh3, sensitivity, exact_contribution


@torch.no_grad()
def _run(args: argparse.Namespace) -> dict:
    os.environ["USE_TK_LOCALCTA_SCALE_NUM"] = str(args.scale_num)
    os.environ["USE_TK_LOCALCTA_V4_FAST_DATA_SR"] = args.fast_data_sr
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    quant = _load_extension(args.runtime_root / QUANT_RELATIVE_PATH)
    quant.tk_localcta_set_global_scale_num(args.scale_num)
    dh, h3, h1, dh1, dh3, sensitivity, exact_contribution = _prepare_reference(
        args, device
    )
    rows, hidden = dh.shape
    dh1_out = torch.empty_like(dh)
    dh3_out = torch.empty_like(dh)
    buffers = quant.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
        rows, hidden, device
    )
    launch = quant.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace

    edges = torch.tensor(BIN_EDGES, device=device, dtype=torch.float32)
    bin_count = torch.zeros(len(BIN_EDGES) - 1, dtype=torch.float64)
    bin_exact = torch.zeros_like(bin_count)
    bin_error = torch.zeros_like(bin_count)
    bin_abs_exact = torch.zeros_like(bin_count)
    bin_zero = torch.zeros_like(bin_count)
    bin_value_error_dot = torch.zeros_like(bin_count)
    bin_value_sq = torch.zeros_like(bin_count)
    sample_relative_gamma_error: list[float] = []
    sample_zero_fraction: list[float] = []
    first_metrics = None
    decode_ceiling_report = None
    tile_amax_report = None
    cutoff_block_count = 0
    cutoff_block_total = 0
    cutoff_element_count = 0
    cutoff_abs_signal = 0.0
    cutoff_total_abs_signal = 0.0

    # The block-relative bins are invariant across stochastic samples.
    block_bin_indices: list[torch.Tensor] = []
    for start in range(0, rows, args.chunk_rows):
        exact = dh3[start : start + args.chunk_rows].float()
        block_amax = exact.abs().reshape(exact.size(0), hidden // 16, 16).amax(dim=2)
        cutoff_blocks = block_amax <= 1.0e-9
        cutoff_elements = cutoff_blocks.repeat_interleave(16, dim=1)
        weighted_abs = (exact * sensitivity[start : start + args.chunk_rows].float()).abs()
        cutoff_block_count += int(cutoff_blocks.sum())
        cutoff_block_total += cutoff_blocks.numel()
        cutoff_element_count += int(cutoff_elements.sum())
        cutoff_abs_signal += float(weighted_abs[cutoff_elements].sum(dtype=torch.float64))
        cutoff_total_abs_signal += float(weighted_abs.sum(dtype=torch.float64))
        relative = exact.abs() / block_amax.repeat_interleave(16, dim=1).clamp_min(1.0e-30)
        block_bin_indices.append(
            torch.bucketize(relative, edges[1:-1], right=False).to(torch.uint8)
        )

    for sample in range(args.samples):
        subsequence = args.subsequence_base + sample * args.subsequence_stride
        result = launch(
            dh,
            h3,
            h1,
            dh1_out,
            dh3_out,
            *buffers,
            False,
            True,
            False,
            args.seed,
            subsequence,
            "row",
        )
        q3 = result[6:12]
        reconstructed = quant.tk_localcta_reconstruct_row(q3[0], q3[1], q3[4])
        torch.cuda.synchronize(device)
        if first_metrics is None:
            first_metrics = _tensor_metrics(reconstructed, dh3)
            tile_ratio_parts = []
            tile_ratio_max = 0.0
            row_sg_grid = q3[4].reshape(rows // 128, hidden // 128).float()
            for tile_row in range(rows // 128):
                actual_tile_amax = (
                    dh3[tile_row * 128 : (tile_row + 1) * 128]
                    .abs()
                    .float()
                    .reshape(128, hidden // 128, 128)
                    .amax(dim=(0, 2))
                )
                reported_tile_amax = row_sg_grid[tile_row] * args.scale_num
                tile_ratios = actual_tile_amax / reported_tile_amax.clamp_min(1.0e-30)
                tile_ratio_max = max(tile_ratio_max, float(tile_ratios.max()))
                tile_ratio_parts.append(tile_ratios.cpu())
            tile_ratios = torch.cat(tile_ratio_parts)
            tile_amax_report = {
                "actual_to_reported_p01": float(torch.quantile(tile_ratios, 0.01)),
                "actual_to_reported_p50": float(torch.quantile(tile_ratios, 0.50)),
                "actual_to_reported_p99": float(torch.quantile(tile_ratios, 0.99)),
                "actual_to_reported_max": tile_ratio_max,
                "underreported_fraction": float((tile_ratios > 1.0).float().mean()),
            }
            del row_sg_grid, tile_ratio_parts, tile_ratios
            max_payload = torch.empty_like(q3[0])
            max_payload.view(torch.uint8).fill_(0x77)
            decode_ceiling = quant.tk_localcta_reconstruct_row(
                max_payload, q3[1], q3[4]
            ).abs()
            ratio_parts = []
            exceeded = 0
            actual_exceeded = 0
            exceeded_abs_signal = 0.0
            exceeded_cutoff = 0
            exceeded_cutoff_abs_signal = 0.0
            total_abs_signal = 0.0
            clipping_gamma_error = 0.0
            ceiling_ratio_max = 0.0
            for ceiling_start in range(0, rows, args.chunk_rows):
                exact_chunk = dh3[
                    ceiling_start : ceiling_start + args.chunk_rows
                ].float()
                sensitivity_chunk = sensitivity[
                    ceiling_start : ceiling_start + args.chunk_rows
                ].float()
                ceiling_chunk = decode_ceiling[
                    ceiling_start : ceiling_start + args.chunk_rows
                ].float()
                valid = exact_chunk != 0
                chunk_ratios = (
                    exact_chunk.abs()[valid]
                    / ceiling_chunk[valid].clamp_min(1.0e-30)
                )
                ceiling_ratio_max = max(
                    ceiling_ratio_max, float(chunk_ratios.max())
                )
                ratio_parts.append(chunk_ratios[::64].cpu())
                over = exact_chunk.abs() > ceiling_chunk
                cutoff = (
                    exact_chunk.abs()
                    .reshape(exact_chunk.size(0), hidden // 16, 16)
                    .amax(dim=2)
                    .le(1.0e-9)
                    .repeat_interleave(16, dim=1)
                )
                exceeded += int(over.sum())
                exceeded_cutoff += int((over & cutoff).sum())
                actual_exceeded += int(
                    (
                        reconstructed[
                            ceiling_start : ceiling_start + args.chunk_rows
                        ].abs()
                        > decode_ceiling[
                            ceiling_start : ceiling_start + args.chunk_rows
                        ].abs()
                    ).sum()
                )
                weighted = exact_chunk * sensitivity_chunk
                exceeded_abs_signal += float(weighted[over].abs().sum(dtype=torch.float64))
                exceeded_cutoff_abs_signal += float(
                    weighted[over & cutoff].abs().sum(dtype=torch.float64)
                )
                total_abs_signal += float(weighted.abs().sum(dtype=torch.float64))
                clipped = exact_chunk.sign() * ceiling_chunk
                clipping_gamma_error += float(
                    ((clipped - exact_chunk) * sensitivity_chunk * over).sum(
                        dtype=torch.float64
                    )
                )
            ratios = torch.cat(ratio_parts)
            decode_ceiling_report = {
                "exceeded_fraction": exceeded / dh3.numel(),
                "actual_exceeded_ceiling_fraction": actual_exceeded / dh3.numel(),
                "exceeded_fraction_of_absolute_gamma_signal": exceeded_abs_signal
                / max(total_abs_signal, 1.0e-300),
                "exceeded_from_1e_9_cutoff_fraction": exceeded_cutoff
                / max(exceeded, 1),
                "exceeded_signal_from_1e_9_cutoff_fraction": exceeded_cutoff_abs_signal
                / max(exceeded_abs_signal, 1.0e-300),
                "clipping_gamma_relative_error": clipping_gamma_error
                / exact_contribution,
                "exact_to_ceiling_ratio_p50": float(torch.quantile(ratios, 0.50)),
                "exact_to_ceiling_ratio_p99": float(torch.quantile(ratios, 0.99)),
                "exact_to_ceiling_ratio_p999": float(torch.quantile(ratios, 0.999)),
                "exact_to_ceiling_ratio_max": ceiling_ratio_max,
            }
            del max_payload, decode_ceiling, ratio_parts, ratios

        weighted_error = 0.0
        zeros = 0
        for chunk_index, start in enumerate(range(0, rows, args.chunk_rows)):
            exact = dh3[start : start + args.chunk_rows].float()
            actual = reconstructed[start : start + args.chunk_rows].float()
            sens = sensitivity[start : start + args.chunk_rows].float()
            indices = block_bin_indices[chunk_index].long()
            exact_weighted = exact * sens
            error_weighted = (actual - exact) * sens
            weighted_error += float(error_weighted.sum(dtype=torch.float64))
            zeros += int((actual == 0).sum())
            flat_indices = indices.reshape(-1)
            bin_count.add_(
                torch.bincount(flat_indices, minlength=bin_count.numel()).cpu()
            )
            bin_zero.add_(
                torch.bincount(
                    flat_indices,
                    weights=(actual == 0).reshape(-1).double(),
                    minlength=bin_count.numel(),
                ).cpu()
            )
            bin_exact.add_(
                torch.bincount(
                    flat_indices,
                    weights=exact_weighted.reshape(-1).double(),
                    minlength=bin_count.numel(),
                ).cpu()
            )
            bin_error.add_(
                torch.bincount(
                    flat_indices,
                    weights=error_weighted.reshape(-1).double(),
                    minlength=bin_count.numel(),
                ).cpu()
            )
            bin_abs_exact.add_(
                torch.bincount(
                    flat_indices,
                    weights=exact_weighted.abs().reshape(-1).double(),
                    minlength=bin_count.numel(),
                ).cpu()
            )
            bin_value_error_dot.add_(
                torch.bincount(
                    flat_indices,
                    weights=((actual - exact) * exact).reshape(-1).double(),
                    minlength=bin_count.numel(),
                ).cpu()
            )
            bin_value_sq.add_(
                torch.bincount(
                    flat_indices,
                    weights=exact.square().reshape(-1).double(),
                    minlength=bin_count.numel(),
                ).cpu()
            )
        sample_relative_gamma_error.append(weighted_error / exact_contribution)
        sample_zero_fraction.append(zeros / dh3.numel())
        del reconstructed, result, q3

    bin_count.div_(args.samples)
    bin_zero.div_(args.samples)
    bin_exact.div_(args.samples)
    bin_error.div_(args.samples)
    bin_abs_exact.div_(args.samples)
    bin_value_error_dot.div_(args.samples)
    bin_value_sq.div_(args.samples)
    bins = []
    for index, (low, high) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        count = float(bin_count[index])
        exact = float(bin_exact[index])
        error = float(bin_error[index])
        bins.append(
            {
                "low": low,
                "high": high,
                "fraction": count / dh3.numel(),
                "mean_zero_probability": float(bin_zero[index]) / max(count, 1.0),
                "exact_gamma_mean_contribution": exact,
                "gamma_error_contribution": error,
                "relative_error_within_bin": error / exact if exact != 0.0 else None,
                "fraction_of_absolute_gamma_signal": float(bin_abs_exact[index])
                / max(float(bin_abs_exact.sum()), 1.0e-300),
                "value_least_squares_gain_error": float(bin_value_error_dot[index])
                / max(float(bin_value_sq[index]), 1.0e-300),
            }
        )

    errors = torch.tensor(sample_relative_gamma_error, dtype=torch.float64)
    fused_dh3_gamma_error = float(
        ((dh3_out.float() - dh3.float()) * sensitivity.float()).sum(
            dtype=torch.float64
        )
    ) / exact_contribution
    report = {
        "runtime_root": str(args.runtime_root),
        "fast_data_sr": args.fast_data_sr == "1",
        "scale_num": float(quant.tk_localcta_get_global_scale_num()),
        "shape": {"rows": rows, "hidden": hidden},
        "samples": args.samples,
        "dh1": _tensor_metrics(dh1_out, dh1),
        "dh3_bf16": _tensor_metrics(dh3_out, dh3),
        "dh3_bf16_gamma_mean_relative_error": fused_dh3_gamma_error,
        "dh3_first_sample": first_metrics,
        "decode_ceiling": decode_ceiling_report,
        "tile_amax_contract": tile_amax_report,
        "absolute_1e_9_cutoff": {
            "block_fraction": cutoff_block_count / cutoff_block_total,
            "element_fraction": cutoff_element_count / dh3.numel(),
            "fraction_of_absolute_gamma_signal": cutoff_abs_signal
            / max(cutoff_total_abs_signal, 1.0e-300),
        },
        "dh3_reference": {
            "rms": float(dh3.float().square().mean().sqrt()),
            "mean_abs": float(dh3.float().abs().mean()),
            "max_abs": float(dh3.float().abs().max()),
            "zero_fraction": float((dh3 == 0).float().mean()),
        },
        "gamma_mean_exact": exact_contribution,
        "gamma_mean_relative_error": {
            "mean": float(errors.mean()),
            "std": float(errors.std(unbiased=False)),
            "min": float(errors.min()),
            "max": float(errors.max()),
        },
        "sample_zero_fraction": {
            "mean": float(torch.tensor(sample_zero_fraction).mean()),
            "min": min(sample_zero_fraction),
            "max": max(sample_zero_fraction),
        },
        "bins": bins,
    }
    return report


def main() -> None:
    args = _parse_args()
    report = _run(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
