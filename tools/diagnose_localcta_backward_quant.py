#!/usr/bin/env python3
"""Isolate localCTA forward, dgrad, and wgrad quantization error.

The same BF16 X, W, and dY tensors feed every path.  This separates errors in
the localCTA producer/consumer contracts from optimizer, collectives, RMSNorm,
and the rest of the model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


DEFAULT_FP4_ROOT = Path("/opt/mfu/EXTERNAL_PATH")
QUANT_RELATIVE_PATH = Path(
    "TK_quantisation/nvfp4_CTA_local_v4/"
    "_tk_quant_localcta_v4.cpython-312-aarch64-linux-gnu.so"
)
GEMM_RELATIVE_PATH = Path(
    "ThunderKittens/kernels/gemm/nvfp4_b200/localCTA_epilogue_v3/"
    "_C_nv_localcta_gemm_v3.cpython-312-aarch64-linux-gnu.so"
)


def _load_extension(name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_quantize(quant, value: torch.Tensor):
    return quant.tk_localcta_quantize_for_gemm_fast(value, True, True)


def _final_sg_quantize(
    quant,
    value: torch.Tensor,
    data_sr_axes: str = "none",
    rng_seed: int = 0,
    rng_subsequence_base: int = 0,
):
    args = (
        value,
        True,
        True,
        data_sr_axes != "none",
        False,
        "none",
        False,
        rng_seed,
        rng_subsequence_base,
        False,
    )
    try:
        return quant.tk_localcta_quantize_for_gemm_final_sg_opt(
            *args, data_sr_axes
        )
    except TypeError:
        if data_sr_axes != "none":
            raise RuntimeError(
                "localCTA extension does not support axis-selective final-SG SR; "
                "load the patched quantizer with --quant-so"
            ) from None
        return quant.tk_localcta_quantize_for_gemm_final_sg_opt(*args)


def _opt_quantize(
    quant,
    value: torch.Tensor,
    data_sr_axes: str = "none",
    rng_seed: int = 0,
    rng_subsequence_base: int = 0,
):
    return quant.tk_localcta_quantize_for_gemm_opt(
        value,
        True,
        True,
        data_sr_axes != "none",
        False,
        "none",
        False,
        rng_seed,
        rng_subsequence_base,
        data_sr_axes,
    )


def _layout(q, axis: str, prepared: bool = False):
    if axis == "row":
        return q[0], q[6] if prepared else q[1], None if prepared else q[4]
    if axis == "col":
        return q[2], q[7] if prepared else q[3], None if prepared else q[5]
    raise ValueError(axis)


@torch.no_grad()
def _run_gemm(gemm, a, a_axis: str, b, b_axis: str, contract: str):
    prepared = contract == "prepared"
    a_fp4, a_sc, a_sg = _layout(a, a_axis, prepared)
    b_fp4, b_sc, b_sg = _layout(b, b_axis, prepared)
    out = torch.empty(
        (a_fp4.size(0), b_fp4.size(0)),
        dtype=torch.bfloat16,
        device=a_fp4.device,
    )
    if prepared:
        gemm.nvfp4_localcta_fast_gemm(a_fp4, a_sc, b_fp4, b_sc, out)
    else:
        gemm.nvfp4_localcta_gemm(
            a_fp4, a_sc, a_sg, b_fp4, b_sc, b_sg, out
        )
    torch.cuda.synchronize(a_fp4.device)
    return out


@torch.no_grad()
def _run_adapter_gemm(
    gemm,
    a,
    a_axis: str,
    b,
    b_axis: str,
    op: str,
    factor: float,
):
    a_fp4, a_sc, a_sg = _layout(a, a_axis)
    b_fp4, b_sc, b_sg = _layout(b, b_axis)
    a_tiles = a_fp4.size(0) // 256
    b_tiles = b_fp4.size(0) // 256
    if op == "dx":
        a_outer = gemm.nvfp4_localcta_prepare_outer_sg(a_sg, a_tiles, True)
        b_outer = gemm.nvfp4_localcta_prepare_w2_dgrad_b_sg(
            b_sg, b_tiles, factor
        )
    elif op == "dw":
        a_outer = gemm.nvfp4_localcta_prepare_split_wgrad_a_sg(
            a_sg, a_tiles, factor
        )
        b_outer = gemm.nvfp4_localcta_prepare_outer_sg(b_sg, b_tiles, False)
    else:
        raise ValueError(op)
    out = torch.empty(
        (a_fp4.size(0), b_fp4.size(0)),
        dtype=torch.bfloat16,
        device=a_fp4.device,
    )
    gemm.nvfp4_localcta_gemm(
        a_fp4, a_sc, a_outer, b_fp4, b_sc, b_outer, out
    )
    torch.cuda.synchronize(a_fp4.device)
    return out


@torch.no_grad()
def _error_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    diff_sq = 0.0
    ref_sq = 0.0
    value_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    zero_count = 0
    finite_count = 0
    row_ratios: list[torch.Tensor] = []
    for start in range(0, value.size(0), 512):
        actual = value[start : start + 512].float()
        target = reference[start : start + 512].float()
        diff = actual - target
        diff_sq += float(diff.square().sum().item())
        ref_sq += float(target.square().sum().item())
        value_sq += float(actual.square().sum().item())
        dot += float((actual * target).sum().item())
        max_abs = max(max_abs, float(diff.abs().max().item()))
        zero_count += int((actual == 0).sum().item())
        finite_count += int(torch.isfinite(actual).sum().item())
        row_ratios.append(
            (
                actual.square().mean(dim=1).sqrt()
                / target.square().mean(dim=1).sqrt().clamp_min(1.0e-30)
            ).cpu()
        )
    gain = dot / max(ref_sq, 1.0e-30)
    gain_corrected_sq = max(value_sq - 2.0 * gain * dot + gain * gain * ref_sq, 0.0)
    ratios = torch.cat(row_ratios)
    return {
        "relative_l2": math.sqrt(diff_sq / max(ref_sq, 1.0e-30)),
        "cosine": dot / max(math.sqrt(ref_sq * value_sq), 1.0e-30),
        "rms_ratio": math.sqrt(value_sq / max(ref_sq, 1.0e-30)),
        "least_squares_gain": gain,
        "gain_corrected_relative_l2": math.sqrt(
            gain_corrected_sq / max(ref_sq, 1.0e-30)
        ),
        "row_rms_ratio_p01": float(torch.quantile(ratios, 0.01).item()),
        "row_rms_ratio_p50": float(torch.quantile(ratios, 0.50).item()),
        "row_rms_ratio_p99": float(torch.quantile(ratios, 0.99).item()),
        "zero_fraction": zero_count / value.numel(),
        "finite_fraction": finite_count / value.numel(),
        "max_abs_error": max_abs,
    }


def _fp4_zero_fraction(value: torch.Tensor) -> float:
    packed = value.view(torch.uint8)
    low = packed & 0x0F
    high = packed >> 4
    zeros = ((low == 0) | (low == 8)).sum() + ((high == 0) | (high == 8)).sum()
    return float(zeros.item()) / (2 * packed.numel())


def _quant_stats(q) -> dict[str, float]:
    stats: dict[str, float] = {}
    for axis, fp4_idx, scale_idx, sg_idx in (
        ("row", 0, 1, 4),
        ("col", 2, 3, 5),
    ):
        fp4 = q[fp4_idx]
        scale = q[scale_idx].float()
        sg = q[sg_idx].float().reshape(-1)
        stats[f"{axis}_fp4_zero_fraction"] = _fp4_zero_fraction(fp4)
        stats[f"{axis}_scale_saturation_fraction"] = float(
            (scale.abs() == 448.0).float().mean().item()
        )
        stats[f"{axis}_sg_min"] = float(sg.min().item())
        stats[f"{axis}_sg_p50"] = float(torch.quantile(sg, 0.50).item())
        stats[f"{axis}_sg_max"] = float(sg.max().item())
    return stats


@torch.no_grad()
def _reconstruct_outer_quant(quant, q, rows: int, cols: int):
    """Reconstruct row/column payloads after expanding the outer-SG contract."""
    row_sg = (
        q[4]
        .reshape(-1)
        .repeat_interleave(2)
        .reshape(rows // 128, 1)
        .expand(rows // 128, cols // 128)
        .contiguous()
    )
    col_sg = (
        q[5]
        .reshape(-1)
        .repeat_interleave(2)
        .reshape(cols // 128, 1)
        .expand(cols // 128, rows // 128)
        .contiguous()
    )
    row = quant.tk_localcta_reconstruct_row(q[0], q[1], row_sg)
    col = quant.tk_localcta_reconstruct_col(q[2], q[3], col_sg).t().contiguous()
    torch.cuda.synchronize(q[0].device)
    return row, col


@torch.no_grad()
def _make_inputs(
    m: int,
    n: int,
    k: int,
    case: str,
    device: torch.device,
    seed: int,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=generator, device=device, dtype=torch.bfloat16)
    w = torch.randn((n, k), generator=generator, device=device, dtype=torch.bfloat16)
    w.mul_(1.0 / math.sqrt(k))
    dy = torch.randn((m, n), generator=generator, device=device, dtype=torch.bfloat16)
    dy.mul_(1.0 / math.sqrt(n))
    if case == "tile-outlier":
        x[:128, 128:256].mul_(32.0)
        w[128:256, 256:384].mul_(32.0)
        dy[256:384, :128].mul_(32.0)
        if m >= 1024 and n >= 512 and k >= 512:
            x[896:1024, 384:512].mul_(8.0)
            w[384:512, :128].mul_(16.0)
            dy[896:1024, 384:512].mul_(16.0)
    elif case != "iid":
        raise ValueError(case)
    return x, w, dy


@torch.no_grad()
def _run_case(args, quant, gemm, case: str) -> dict[str, Any]:
    device = torch.device("cuda", args.device)
    x, w, dy = _make_inputs(args.m, args.n, args.k, case, device, args.seed)
    references = {
        "forward": torch.mm(x, w.t()),
        "dx": torch.mm(dy, w),
        "dw": torch.mm(dy.t(), x),
    }

    raw = {name: _raw_quantize(quant, value) for name, value in (("x", x), ("w", w), ("dy", dy))}
    final = {name: _final_sg_quantize(quant, value) for name, value in (("x", x), ("w", w), ("dy", dy))}
    torch.cuda.synchronize(device)

    operands = {
        "forward": ("x", "row", "w", "row"),
        "dx": ("dy", "row", "w", "col"),
        "dw": ("dy", "col", "x", "col"),
    }
    results: dict[str, dict[str, float]] = {}
    for op, (a_name, a_axis, b_name, b_axis) in operands.items():
        for contract, source in (
            ("chunk_grid", raw),
            ("final_sg", final),
            ("prepared", raw),
        ):
            output = _run_gemm(
                gemm,
                source[a_name],
                a_axis,
                source[b_name],
                b_axis,
                "prepared" if contract == "prepared" else contract,
            )
            results[f"{op}.{contract}"] = _error_metrics(output, references[op])
            del output

    for op, factor in (("dx", 1.0), ("dx", 0.731), ("dw", 1.0), ("dw", 0.709)):
        a_name, a_axis, b_name, b_axis = operands[op]
        output = _run_adapter_gemm(
            gemm,
            raw[a_name],
            a_axis,
            raw[b_name],
            b_axis,
            op,
            factor,
        )
        results[f"{op}.chunk_to_outer_{factor:g}"] = _error_metrics(
            output, references[op]
        )
        del output

    grad_sr_axis_contract: list[dict[str, Any]] = []
    if args.grad_sr_axes != "none":
        grad_det = _opt_quantize(quant, dy)
        grad_det_row_bf16, grad_det_col_bf16 = _reconstruct_outer_quant(
            quant, grad_det, args.m, args.n
        )
        results["dy.grad_opt_row_reconstruct"] = _error_metrics(
            grad_det_row_bf16, dy
        )
        results["dy.grad_opt_col_reconstruct"] = _error_metrics(
            grad_det_col_bf16, dy
        )
        det_dx = _run_gemm(gemm, grad_det, "row", raw["w"], "col", "chunk_grid")
        det_dw = _run_gemm(gemm, grad_det, "col", raw["x"], "col", "chunk_grid")
        results["dx.grad_opt_mixed_contract"] = _error_metrics(
            det_dx, references["dx"]
        )
        results["dw.grad_opt_mixed_contract"] = _error_metrics(
            det_dw, references["dw"]
        )
        matched_det_dx = _run_gemm(
            gemm, grad_det, "row", final["w"], "col", "final_sg"
        )
        matched_det_dw = _run_gemm(
            gemm, grad_det, "col", final["x"], "col", "final_sg"
        )
        results["dx.grad_opt_outer_matched"] = _error_metrics(
            matched_det_dx, references["dx"]
        )
        results["dw.grad_opt_outer_matched"] = _error_metrics(
            matched_det_dw, references["dw"]
        )
        grad_final_det = _final_sg_quantize(quant, dy)
        final_det_dx = _run_gemm(
            gemm, grad_final_det, "row", final["w"], "col", "final_sg"
        )
        final_det_dw = _run_gemm(
            gemm, grad_final_det, "col", final["x"], "col", "final_sg"
        )
        results["dx.grad_final_sg_deterministic"] = _error_metrics(
            final_det_dx, references["dx"]
        )
        results["dw.grad_final_sg_deterministic"] = _error_metrics(
            final_det_dw, references["dw"]
        )
        sr_dx_sum = torch.zeros_like(det_dx, dtype=torch.float32)
        sr_dw_sum = torch.zeros_like(det_dw, dtype=torch.float32)
        matched_sr_dx_sum = torch.zeros_like(matched_det_dx, dtype=torch.float32)
        matched_sr_dw_sum = torch.zeros_like(matched_det_dw, dtype=torch.float32)
        final_sr_dx_sum = torch.zeros_like(final_det_dx, dtype=torch.float32)
        final_sr_dw_sum = torch.zeros_like(final_det_dw, dtype=torch.float32)
        for sample in range(args.sr_samples):
            subsequence = (
                args.sr_subsequence_base + sample * args.sr_subsequence_stride
            )
            grad_sr = _opt_quantize(
                quant,
                dy,
                args.grad_sr_axes,
                args.sr_seed,
                subsequence,
            )
            prefix = f"grad_sr_{args.grad_sr_axes}.sample{sample}"
            grad_sr_row_bf16, grad_sr_col_bf16 = _reconstruct_outer_quant(
                quant, grad_sr, args.m, args.n
            )
            results[f"dy.row_{prefix}"] = _error_metrics(
                grad_sr_row_bf16, dy
            )
            results[f"dy.col_{prefix}"] = _error_metrics(
                grad_sr_col_bf16, dy
            )
            sr_dx = _run_gemm(gemm, grad_sr, "row", raw["w"], "col", "chunk_grid")
            sr_dw = _run_gemm(gemm, grad_sr, "col", raw["x"], "col", "chunk_grid")
            results[f"dx.mixed_contract_{prefix}"] = _error_metrics(
                sr_dx, references["dx"]
            )
            results[f"dw.mixed_contract_{prefix}"] = _error_metrics(
                sr_dw, references["dw"]
            )
            matched_sr_dx = _run_gemm(
                gemm, grad_sr, "row", final["w"], "col", "final_sg"
            )
            matched_sr_dw = _run_gemm(
                gemm, grad_sr, "col", final["x"], "col", "final_sg"
            )
            results[f"dx.outer_matched_{prefix}"] = _error_metrics(
                matched_sr_dx, references["dx"]
            )
            results[f"dw.outer_matched_{prefix}"] = _error_metrics(
                matched_sr_dw, references["dw"]
            )
            grad_final_sr = _final_sg_quantize(
                quant,
                dy,
                args.grad_sr_axes,
                args.sr_seed,
                subsequence,
            )
            final_sr_dx = _run_gemm(
                gemm, grad_final_sr, "row", final["w"], "col", "final_sg"
            )
            final_sr_dw = _run_gemm(
                gemm, grad_final_sr, "col", final["x"], "col", "final_sg"
            )
            results[f"dx.final_sg_{prefix}"] = _error_metrics(
                final_sr_dx, references["dx"]
            )
            results[f"dw.final_sg_{prefix}"] = _error_metrics(
                final_sr_dw, references["dw"]
            )
            grad_sr_axis_contract.append(
                {
                    "sample": sample,
                    "rng_subsequence_base": subsequence,
                    "generic_row": _quant_axis_delta(grad_sr, grad_det, "row"),
                    "generic_col": _quant_axis_delta(grad_sr, grad_det, "col"),
                    "final_sg_row": _quant_axis_delta(
                        grad_final_sr, grad_final_det, "row"
                    ),
                    "final_sg_col": _quant_axis_delta(
                        grad_final_sr, grad_final_det, "col"
                    ),
                }
            )
            sr_dx_sum.add_(sr_dx)
            sr_dw_sum.add_(sr_dw)
            matched_sr_dx_sum.add_(matched_sr_dx)
            matched_sr_dw_sum.add_(matched_sr_dw)
            final_sr_dx_sum.add_(final_sr_dx)
            final_sr_dw_sum.add_(final_sr_dw)
            del (
                grad_sr,
                sr_dx,
                sr_dw,
                matched_sr_dx,
                matched_sr_dw,
                grad_sr_row_bf16,
                grad_sr_col_bf16,
                grad_final_sr,
                final_sr_dx,
                final_sr_dw,
            )
        sr_dx_sum.div_(args.sr_samples)
        sr_dw_sum.div_(args.sr_samples)
        results[
            f"dx.mixed_contract_grad_sr_{args.grad_sr_axes}.ensemble_mean"
        ] = _error_metrics(sr_dx_sum, references["dx"])
        results[
            f"dw.mixed_contract_grad_sr_{args.grad_sr_axes}.ensemble_mean"
        ] = _error_metrics(sr_dw_sum, references["dw"])
        matched_sr_dx_sum.div_(args.sr_samples)
        matched_sr_dw_sum.div_(args.sr_samples)
        results[
            f"dx.outer_matched_grad_sr_{args.grad_sr_axes}.ensemble_mean"
        ] = _error_metrics(matched_sr_dx_sum, references["dx"])
        results[
            f"dw.outer_matched_grad_sr_{args.grad_sr_axes}.ensemble_mean"
        ] = _error_metrics(matched_sr_dw_sum, references["dw"])
        final_sr_dx_sum.div_(args.sr_samples)
        final_sr_dw_sum.div_(args.sr_samples)
        results[
            f"dx.final_sg_grad_sr_{args.grad_sr_axes}.ensemble_mean"
        ] = _error_metrics(final_sr_dx_sum, references["dx"])
        results[
            f"dw.final_sg_grad_sr_{args.grad_sr_axes}.ensemble_mean"
        ] = _error_metrics(final_sr_dw_sum, references["dw"])
        del (
            grad_det,
            grad_det_row_bf16,
            grad_det_col_bf16,
            det_dx,
            det_dw,
            matched_det_dx,
            matched_det_dw,
            grad_final_det,
            final_det_dx,
            final_det_dw,
            sr_dx_sum,
            sr_dw_sum,
            matched_sr_dx_sum,
            matched_sr_dw_sum,
            final_sr_dx_sum,
            final_sr_dw_sum,
        )

    report = {
        "case": case,
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "raw_quant_stats": {name: _quant_stats(value) for name, value in raw.items()},
        "final_sg_quant_stats": {
            name: _quant_stats(value) for name, value in final.items()
        },
        "grad_sr_axis_contract": grad_sr_axis_contract,
        "results": results,
    }
    del references, raw, final, x, w, dy
    torch.cuda.empty_cache()
    return report


def _split_fused_result(result):
    return tuple(result[:6]), tuple(result[6:12])


@torch.no_grad()
def _fused_swiglu_derivative_quantize(
    quant,
    dh: torch.Tensor,
    h3: torch.Tensor,
    h1: torch.Tensor,
    finalize_contract: bool,
    data_sr_axes: str = "none",
    rng_seed: int = 0,
    rng_subsequence_base: int = 0,
):
    dh1 = torch.empty_like(dh)
    dh3 = torch.empty_like(dh)
    buffers = quant.tk_localcta_silu_deriv_quantize_split_for_gemm_alloc(
        dh.size(0), dh.size(1), dh.device
    )
    launch = quant.tk_localcta_silu_deriv_quantize_split_for_gemm_launch_inplace
    args = (
        dh,
        h3,
        h1,
        dh1,
        dh3,
        *buffers,
        finalize_contract,
        data_sr_axes != "none",
        False,
        rng_seed,
        rng_subsequence_base,
    )
    try:
        result = launch(*args, data_sr_axes)
    except TypeError:
        if data_sr_axes != "none":
            raise RuntimeError(
                "localCTA extension does not support axis-selective data SR; "
                "load the production quantizer with --quant-so"
            ) from None
        result = launch(*args)
    torch.cuda.synchronize(dh.device)
    q1, q3 = _split_fused_result(result)
    return dh1, dh3, q1, q3


def _quant_axis_delta(actual, expected, axis: str) -> dict[str, float]:
    indices = (0, 1, 4) if axis == "row" else (2, 3, 5)
    names = ("fp4", "scale", "sg")
    result: dict[str, float] = {}
    for name, index in zip(names, indices):
        lhs = actual[index]
        rhs = expected[index]
        if name in {"fp4", "scale"}:
            lhs = lhs.contiguous().view(torch.uint8)
            rhs = rhs.contiguous().view(torch.uint8)
        result[f"{name}_mismatch_fraction"] = float(
            (lhs != rhs).float().mean().item()
        )
    return result


@torch.no_grad()
def _run_swiglu_case(args, quant, gemm) -> dict[str, Any]:
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    m, h, k = args.m, args.n, args.k
    x = torch.randn((m, k), generator=generator, device=device, dtype=torch.bfloat16)
    h1 = torch.randn((m, h), generator=generator, device=device, dtype=torch.bfloat16)
    h3 = torch.randn((m, h), generator=generator, device=device, dtype=torch.bfloat16)
    dh = torch.randn((m, h), generator=generator, device=device, dtype=torch.bfloat16)
    dh.mul_(1.0 / math.sqrt(h))
    w1 = torch.randn((h, k), generator=generator, device=device, dtype=torch.bfloat16)
    w3 = torch.randn((h, k), generator=generator, device=device, dtype=torch.bfloat16)
    w1.mul_(1.0 / math.sqrt(k))
    w3.mul_(1.0 / math.sqrt(k))

    h1_f32 = h1.float()
    sigmoid = torch.sigmoid(h1_f32)
    dh1_ref = (
        dh.float()
        * h3.float()
        * sigmoid
        * (1.0 + h1_f32 * (1.0 - sigmoid))
    ).to(torch.bfloat16)
    dh3_ref = (dh.float() * h1_f32 * sigmoid).to(torch.bfloat16)
    del h1_f32, sigmoid

    fused_dh1, fused_dh3, fused_q1, fused_q3 = _fused_swiglu_derivative_quantize(
        quant, dh, h3, h1, True
    )
    generic_q1 = _final_sg_quantize(quant, dh1_ref)
    generic_q3 = _final_sg_quantize(quant, dh3_ref)
    x_q = _final_sg_quantize(quant, x)
    w1_q = _final_sg_quantize(quant, w1)
    w3_q = _final_sg_quantize(quant, w3)

    dx_ref = torch.mm(dh1_ref, w1) + torch.mm(dh3_ref, w3)
    dw1_ref = torch.mm(dh1_ref.t(), x)
    dw3_ref = torch.mm(dh3_ref.t(), x)

    results = {
        "derivative.dh1_bf16": _error_metrics(fused_dh1, dh1_ref),
        "derivative.dh3_bf16": _error_metrics(fused_dh3, dh3_ref),
    }
    for label, q1, q3 in (
        ("generic_final_sg", generic_q1, generic_q3),
        ("fused_final_sg", fused_q1, fused_q3),
    ):
        dx1 = _run_gemm(gemm, q1, "row", w1_q, "col", "final_sg")
        dx3 = _run_gemm(gemm, q3, "row", w3_q, "col", "final_sg")
        results[f"dx.{label}"] = _error_metrics(dx1 + dx3, dx_ref)
        del dx1, dx3
        dw1 = _run_gemm(gemm, q1, "col", x_q, "col", "final_sg")
        results[f"dw1.{label}"] = _error_metrics(dw1, dw1_ref)
        del dw1
        dw3 = _run_gemm(gemm, q3, "col", x_q, "col", "final_sg")
        results[f"dw3.{label}"] = _error_metrics(dw3, dw3_ref)
        del dw3

    sr_axis_contract: list[dict[str, Any]] = []
    sr_outputs: dict[str, torch.Tensor] = {}
    if args.grad_sr_axes != "none":
        for sample in range(args.sr_samples):
            subsequence = (
                args.sr_subsequence_base + sample * args.sr_subsequence_stride
            )
            sr_dh1, sr_dh3, sr_q1, sr_q3 = _fused_swiglu_derivative_quantize(
                quant,
                dh,
                h3,
                h1,
                True,
                args.grad_sr_axes,
                args.sr_seed,
                subsequence,
            )
            sr_dx1 = _run_gemm(gemm, sr_q1, "row", w1_q, "col", "final_sg")
            sr_dx3 = _run_gemm(gemm, sr_q3, "row", w3_q, "col", "final_sg")
            sr_dx = sr_dx1 + sr_dx3
            sr_dw1 = _run_gemm(gemm, sr_q1, "col", x_q, "col", "final_sg")
            sr_dw3 = _run_gemm(gemm, sr_q3, "col", x_q, "col", "final_sg")
            prefix = f"fused_sr_{args.grad_sr_axes}.sample{sample}"
            results[f"dx.{prefix}"] = _error_metrics(sr_dx, dx_ref)
            results[f"dw1.{prefix}"] = _error_metrics(sr_dw1, dw1_ref)
            results[f"dw3.{prefix}"] = _error_metrics(sr_dw3, dw3_ref)
            sr_axis_contract.append(
                {
                    "sample": sample,
                    "rng_subsequence_base": subsequence,
                    "dh1_bf16": _error_metrics(sr_dh1, fused_dh1),
                    "dh3_bf16": _error_metrics(sr_dh3, fused_dh3),
                    "dh1_row": _quant_axis_delta(sr_q1, fused_q1, "row"),
                    "dh1_col": _quant_axis_delta(sr_q1, fused_q1, "col"),
                    "dh3_row": _quant_axis_delta(sr_q3, fused_q3, "row"),
                    "dh3_col": _quant_axis_delta(sr_q3, fused_q3, "col"),
                }
            )
            for name, value in (
                ("dx", sr_dx),
                ("dw1", sr_dw1),
                ("dw3", sr_dw3),
            ):
                if name not in sr_outputs:
                    sr_outputs[name] = value.float()
                else:
                    sr_outputs[name].add_(value)
            del sr_dh1, sr_dh3, sr_q1, sr_q3, sr_dx1, sr_dx3, sr_dx, sr_dw1, sr_dw3

        for name, reference in (
            ("dx", dx_ref),
            ("dw1", dw1_ref),
            ("dw3", dw3_ref),
        ):
            sr_outputs[name].div_(args.sr_samples)
            results[f"{name}.fused_sr_{args.grad_sr_axes}.ensemble_mean"] = (
                _error_metrics(sr_outputs[name], reference)
            )

    report = {
        "case": "swiglu-derivative",
        "shape": {"m": m, "n": h, "k": k},
        "fused_quant_stats": {
            "dh1": _quant_stats(fused_q1),
            "dh3": _quant_stats(fused_q3),
        },
        "generic_quant_stats": {
            "dh1": _quant_stats(generic_q1),
            "dh3": _quant_stats(generic_q3),
        },
        "sr_axis_contract": sr_axis_contract,
        "results": results,
    }
    del (
        x,
        h1,
        h3,
        dh,
        w1,
        w3,
        dh1_ref,
        dh3_ref,
        fused_dh1,
        fused_dh3,
        fused_q1,
        fused_q3,
        generic_q1,
        generic_q3,
        x_q,
        w1_q,
        w3_q,
        dx_ref,
        dw1_ref,
        dw3_ref,
    )
    del sr_outputs
    torch.cuda.empty_cache()
    return report


def _print_report(report: dict[str, Any]) -> None:
    shape = report["shape"]
    print(
        f"\ncase={report['case']} shape="
        f"M={shape['m']} N={shape['n']} K={shape['k']}"
    )
    print(
        f"{'operation.contract':<35} {'rel_l2':>10} {'cosine':>10} "
        f"{'gain':>10} {'gain_adj':>10} {'row_p01':>10} {'row_p99':>10}"
    )
    for name, metrics in report["results"].items():
        print(
            f"{name:<35} {metrics['relative_l2']:>10.5f} "
            f"{metrics['cosine']:>10.6f} {metrics['least_squares_gain']:>10.5f} "
            f"{metrics['gain_corrected_relative_l2']:>10.5f} "
            f"{metrics['row_rms_ratio_p01']:>10.5f} "
            f"{metrics['row_rms_ratio_p99']:>10.5f}"
        )
    quant_stats = report.get("raw_quant_stats", report.get("fused_quant_stats", {}))
    quant_label = (
        "raw chunk-grid" if "raw_quant_stats" in report else "fused final-SG"
    )
    print(f"quantized operand zero fractions ({quant_label}):")
    for name, stats in quant_stats.items():
        print(
            f"  {name:<3} row={stats['row_fp4_zero_fraction']:.5f} "
            f"col={stats['col_fp4_zero_fraction']:.5f} "
            f"row_scale_sat={stats['row_scale_saturation_fraction']:.5f} "
            f"col_scale_sat={stats['col_scale_saturation_fraction']:.5f}"
        )
    if report.get("grad_sr_axis_contract"):
        print("Gradient SR axis contract (mismatch versus deterministic payload):")
        for item in report["grad_sr_axis_contract"]:
            print(
                f"  sample={item['sample']} subsequence={item['rng_subsequence_base']} "
                f"generic_row_fp4={item['generic_row']['fp4_mismatch_fraction']:.5f} "
                f"generic_col_fp4={item['generic_col']['fp4_mismatch_fraction']:.5f} "
                f"final_row_fp4={item['final_sg_row']['fp4_mismatch_fraction']:.5f} "
                f"final_col_fp4={item['final_sg_col']['fp4_mismatch_fraction']:.5f}"
            )
    if report.get("sr_axis_contract"):
        print("SR axis contract (mismatch fractions versus deterministic payload):")
        for item in report["sr_axis_contract"]:
            print(
                f"  sample={item['sample']} subsequence={item['rng_subsequence_base']} "
                f"dh1_row_fp4={item['dh1_row']['fp4_mismatch_fraction']:.5f} "
                f"dh1_col_fp4={item['dh1_col']['fp4_mismatch_fraction']:.5f} "
                f"dh3_row_fp4={item['dh3_row']['fp4_mismatch_fraction']:.5f} "
                f"dh3_col_fp4={item['dh3_col']['fp4_mismatch_fraction']:.5f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--case", choices=("iid", "tile-outlier", "both"), default="both")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--scale-num", type=float, default=448.0)
    parser.add_argument(
        "--include-swiglu",
        action="store_true",
        help="Also isolate the fused SwiGLU derivative producer and W1/W3 backward.",
    )
    parser.add_argument(
        "--grad-sr-axes",
        choices=("none", "row", "col", "both"),
        default="none",
        help=(
            "Apply production axis-selective SR to the generic backward producer "
            "and, with --include-swiglu, the fused derivative producer."
        ),
    )
    parser.add_argument("--sr-samples", type=int, default=4)
    parser.add_argument("--sr-seed", type=int, default=0)
    parser.add_argument("--sr-subsequence-base", type=int, default=0)
    parser.add_argument("--sr-subsequence-stride", type=int, default=1_000_000_000)
    parser.add_argument("--fp4-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument("--quant-so", type=Path)
    parser.add_argument("--gemm-so", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    for value, name in ((args.m, "m"), (args.n, "n"), (args.k, "k")):
        if value % 256 != 0:
            parser.error(f"--{name} must be divisible by 256")
    if args.scale_num <= 0 or not math.isfinite(args.scale_num):
        parser.error("--scale-num must be finite and positive")
    if args.sr_samples <= 0:
        parser.error("--sr-samples must be positive")
    if args.sr_subsequence_base < 0 or args.sr_subsequence_stride <= 0:
        parser.error("SR subsequence values must be non-negative with a positive stride")
    torch.cuda.set_device(args.device)
    quant_path = args.quant_so or args.fp4_root / QUANT_RELATIVE_PATH
    gemm_path = args.gemm_so or args.fp4_root / GEMM_RELATIVE_PATH
    quant = _load_extension("_tk_quant_localcta_v4", quant_path)
    gemm = _load_extension("_C_nv_localcta_gemm_v3", gemm_path)
    quant.tk_localcta_set_global_scale_num(float(args.scale_num))

    cases = ("iid", "tile-outlier") if args.case == "both" else (args.case,)
    reports = [_run_case(args, quant, gemm, case) for case in cases]
    if args.include_swiglu:
        reports.append(_run_swiglu_case(args, quant, gemm))
    for report in reports:
        _print_report(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
