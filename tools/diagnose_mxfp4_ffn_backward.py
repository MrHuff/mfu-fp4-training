#!/usr/bin/env python3
"""Diagnose isolated FFN backward for MXFP4 vs localCTA NVFP4.

This is benchmark-oriented and reuses the synthetic 1B model build path.
It reports:
- isolated FFN block total timing for MXFP4 baseline / MXFP4 fused split2 / localCTA / localCTA fused
- MXFP4 manual backward stage timings for baseline vs fused split2
- profiler-derived category sums for all four modes
- FFN numerics sanity for MXFP4 fused split2 vs MXFP4 baseline
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import sys
from typing import Any

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import compare_mxfp4_fused_numerics as numerics
from low_bits_training.quantization.fused_te_linear import _as_contiguous_bf16, _get_te_fused
from low_bits_training.quantization.mxfp4_backend import (
    mxfp4_batched_gemm,
    mxfp4_fused_silu_deriv_quantize_split2_row_and_col,
    mxfp4_gemm,
    mxfp4_quantize_split2_row_and_col,
    mxfp4_split2_dgrad_strided_onepass_gemm,
)
from low_bits_training.quantization.mxfp4_fused_linear import (
    _MXFP4RowCol,
    _get_mxfp4_ffn_bwd_state,
    _mxfp4_rmsnorm_backward,
    _quantize_row_col_bf16,
    _rmsnorm_quantize_row_col_bf16,
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(-1)
    bv = b.detach().float().reshape(-1)
    denom = float(av.norm() * bv.norm())
    if denom == 0.0:
        return float("nan")
    return float((av @ bv) / denom)


def finite_status(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def _cuda_empty() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _bench_cuda(fn, *, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return float(statistics.median(samples))


def _mxfp4_mode_kwargs(
    fused_ffn_split2: str,
    *,
    split2_ffn: str = "0",
    split2_onepass_dgrad_ffn: str = "0",
) -> dict[str, str]:
    return {
        "mode": "mxfp4_tk_fused",
        "qkv": "1",
        "ffn": "0",
        "split3_qkv": "1",
        "split2_ffn": split2_ffn,
        "fused_silu_split2_ffn": fused_ffn_split2,
        "split2_onepass_dgrad_ffn": split2_onepass_dgrad_ffn,
    }


def _mode_kwargs(name: str) -> dict[str, str]:
    if name == "mxfp4_baseline":
        return _mxfp4_mode_kwargs("0")
    if name == "mxfp4_split2_quant":
        return _mxfp4_mode_kwargs("0", split2_ffn="1")
    if name == "mxfp4_split2_onepass_dgrad":
        return _mxfp4_mode_kwargs("0", split2_ffn="1", split2_onepass_dgrad_ffn="1")
    if name == "mxfp4_fused_split2":
        return _mxfp4_mode_kwargs("1")
    if name == "fp4_localcta":
        return {
            "mode": "fp4_localcta",
            "qkv": "0",
            "ffn": "0",
            "split3_qkv": "0",
            "split2_ffn": "0",
            "fused_silu_split2_ffn": "0",
        }
    if name == "fp4_localcta_fused":
        return {
            "mode": "fp4_localcta_fused",
            "qkv": "0",
            "ffn": "0",
            "split3_qkv": "0",
            "split2_ffn": "0",
            "fused_silu_split2_ffn": "0",
        }
    raise ValueError(f"unknown mode name: {name}")


def build_ffn_for_mode(
    name: str,
    device: str,
    flavor: str,
    seed: int,
    *,
    mxfp4_backend_version: str,
):
    kwargs = _mode_kwargs(name)
    model, model_args = numerics.build_model_for_mode(
        kwargs["mode"],
        device,
        flavor,
        seed,
        mxfp4_backend_version=mxfp4_backend_version,
        qkv=kwargs["qkv"],
        ffn=kwargs["ffn"],
        split3_qkv=kwargs["split3_qkv"],
        split2_ffn=kwargs["split2_ffn"],
        fused_silu_split2_ffn=kwargs["fused_silu_split2_ffn"],
        split2_onepass_dgrad_ffn=kwargs.get("split2_onepass_dgrad_ffn", "0"),
    )
    _, _, ffn = numerics.fetch_block_refs(model)
    return model, model_args, ffn


def eval_ffn_outputs(
    name: str,
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str,
) -> dict[str, Any]:
    model = None
    try:
        model, model_args, ffn = build_ffn_for_mode(
            name,
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
        )
        torch.manual_seed(seed + 301)
        x = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        y = ffn(x)
        (y * gy).sum().backward()
        return {
            "forward": {"y": y.detach().cpu()},
            "backward": {"dx": x.grad.detach().cpu()},
            "finite": {
                "forward": finite_status(y),
                "backward": finite_status(x.grad),
            },
        }
    finally:
        numerics.free_model(model)


def compare_mxfp4_ffn_numerics(
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str,
) -> dict[str, Any]:
    model = None
    try:
        model, model_args, ffn = build_ffn_for_mode(
            "mxfp4_baseline",
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
        )
        torch.manual_seed(seed + 301)
        x_seed = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        x_base = x_seed.detach().clone().requires_grad_(True)
        x_fused = x_seed.detach().clone().requires_grad_(True)

        prev = os.environ.get("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN")
        os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = "0"
        y_base = ffn(x_base)
        os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = "1"
        y_fused = ffn(x_fused)
        (y_base * gy).sum().backward()
        (y_fused * gy).sum().backward()
        if prev is None:
            del os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"]
        else:
            os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = prev
        return {
            "finite": {
                "baseline_forward": finite_status(y_base),
                "baseline_backward": finite_status(x_base.grad),
                "fused_forward": finite_status(y_fused),
                "fused_backward": finite_status(x_fused.grad),
            },
            "y_cos": cosine(y_fused, y_base),
            "dx_cos": cosine(x_fused.grad, x_base.grad),
        }
    finally:
        numerics.free_model(model)


def _split2_outputs_to_rowcols(
    row_fp4: torch.Tensor,
    row_sc: torch.Tensor,
    col_fp4: torch.Tensor,
    col_sc: torch.Tensor,
    H: int,
) -> tuple[_MXFP4RowCol, _MXFP4RowCol]:
    H_packed = H // 2
    H_sc = H // 128
    row_fp4_u8 = row_fp4.view(torch.uint8)
    col_fp4_u8 = col_fp4.view(torch.uint8)
    dh1_q = _MXFP4RowCol(
        row_fp4=row_fp4_u8[:, :H_packed].contiguous().view(torch.float4_e2m1fn_x2),
        row_sc=row_sc[:, :H_sc].contiguous(),
        col_fp4=col_fp4_u8[:H].contiguous().view(torch.float4_e2m1fn_x2),
        col_sc=col_sc[:H_sc].contiguous(),
    )
    dh3_q = _MXFP4RowCol(
        row_fp4=row_fp4_u8[:, H_packed:].contiguous().view(torch.float4_e2m1fn_x2),
        row_sc=row_sc[:, H_sc:].contiguous(),
        col_fp4=col_fp4_u8[H:].contiguous().view(torch.float4_e2m1fn_x2),
        col_sc=col_sc[H_sc:].contiguous(),
    )
    return dh1_q, dh3_q


def compare_mxfp4_ffn_variant_numerics(
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str,
    variant: str,
) -> dict[str, Any]:
    model = None
    prev_split2 = os.environ.get("MXFP4_USE_SPLIT2_FFN_QUANT")
    prev_fused = os.environ.get("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN")
    prev_onepass = os.environ.get("MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD")
    try:
        model, model_args, ffn = build_ffn_for_mode(
            "mxfp4_baseline",
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
        )
        torch.manual_seed(seed + 301)
        x_seed = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        x_base = x_seed.detach().clone().requires_grad_(True)
        x_variant = x_seed.detach().clone().requires_grad_(True)

        os.environ["MXFP4_USE_SPLIT2_FFN_QUANT"] = "0"
        os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = "0"
        os.environ["MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD"] = "0"
        y_base = ffn(x_base)

        os.environ["MXFP4_USE_SPLIT2_FFN_QUANT"] = "1" if variant in ("split2_quant", "split2_onepass_dgrad") else "0"
        os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = "1" if variant == "fused_split2" else "0"
        os.environ["MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD"] = "1" if variant == "split2_onepass_dgrad" else "0"
        y_variant = ffn(x_variant)

        (y_base * gy).sum().backward()
        (y_variant * gy).sum().backward()

        if prev_split2 is None:
            del os.environ["MXFP4_USE_SPLIT2_FFN_QUANT"]
        else:
            os.environ["MXFP4_USE_SPLIT2_FFN_QUANT"] = prev_split2
        if prev_fused is None:
            del os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"]
        else:
            os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = prev_fused
        if prev_onepass is None:
            del os.environ["MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD"]
        else:
            os.environ["MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD"] = prev_onepass

        return {
            "finite": {
                "baseline_forward": finite_status(y_base),
                "baseline_backward": finite_status(x_base.grad),
                "variant_forward": finite_status(y_variant),
                "variant_backward": finite_status(x_variant.grad),
            },
            "y_cos": cosine(y_variant, y_base),
            "dx_cos": cosine(x_variant.grad, x_base.grad),
        }
    except Exception as exc:
        return {"error": repr(exc)}
    finally:
        if prev_split2 is None:
            os.environ.pop("MXFP4_USE_SPLIT2_FFN_QUANT", None)
        else:
            os.environ["MXFP4_USE_SPLIT2_FFN_QUANT"] = prev_split2
        if prev_fused is None:
            os.environ.pop("MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN", None)
        else:
            os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = prev_fused
        if prev_onepass is None:
            os.environ.pop("MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD", None)
        else:
            os.environ["MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD"] = prev_onepass
        try:
            numerics.free_model(model)
        except Exception:
            pass


def measure_ffn_block_ms(
    name: str,
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str,
    warmup: int,
    steps: int,
) -> float:
    model = None
    try:
        model, model_args, ffn = build_ffn_for_mode(
            name,
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
        )
        torch.manual_seed(seed + 302)
        x = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)

        def run_one() -> None:
            if x.grad is not None:
                x.grad = None
            ffn.zero_grad(set_to_none=True)
            y = ffn(x)
            (y * gy).sum().backward()

        return _bench_cuda(run_one, warmup=warmup, steps=steps)
    finally:
        numerics.free_model(model)


def _build_mxfp4_ffn_state(
    ffn,
    x: torch.Tensor,
) -> dict[str, Any]:
    te_fused = _get_te_fused()
    inp = _as_contiguous_bf16(x)
    nw = _as_contiguous_bf16(ffn.norm_weight.detach())
    x_q, inv_rms = _rmsnorm_quantize_row_col_bf16(te_fused, inp, nw, float(ffn.epsilon), kind="ffn")
    w1_q = _quantize_row_col_bf16(_as_contiguous_bf16(ffn.w1_weight.detach()))
    w3_q = _quantize_row_col_bf16(_as_contiguous_bf16(ffn.w3_weight.detach()))
    w2_q = _quantize_row_col_bf16(_as_contiguous_bf16(ffn.w2_weight.detach()))
    M, K = inp.shape
    H = ffn.w1_weight.shape[0]
    h1_raw = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
    h3 = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
    mxfp4_batched_gemm(
        [x_q.row_fp4, x_q.row_fp4],
        [x_q.row_sc, x_q.row_sc],
        [w1_q.row_fp4, w3_q.row_fp4],
        [w1_q.row_sc, w3_q.row_sc],
        [h1_raw, h3],
    )
    h = torch.empty(M, H, dtype=torch.bfloat16, device=inp.device)
    if hasattr(te_fused, "fused_silu_mul_bf16_out_no_amax"):
        te_fused.fused_silu_mul_bf16_out_no_amax(h1_raw, h3, h)
    elif hasattr(te_fused, "fused_silu_mul_bf16_out"):
        amax = torch.empty(1, dtype=torch.float32, device=inp.device)
        te_fused.fused_silu_mul_bf16_out(h1_raw, h3, h, amax)
    else:
        h = te_fused.fused_silu_mul_bf16(h1_raw, h3)[0]
    h_q = _quantize_row_col_bf16(h)
    return {
        "inp": inp,
        "nw": nw,
        "inv_rms": inv_rms,
        "x_q": x_q,
        "w1_q": w1_q,
        "w3_q": w3_q,
        "w2_q": w2_q,
        "h1_raw": h1_raw,
        "h3": h3,
        "h_q": h_q,
        "te_fused": te_fused,
    }


def manual_mxfp4_ffn_backward_breakdown(
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str,
    split2_quant: bool,
    fused_split2: bool,
    split2_onepass_dgrad: bool = False,
    warmup: int,
    steps: int,
) -> dict[str, float]:
    if split2_onepass_dgrad:
        name = "mxfp4_split2_onepass_dgrad"
    elif fused_split2:
        name = "mxfp4_fused_split2"
    elif split2_quant:
        name = "mxfp4_split2_quant"
    else:
        name = "mxfp4_baseline"
    model = None
    try:
        model, model_args, ffn = build_ffn_for_mode(
            name,
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
        )
        torch.manual_seed(seed + 303)
        x = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        dY = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        fwd_state = _build_mxfp4_ffn_state(ffn, x)
        inp = fwd_state["inp"]
        H = ffn.w1_weight.shape[0]
        K = model_args.dim

        def alloc_state() -> dict[str, torch.Tensor]:
            return _get_mxfp4_ffn_bwd_state(M, K, H, inp.device)

        def quant_dy() -> None:
            _quantize_row_col_bf16(dY)

        dy_q = _quantize_row_col_bf16(dY)

        def dh_stage() -> None:
            dh = alloc_state()["dh"]
            mxfp4_gemm(dy_q.row_fp4, dy_q.row_sc, fwd_state["w2_q"].col_fp4, fwd_state["w2_q"].col_sc, dh)

        dh_stage()
        dh = alloc_state()["dh"]
        mxfp4_gemm(dy_q.row_fp4, dy_q.row_sc, fwd_state["w2_q"].col_fp4, fwd_state["w2_q"].col_sc, dh)

        def grad_w2_stage() -> None:
            grad_w2 = alloc_state()["grad_w2"]
            mxfp4_gemm(dy_q.col_fp4, dy_q.col_sc, fwd_state["h_q"].col_fp4, fwd_state["h_q"].col_sc, grad_w2)

        def producer_stage() -> None:
            state = alloc_state()
            dh1 = state["dh1"]
            dh3 = state["dh3"]
            te_fused = fwd_state["te_fused"]
            if hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
                te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(dh, fwd_state["h3"], fwd_state["h1_raw"], dh1, dh3)
            elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out"):
                amax1 = torch.empty(1, dtype=torch.float32, device=dh.device)
                amax2 = torch.empty(1, dtype=torch.float32, device=dh.device)
                te_fused.fused_silu_deriv_dual_mul_bf16_out(dh, fwd_state["h3"], fwd_state["h1_raw"], dh1, dh3, amax1, amax2)
            else:
                tmp1, tmp3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, fwd_state["h3"], fwd_state["h1_raw"])
                dh1.copy_(tmp1)
                dh3.copy_(tmp3)

        producer_stage()
        state = alloc_state()
        te_fused = fwd_state["te_fused"]
        if hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
            te_fused.fused_silu_deriv_dual_mul_bf16_out_no_amax(dh, fwd_state["h3"], fwd_state["h1_raw"], state["dh1"], state["dh3"])
        elif hasattr(te_fused, "fused_silu_deriv_dual_mul_bf16_out"):
            amax1 = torch.empty(1, dtype=torch.float32, device=dh.device)
            amax2 = torch.empty(1, dtype=torch.float32, device=dh.device)
            te_fused.fused_silu_deriv_dual_mul_bf16_out(dh, fwd_state["h3"], fwd_state["h1_raw"], state["dh1"], state["dh3"], amax1, amax2)
        else:
            tmp1, tmp3, _, _ = te_fused.fused_silu_deriv_dual_mul_bf16(dh, fwd_state["h3"], fwd_state["h1_raw"])
            state["dh1"].copy_(tmp1)
            state["dh3"].copy_(tmp3)

        def quant_stage() -> None:
            if split2_quant:
                mxfp4_quantize_split2_row_and_col(state["dh1"], state["dh3"])
            else:
                _quantize_row_col_bf16(state["dh1"])
                _quantize_row_col_bf16(state["dh3"])

        def fused_producer_quant_stage() -> None:
            mxfp4_fused_silu_deriv_quantize_split2_row_and_col(dh, fwd_state["h3"], fwd_state["h1_raw"])

        split2_row_fp4 = None
        split2_row_sc = None
        split2_h_packed = None
        split2_h_sc = None
        if fused_split2:
            row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_deriv_quantize_split2_row_and_col(
                dh, fwd_state["h3"], fwd_state["h1_raw"]
            )
            dh1_q = _MXFP4RowCol(row_fp4=row_fp4[0], row_sc=row_sc[0], col_fp4=col_fp4[0], col_sc=col_sc[0])
            dh3_q = _MXFP4RowCol(row_fp4=row_fp4[1], row_sc=row_sc[1], col_fp4=col_fp4[1], col_sc=col_sc[1])
        elif split2_quant or split2_onepass_dgrad:
            row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_split2_row_and_col(state["dh1"], state["dh3"])
            split2_row_fp4 = row_fp4
            split2_row_sc = row_sc
            split2_h_packed = H // 2
            split2_h_sc = H // 128
            dh1_q, dh3_q = _split2_outputs_to_rowcols(row_fp4, row_sc, col_fp4, col_sc, H)
        else:
            dh1_q = _quantize_row_col_bf16(state["dh1"])
            dh3_q = _quantize_row_col_bf16(state["dh3"])

        def dgrad_stage() -> None:
            state = alloc_state()
            if split2_onepass_dgrad:
                mxfp4_split2_dgrad_strided_onepass_gemm(
                    split2_row_fp4,
                    [
                        split2_row_sc.narrow(1, 0, split2_h_sc),
                        split2_row_sc.narrow(1, split2_h_sc, split2_h_sc),
                    ],
                    [0, split2_h_packed],
                    [split2_h_packed, split2_h_packed],
                    [fwd_state["w1_q"].col_fp4, fwd_state["w3_q"].col_fp4],
                    [fwd_state["w1_q"].col_sc, fwd_state["w3_q"].col_sc],
                    state["dx0"],
                    -1,
                )
            else:
                mxfp4_batched_gemm(
                    [dh1_q.row_fp4, dh3_q.row_fp4],
                    [dh1_q.row_sc, dh3_q.row_sc],
                    [fwd_state["w1_q"].col_fp4, fwd_state["w3_q"].col_fp4],
                    [fwd_state["w1_q"].col_sc, fwd_state["w3_q"].col_sc],
                    [state["dx0"], state["dx1"]],
                )
                state["dx0"].add_(state["dx1"])

        def wgrad_stage() -> None:
            state = alloc_state()
            mxfp4_batched_gemm(
                [dh1_q.col_fp4, dh3_q.col_fp4],
                [dh1_q.col_sc, dh3_q.col_sc],
                [fwd_state["x_q"].col_fp4, fwd_state["x_q"].col_fp4],
                [fwd_state["x_q"].col_sc, fwd_state["x_q"].col_sc],
                [state["grad_w1"], state["grad_w3"]],
            )

        dgrad_stage()
        dx_normed = alloc_state()["dx0"]

        def rmsnorm_bwd_stage() -> None:
            _mxfp4_rmsnorm_backward(
                fwd_state["te_fused"],
                dx_normed,
                inp,
                fwd_state["nw"],
                fwd_state["inv_rms"],
            )

        def full_stage() -> None:
            state = alloc_state()
            dy_local = _quantize_row_col_bf16(dY)
            mxfp4_gemm(dy_local.row_fp4, dy_local.row_sc, fwd_state["w2_q"].col_fp4, fwd_state["w2_q"].col_sc, state["dh"])
            mxfp4_gemm(dy_local.col_fp4, dy_local.col_sc, fwd_state["h_q"].col_fp4, fwd_state["h_q"].col_sc, state["grad_w2"])
            if fused_split2:
                row_fp4, row_sc, col_fp4, col_sc = mxfp4_fused_silu_deriv_quantize_split2_row_and_col(
                    state["dh"], fwd_state["h3"], fwd_state["h1_raw"]
                )
                dh1_local = _MXFP4RowCol(row_fp4=row_fp4[0], row_sc=row_sc[0], col_fp4=col_fp4[0], col_sc=col_sc[0])
                dh3_local = _MXFP4RowCol(row_fp4=row_fp4[1], row_sc=row_sc[1], col_fp4=col_fp4[1], col_sc=col_sc[1])
            else:
                te_local = fwd_state["te_fused"]
                if hasattr(te_local, "fused_silu_deriv_dual_mul_bf16_out_no_amax"):
                    te_local.fused_silu_deriv_dual_mul_bf16_out_no_amax(
                        state["dh"], fwd_state["h3"], fwd_state["h1_raw"], state["dh1"], state["dh3"]
                    )
                elif hasattr(te_local, "fused_silu_deriv_dual_mul_bf16_out"):
                    amax1 = torch.empty(1, dtype=torch.float32, device=inp.device)
                    amax2 = torch.empty(1, dtype=torch.float32, device=inp.device)
                    te_local.fused_silu_deriv_dual_mul_bf16_out(
                        state["dh"], fwd_state["h3"], fwd_state["h1_raw"], state["dh1"], state["dh3"], amax1, amax2
                    )
                else:
                    tmp1, tmp3, _, _ = te_local.fused_silu_deriv_dual_mul_bf16(state["dh"], fwd_state["h3"], fwd_state["h1_raw"])
                    state["dh1"].copy_(tmp1)
                    state["dh3"].copy_(tmp3)
                if split2_quant or split2_onepass_dgrad:
                    row_fp4, row_sc, col_fp4, col_sc = mxfp4_quantize_split2_row_and_col(state["dh1"], state["dh3"])
                    dh1_local, dh3_local = _split2_outputs_to_rowcols(row_fp4, row_sc, col_fp4, col_sc, H)
                else:
                    dh1_local = _quantize_row_col_bf16(state["dh1"])
                    dh3_local = _quantize_row_col_bf16(state["dh3"])
            if split2_onepass_dgrad:
                mxfp4_split2_dgrad_strided_onepass_gemm(
                    row_fp4,
                    [row_sc.narrow(1, 0, H // 128), row_sc.narrow(1, H // 128, H // 128)],
                    [0, H // 2],
                    [H // 2, H // 2],
                    [fwd_state["w1_q"].col_fp4, fwd_state["w3_q"].col_fp4],
                    [fwd_state["w1_q"].col_sc, fwd_state["w3_q"].col_sc],
                    state["dx0"],
                    -1,
                )
            else:
                mxfp4_batched_gemm(
                    [dh1_local.row_fp4, dh3_local.row_fp4],
                    [dh1_local.row_sc, dh3_local.row_sc],
                    [fwd_state["w1_q"].col_fp4, fwd_state["w3_q"].col_fp4],
                    [fwd_state["w1_q"].col_sc, fwd_state["w3_q"].col_sc],
                    [state["dx0"], state["dx1"]],
                )
                state["dx0"].add_(state["dx1"])
            mxfp4_batched_gemm(
                [dh1_local.col_fp4, dh3_local.col_fp4],
                [dh1_local.col_sc, dh3_local.col_sc],
                [fwd_state["x_q"].col_fp4, fwd_state["x_q"].col_fp4],
                [fwd_state["x_q"].col_sc, fwd_state["x_q"].col_sc],
                [state["grad_w1"], state["grad_w3"]],
            )
            _mxfp4_rmsnorm_backward(
                fwd_state["te_fused"],
                state["dx0"],
                inp,
                fwd_state["nw"],
                fwd_state["inv_rms"],
            )

        out = {
            "dY_quant_ms": _bench_cuda(quant_dy, warmup=warmup, steps=steps),
            "dh_w2_dgrad_ms": _bench_cuda(dh_stage, warmup=warmup, steps=steps),
            "grad_w2_wgrad_ms": _bench_cuda(grad_w2_stage, warmup=warmup, steps=steps),
            "dgrad_gemm_ms": _bench_cuda(dgrad_stage, warmup=warmup, steps=steps),
            "wgrad_gemm_ms": _bench_cuda(wgrad_stage, warmup=warmup, steps=steps),
            "rmsnorm_bwd_ms": _bench_cuda(rmsnorm_bwd_stage, warmup=warmup, steps=steps),
            "full_manual_backward_ms": _bench_cuda(full_stage, warmup=warmup, steps=steps),
        }
        if fused_split2:
            out["producer_quant_ms"] = _bench_cuda(fused_producer_quant_stage, warmup=warmup, steps=steps)
        else:
            out["producer_ms"] = _bench_cuda(producer_stage, warmup=warmup, steps=steps)
            out["quant_ms"] = _bench_cuda(quant_stage, warmup=warmup, steps=steps)
        named_sum = sum(v for k, v in out.items() if k != "full_manual_backward_ms")
        out["packaging_other_ms"] = max(out["full_manual_backward_ms"] - named_sum, 0.0)
        return out
    finally:
        numerics.free_model(model)


def _category_map(key: str) -> str | None:
    if re.search(r"silu_deriv", key):
        return "derivative_producer_ms"
    if re.search(r"quantize|localcta_fused_norm_to_bf16|mxfp4_v[34]_.*kernel", key):
        return "quantization_ms"
    if re.search(r"gemm|generic_gemm", key):
        return "gemm_ms"
    if re.search(r"copy_|cat|slice|narrow|view|reshape|clone|empty|memcpy", key):
        return "wrapper_packaging_ms"
    if re.search(r"rmsnorm_backward|dgamma|dx_only", key):
        return "rmsnorm_backward_ms"
    return None


def profile_ffn_backward_categories(
    name: str,
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str,
) -> dict[str, Any]:
    model = None
    try:
        model, model_args, ffn = build_ffn_for_mode(
            name,
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
        )
        torch.manual_seed(seed + 304)
        x = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        y = ffn(x)
        (y * gy).sum().backward()
        torch.cuda.synchronize()
        x = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            y = ffn(x)
            (y * gy).sum().backward()
            torch.cuda.synchronize()
        out: dict[str, float] = {
            "derivative_producer_ms": 0.0,
            "quantization_ms": 0.0,
            "gemm_ms": 0.0,
            "wrapper_packaging_ms": 0.0,
            "rmsnorm_backward_ms": 0.0,
        }
        top: list[dict[str, Any]] = []
        for evt in prof.key_averages():
            key = evt.key
            self_cuda_ms = float(evt.self_device_time_total) / 1000.0
            self_cpu_ms = float(evt.self_cpu_time_total) / 1000.0
            cat = _category_map(key)
            if cat is not None:
                out[cat] += self_cuda_ms
            if self_cuda_ms > 0.0:
                top.append({"key": key, "self_cuda_ms": self_cuda_ms, "self_cpu_ms": self_cpu_ms})
        top.sort(key=lambda item: item["self_cuda_ms"], reverse=True)
        out["top_cuda_events"] = top[:12]
        return out
    finally:
        numerics.free_model(model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--flavor", default="1B", choices=["1B", "1B_legacy"])
    parser.add_argument("--m", type=int, default=65536)
    parser.add_argument("--mxfp4-backend-version", default="v4")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=["isolated", "manual", "numerics", "profile"],
        default=["isolated", "manual", "numerics", "profile"],
    )
    args = parser.parse_args()

    device = f"cuda:{args.device_index}"
    report: dict[str, Any] = {
        "device_index": args.device_index,
        "flavor": args.flavor,
        "m": args.m,
        "mxfp4_backend_version": args.mxfp4_backend_version,
    }

    if "isolated" in args.sections:
        report["isolated_ffn_block_ms"] = {
            name: measure_ffn_block_ms(
                name,
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                warmup=args.warmup,
                steps=args.steps,
            )
            for name in (
                "mxfp4_baseline",
                "mxfp4_split2_quant",
                "mxfp4_split2_onepass_dgrad",
                "mxfp4_fused_split2",
                "fp4_localcta",
                "fp4_localcta_fused",
            )
        }

    if "manual" in args.sections:
        report["mxfp4_manual_backward_ms"] = {
            "baseline": manual_mxfp4_ffn_backward_breakdown(
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                split2_quant=False,
                fused_split2=False,
                warmup=args.warmup,
                steps=args.steps,
            ),
            "split2_quant": manual_mxfp4_ffn_backward_breakdown(
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                split2_quant=True,
                fused_split2=False,
                split2_onepass_dgrad=False,
                warmup=args.warmup,
                steps=args.steps,
            ),
            "split2_onepass_dgrad": manual_mxfp4_ffn_backward_breakdown(
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                split2_quant=False,
                fused_split2=False,
                split2_onepass_dgrad=True,
                warmup=args.warmup,
                steps=args.steps,
            ),
            "fused_split2": manual_mxfp4_ffn_backward_breakdown(
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                split2_quant=False,
                fused_split2=True,
                split2_onepass_dgrad=False,
                warmup=args.warmup,
                steps=args.steps,
            ),
        }

    if "numerics" in args.sections:
        report["numerics"] = compare_mxfp4_ffn_numerics(
            device,
            args.flavor,
            args.m,
            args.seed,
            mxfp4_backend_version=args.mxfp4_backend_version,
        )
        report["variant_numerics"] = {
            "split2_quant": compare_mxfp4_ffn_variant_numerics(
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                variant="split2_quant",
            ),
            "split2_onepass_dgrad": compare_mxfp4_ffn_variant_numerics(
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                variant="split2_onepass_dgrad",
            ),
            "fused_split2": compare_mxfp4_ffn_variant_numerics(
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                variant="fused_split2",
            ),
        }

    if "profile" in args.sections:
        report["profile_category_ms"] = {
            name: profile_ffn_backward_categories(
                name,
                device,
                args.flavor,
                args.m,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
            )
            for name in (
                "mxfp4_baseline",
                "mxfp4_split2_quant",
                "mxfp4_split2_onepass_dgrad",
                "mxfp4_fused_split2",
                "fp4_localcta",
                "fp4_localcta_fused",
            )
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
