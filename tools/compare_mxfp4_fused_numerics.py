#!/usr/bin/env python3
"""Compare benchmark MXFP4 fused numerics across fused-rmsnorm-quant modes.

This is intentionally benchmark-oriented and reuses the synthetic 1B build path.
It compares:
- MXFP4 fused v4 baseline (QKV=0, FFN=0)
- MXFP4 fused v4 QKV-only fused-rmsnorm-quant (QKV=1, FFN=0)
- MXFP4 fused v4 all-on fused-rmsnorm-quant (QKV=1, FFN=1)
- localCTA fused NVFP4
- TE-native MXFP4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from typing import Any

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCHTITAN_ROOT = os.path.join(REPO_ROOT, "torchtitan_submodule")
FALLBACK_TORCHTITAN_ROOT = "/opt/mfu/EXTERNAL_PATH"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, TORCHTITAN_ROOT)
if os.path.isdir(FALLBACK_TORCHTITAN_ROOT) and FALLBACK_TORCHTITAN_ROOT not in sys.path:
    sys.path.insert(0, FALLBACK_TORCHTITAN_ROOT)

import bench_synth_1b_fp4 as bench


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    av = a.detach().float().reshape(-1)
    bv = b.detach().float().reshape(-1)
    denom = float(av.norm() * bv.norm())
    if denom == 0.0:
        return float("nan")
    return float((av @ bv) / denom)


def finite_status(*tensors: torch.Tensor) -> bool:
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def set_mxfp4_env(
    qkv: str = "0",
    ffn: str = "0",
    split3_qkv: str = "0",
    split2_ffn: str = "0",
    fused_silu_split2_ffn: str = "0",
    split2_onepass_dgrad_ffn: str = "0",
) -> None:
    os.environ["MXFP4_USE_FUSED_RMSNORM_QUANT"] = "0"
    os.environ["MXFP4_USE_FUSED_RMSNORM_QUANT_QKV"] = qkv
    os.environ["MXFP4_USE_FUSED_RMSNORM_QUANT_FFN"] = ffn
    os.environ["MXFP4_USE_SPLIT3_QKV_QUANT"] = split3_qkv
    os.environ["MXFP4_USE_SPLIT2_FFN_QUANT"] = split2_ffn
    os.environ["MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN"] = fused_silu_split2_ffn
    os.environ["MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD"] = split2_onepass_dgrad_ffn


def build_model_for_mode(
    mode: str,
    device: str,
    flavor: str,
    seed: int,
    *,
    mxfp4_backend_version: str = "v4",
    qkv: str = "0",
    ffn: str = "0",
    split3_qkv: str = "0",
    split2_ffn: str = "0",
    fused_silu_split2_ffn: str = "0",
    split2_onepass_dgrad_ffn: str = "0",
):
    if device.startswith("cuda:"):
        torch.cuda.set_device(int(device.split(":")[1]))
    torch.manual_seed(seed)
    set_mxfp4_env(
        qkv=qkv,
        ffn=ffn,
        split3_qkv=split3_qkv,
        split2_ffn=split2_ffn,
        fused_silu_split2_ffn=fused_silu_split2_ffn,
        split2_onepass_dgrad_ffn=split2_onepass_dgrad_ffn,
    )
    bench.configure_env(mode, mxfp4_backend_version)
    model, model_args = bench.build_model(flavor, mode, device)
    return model, model_args


def free_model(model: torch.nn.Module | None) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()


def fetch_block_refs(model):
    block = model.layers["0"]
    attn = block.attention.fused if hasattr(block.attention, "fused") else block.attention
    ffn = block.feed_forward
    return block, attn, ffn


def eval_qkv_case(
    mode: str,
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str = "v4",
    qkv: str = "0",
    ffn: str = "0",
    split3_qkv: str = "0",
) -> dict[str, Any]:
    model = None
    try:
        model, model_args = build_model_for_mode(
            mode,
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
            qkv=qkv,
            ffn=ffn,
            split3_qkv=split3_qkv,
        )
        _, attn, _ = fetch_block_refs(model)
        torch.manual_seed(seed + 100)
        x = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        gq = torch.randn(M, attn.q_dim, device=device, dtype=torch.bfloat16)
        gk = torch.randn(M, attn.k_dim, device=device, dtype=torch.bfloat16)
        gv = torch.randn(M, attn.v_dim, device=device, dtype=torch.bfloat16)
        q, k, v = attn.forward_qkv(x)
        (((q * gq).sum() + (k * gk).sum() + (v * gv).sum())).backward()
        return {
            "forward": {
                "q": q.detach().cpu(),
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),
            },
            "backward": {
                "dx": x.grad.detach().cpu(),
            },
            "finite": {
                "forward": finite_status(q, k, v),
                "backward": finite_status(x.grad),
            },
        }
    finally:
        free_model(model)


def eval_wo_case(
    mode: str,
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str = "v4",
    qkv: str = "0",
    ffn: str = "0",
    split3_qkv: str = "0",
) -> dict[str, Any]:
    model = None
    try:
        model, model_args = build_model_for_mode(
            mode,
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
            qkv=qkv,
            ffn=ffn,
            split3_qkv=split3_qkv,
        )
        _, attn, _ = fetch_block_refs(model)
        torch.manual_seed(seed + 200)
        x = torch.randn(M, attn.q_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        y = attn.forward_wo(x)
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
        free_model(model)


def eval_ffn_case(
    mode: str,
    device: str,
    flavor: str,
    M: int,
    seed: int,
    *,
    mxfp4_backend_version: str = "v4",
    qkv: str = "0",
    ffn: str = "0",
    split3_qkv: str = "0",
) -> dict[str, Any]:
    model = None
    try:
        model, model_args = build_model_for_mode(
            mode,
            device,
            flavor,
            seed,
            mxfp4_backend_version=mxfp4_backend_version,
            qkv=qkv,
            ffn=ffn,
            split3_qkv=split3_qkv,
        )
        _, _, block_ffn = fetch_block_refs(model)
        torch.manual_seed(seed + 300)
        x = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        gy = torch.randn(M, model_args.dim, device=device, dtype=torch.bfloat16)
        y = block_ffn(x)
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
        free_model(model)


def compare_case(ref: dict[str, Any], other: dict[str, Any], forward_keys: list[str]) -> dict[str, Any]:
    if "error" in ref:
        return {"error": f"reference failed: {ref['error']}"}
    if "error" in other:
        return {"error": other["error"]}
    out: dict[str, Any] = {
        "finite": {
            "forward": other["finite"]["forward"],
            "backward": other["finite"]["backward"],
        }
    }
    for key in forward_keys:
        out[f"{key}_cos"] = cosine(other["forward"][key], ref["forward"][key])
    if "dx" in ref["backward"] and "dx" in other["backward"]:
        out["dx_cos"] = cosine(other["backward"]["dx"], ref["backward"]["dx"])
    return out


def safe_eval(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - benchmark robustness
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"{type(exc).__name__}: {exc}"}


def safe_synth(*args, **kwargs) -> dict[str, Any]:
    try:
        return synthetic_step(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - benchmark robustness
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"{type(exc).__name__}: {exc}"}


def synthetic_step(
    mode: str,
    device_index: int,
    flavor: str,
    *,
    mxfp4_backend_version: str = "v4",
    qkv: str = "0",
    ffn: str = "0",
    split3_qkv: str = "0",
) -> dict[str, Any]:
    set_mxfp4_env(qkv=qkv, ffn=ffn, split3_qkv=split3_qkv)
    kwargs: dict[str, Any] = {"device_index": device_index}
    if mode == "mxfp4_tk_fused":
        kwargs["mxfp4_backend_version"] = mxfp4_backend_version
    return bench.run_one(mode, flavor, 64, 1024, 1, 1, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--flavor", default="1B", choices=["1B", "1B_legacy"])
    parser.add_argument("--m-values", type=int, nargs="+", default=[4096, 65536])
    parser.add_argument("--mxfp4-backend-version", default="v4")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    device = f"cuda:{args.device_index}"

    modes = {
        "mxfp4_off": ("mxfp4_tk_fused", "0", "0", "0"),
        "mxfp4_qkv_only": ("mxfp4_tk_fused", "1", "0", "1"),
        "mxfp4_all_on": ("mxfp4_tk_fused", "1", "1", "1"),
        "localcta_fused": ("fp4_localcta_fused", "0", "0", "0"),
        "te_mxfp4_native": ("te_mxfp4_native", "0", "0", "0"),
    }

    report: dict[str, Any] = {
        "device_index": args.device_index,
        "flavor": args.flavor,
        "mxfp4_backend_version": args.mxfp4_backend_version,
        "blocks": {},
        "synthetic_1b_step": {},
    }

    for M in args.m_values:
        block_report: dict[str, Any] = {}
        qkv_results = {
            name: safe_eval(
                eval_qkv_case,
                mode,
                device,
                args.flavor,
                M,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                qkv=qkv,
                ffn=ffn,
                split3_qkv=split3_qkv,
            )
            for name, (mode, qkv, ffn, split3_qkv) in modes.items()
        }
        wo_results = {
            name: safe_eval(
                eval_wo_case,
                mode,
                device,
                args.flavor,
                M,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                qkv=qkv,
                ffn=ffn,
                split3_qkv=split3_qkv,
            )
            for name, (mode, qkv, ffn, split3_qkv) in modes.items()
        }
        ffn_results = {
            name: safe_eval(
                eval_ffn_case,
                mode,
                device,
                args.flavor,
                M,
                args.seed,
                mxfp4_backend_version=args.mxfp4_backend_version,
                qkv=qkv,
                ffn=ffn,
                split3_qkv=split3_qkv,
            )
            for name, (mode, qkv, ffn, split3_qkv) in modes.items()
        }

        block_report["qkv"] = {
            "mxfp4_qkv_only_vs_off": compare_case(qkv_results["mxfp4_off"], qkv_results["mxfp4_qkv_only"], ["q", "k", "v"]),
            "mxfp4_all_on_vs_off": compare_case(qkv_results["mxfp4_off"], qkv_results["mxfp4_all_on"], ["q", "k", "v"]),
            "localcta_fused_vs_off": compare_case(qkv_results["mxfp4_off"], qkv_results["localcta_fused"], ["q", "k", "v"]),
            "te_mxfp4_native_vs_off": compare_case(qkv_results["mxfp4_off"], qkv_results["te_mxfp4_native"], ["q", "k", "v"]),
        }
        block_report["wo"] = {
            "mxfp4_qkv_only_vs_off": compare_case(wo_results["mxfp4_off"], wo_results["mxfp4_qkv_only"], ["y"]),
            "mxfp4_all_on_vs_off": compare_case(wo_results["mxfp4_off"], wo_results["mxfp4_all_on"], ["y"]),
            "localcta_fused_vs_off": compare_case(wo_results["mxfp4_off"], wo_results["localcta_fused"], ["y"]),
            "te_mxfp4_native_vs_off": compare_case(wo_results["mxfp4_off"], wo_results["te_mxfp4_native"], ["y"]),
        }
        block_report["ffn"] = {
            "mxfp4_qkv_only_vs_off": compare_case(ffn_results["mxfp4_off"], ffn_results["mxfp4_qkv_only"], ["y"]),
            "mxfp4_all_on_vs_off": compare_case(ffn_results["mxfp4_off"], ffn_results["mxfp4_all_on"], ["y"]),
            "localcta_fused_vs_off": compare_case(ffn_results["mxfp4_off"], ffn_results["localcta_fused"], ["y"]),
            "te_mxfp4_native_vs_off": compare_case(ffn_results["mxfp4_off"], ffn_results["te_mxfp4_native"], ["y"]),
        }
        report["blocks"][f"M={M}"] = block_report

    for name, (mode, qkv, ffn, split3_qkv) in modes.items():
        stats = safe_synth(
            mode,
            args.device_index,
            args.flavor,
            mxfp4_backend_version=args.mxfp4_backend_version,
            qkv=qkv,
            ffn=ffn,
            split3_qkv=split3_qkv,
        )
        if "error" in stats:
            report["synthetic_1b_step"][name] = stats
        else:
            report["synthetic_1b_step"][name] = {
                "total_ms": stats["total_ms"],
                "forward_ms": stats["forward_ms"],
                "backward_ms": stats["backward_ms"],
                "loss_median": stats["loss_median"],
                "peak_mem_gib": stats["peak_mem_gib"],
            }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
