#!/usr/bin/env python3
"""Run local 8B BF16/FP4 LBT experiments for the NVFP4 paper proxy.

The default model flavor is a Llama/TorchTitan proxy for the NVFP4 paper's
8B Nemotron-H hybrid Mamba-Transformer. It matches the paper's transformer-side
dimensions where this codebase can: hidden size 4096, FFN hidden size 21504,
32 query heads, 4 KV heads, sequence length 8192, and a final-eight-FFN
high-precision policy. It is not the exact 52-block Mamba/attention hybrid.

For reproducing the older Llama 3 8B throughput high-water numbers, pass
--model-flavor 8B_llama3_blog.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as stats
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP4_ROOT = Path("/opt/mfu/EXTERNAL_PATH")
DEFAULT_TE213_STAGE = Path("/tmp/te213_stage")
DEFAULT_TE213_LIB = DEFAULT_TE213_STAGE / "nvidia" / "cu13" / "lib"
DEFAULT_HF_ASSETS = Path("./assets/hf/Meta-Llama-3-8B")
DEFAULT_TRAIN_DATA = Path("/tmp/lbt_packed/slimpajama_1b_tokens_20260529")
DEFAULT_VALIDATION_DATA = Path("/tmp/lbt_packed/c4_validation_heldout_16m_20260603")
DEFAULT_BLACKLISTED_GPUS = {"2"}
DEFAULT_MODEL_FLAVOR = os.environ.get("LBT_NVFP4_8B_MODEL_FLAVOR", "8B_nvfp4_paper_proxy")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nvpaper_1p2b_numerics_500 import (  # noqa: E402
    parse_step_metrics,
    parse_validation_metrics,
    summarize_rows,
    summarize_validation,
)


@dataclass(frozen=True)
class Case:
    name: str
    config: str
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()
    needs_te213: bool = False


def _disable_tk_env() -> dict[str, str]:
    return {
        "USE_TK_GEMM": "0",
        "USE_TK_QUANT": "0",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_FUSED": "0",
        "USE_MXFP4_TK_BACKEND": "0",
        "USE_MXFP4_TK_FUSED": "0",
    }


def _safe_reference_attention_env() -> dict[str, str]:
    return {
        "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "0",
        "LOW_BITS_SDPA_BACKENDS": "flash",
    }


def _native_nvfp4_cce_env() -> dict[str, str]:
    return {
        "FP4_CCE_ASSUME_NONEMPTY_LABELS": "1",
        "FP4_CCE_NVFP4_EXACT_NORM_QUANT": "0",
        "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
        "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
        "FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED": "1",
        "FP4_CCE_V4_NVFP4_G_CACHE": "0",
        "FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE": "1",
        "FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE": "0",
        "FP4_CCE_V4_NVFP4_P_DATA_SR": "1",
        "FP4_CCE_V4_NVFP4_P_SCALE_SR": "0",
        "FP4_CCE_V4_NVFP4_P_TARGET_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_P_TOP1_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_STAGED_P_CACHE": "1",
        "FP4_CCE_V4_NVFP4_TILED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_TMA_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_FUSED_STAGED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_DIRECT_SOFTMAX": "0",
        "FP4_CCE_V4_STRICT_FUSED_SPARSE": "1",
    }


def _native_nvfp4_cce_args() -> tuple[str, ...]:
    return (
        "--fp4-cce.enabled",
        "--fp4-cce.backend",
        "nvfp4",
        "--fp4-cce.implementation",
        "v4",
        "--fp4-cce.quant-mode",
        "enc",
    )


def _nvfp4_noextras_env() -> dict[str, str]:
    return {
        "NVTE_NVFP4_DISABLE_RHT": "1",
        "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "NVTE_NVFP4_ENCODE_CENTRIC": "0",
        "NVFP4_USE_RHT": "0",
        "NVFP4_RHT_ACTIVATION": "0",
        "NVFP4_RHT_GRAD": "0",
        "NVFP4_RHT_WEIGHT": "0",
        "NVFP4_USE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SR_ACTIVATION": "0",
        "NVFP4_SR_GRAD": "0",
        "NVFP4_SR_WEIGHT": "0",
        "NVFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SCALE_SR_ACTIVATION": "0",
        "NVFP4_SCALE_SR_GRAD": "0",
        "NVFP4_SCALE_SR_WEIGHT": "0",
    }


def _tk_v5_swiglu_env() -> dict[str, str]:
    return {
        **_nvfp4_noextras_env(),
        "USE_TK_GEMM": "1",
        "USE_TK_QUANT": "1",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_FUSED": "0",
        "FP4_ATTN_BACKEND": "tk",
        "FP4_FFN_BACKEND": "tk",
        "NVTE_CUSTOM_QUANT": "1",
        "USE_TK_QKV_ROPE_EPILOGUE": "1",
        "USE_TK_V5_STRIDED_Q_ATTN": "1",
        # The grouped v5 quantizer now owns its asynchronous TMA descriptor
        # staging per call, so the exact 8B fused QKV route is stream-safe.
        "USE_TK_QKV_FUSED_SUM_RMS": "1",
        "USE_TK_QKV_PLAIN_BATCHED_ACCUM_DGRAD": "0",
        "USE_TK_FFN_SPLIT2_OPT_PRODUCER": "1",
        "USE_TK_FFN_SPLIT_CACHE": "1",
        "USE_TK_FFN_RECOMPUTE_H13": "1",
        "USE_TK_FFN_CACHED_RETURN_TRANSPOSE": "1",
        "USE_TK_FFN_FUSED_SUM_RMS": "1",
        "USE_TK_FFN_OVERLAP_RMS_WGRAD": "1",
        "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
        "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
        "FP4_CCE_NVFP4_EXACT_NORM_QUANT": "0",
        "FP4_CCE_ASSUME_NONEMPTY_LABELS": "1",
        "FP4_CCE_V4_NVFP4_G_CACHE": "0",
        "FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE": "1",
        "FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE": "0",
        "FP4_CCE_V4_NVFP4_P_DATA_SR": "1",
        "FP4_CCE_V4_NVFP4_P_SCALE_SR": "0",
        "FP4_CCE_V4_NVFP4_P_TARGET_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_P_TOP1_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_STAGED_P_CACHE": "1",
        "FP4_CCE_V4_NVFP4_TILED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_TMA_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_FUSED_STAGED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_DIRECT_SOFTMAX": "0",
        "FP4_CCE_V4_STRICT_FUSED_SPARSE": "1",
        "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_G_CACHE": "auto",
        "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_CHUNK": "1024",
    }


def _nvfp4_v5_actgrad_rht_sr_env() -> dict[str, str]:
    # Regular-TK grouped QKV weight quantization does not implement native
    # weight RHT/SR yet. Keep the v5 recipe on activation/grad extras, which
    # is the supported path that avoids the no-extras grad-norm NaNs.
    return {
        "NVTE_NVFP4_DISABLE_RHT": "0",
        "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_USE_RHT": "1",
        "NVFP4_RHT_ACTIVATION": "1",
        "NVFP4_RHT_GRAD": "1",
        "NVFP4_RHT_WEIGHT": "0",
        "NVFP4_RHT_AXES": "row",
        "NVFP4_USE_STOCHASTIC_ROUNDING": "1",
        "NVFP4_SR_ACTIVATION": "1",
        "NVFP4_SR_GRAD": "1",
        "NVFP4_SR_WEIGHT": "0",
    }


def _tk_v5_recipe_swiglu_env() -> dict[str, str]:
    env = _tk_v5_swiglu_env()
    env.update(_nvfp4_v5_actgrad_rht_sr_env())
    return env


def _delayed_refresh_interval() -> str:
    return os.environ.get("LBT_FP4_DELAYED_REFRESH_INTERVAL", "1")


def _paper_tail_ffn_bf16_layers() -> str:
    return os.environ.get("LBT_FP4_PAPER_TAIL_FFN_BF16_LAYERS", "8")


def _paper_tail_ffn_bf16_env() -> dict[str, str]:
    return {"FP4_KEEP_LAST_N_FFNS_BF16": _paper_tail_ffn_bf16_layers()}


def _mixed_tail_mxfp4_layers() -> str:
    return os.environ.get("LBT_FP4_MIXED_TAIL_LAYERS", "8")


def _tk_v5_delayed_swiglu_env(
    *,
    direct_split: bool = False,
    no_collect: bool = False,
) -> dict[str, str]:
    env = _tk_v5_swiglu_env()
    env.update(
        {
            "USE_TK_FFN_V5_DELAYED_SILU_DERIV": "1",
            "USE_TK_FFN_V5_DELAYED_REFRESH_INTERVAL": _delayed_refresh_interval(),
        }
    )
    if direct_split:
        env["USE_TK_FFN_V5_DELAYED_DIRECT_SPLIT"] = "1"
    if no_collect:
        env["USE_TK_FFN_V5_DELAYED_NO_COLLECT"] = "1"
    return env


def _tk_v5_recipe_delayed_swiglu_env(
    *,
    direct_split: bool = False,
    no_collect: bool = False,
) -> dict[str, str]:
    env = _tk_v5_delayed_swiglu_env(
        direct_split=direct_split,
        no_collect=no_collect,
    )
    env.update(_nvfp4_v5_actgrad_rht_sr_env())
    return env


def _tk_v5_h13_delayed_env(no_collect: bool = False) -> dict[str, str]:
    env = _tk_v5_swiglu_env()
    env.update(
        {
            "USE_TK_FFN_SPLIT_CACHE": "0",
            "USE_TK_FFN_H13_DELAYED_SILU_DERIV": "1",
            "USE_TK_FFN_H13_DELAYED_REFRESH_INTERVAL": _delayed_refresh_interval(),
            "USE_TK_FFN_V5_DELAYED_SILU_DERIV": "0",
        }
    )
    if no_collect:
        env["USE_TK_FFN_H13_DELAYED_NO_COLLECT"] = "1"
    return env


def _tk_v5_highwater_env(
    *,
    delayed: bool = False,
    no_collect: bool = False,
) -> dict[str, str]:
    env = (
        _tk_v5_delayed_swiglu_env(direct_split=True, no_collect=no_collect)
        if delayed
        else _tk_v5_swiglu_env()
    )
    env.update(
        {
            "USE_TK_QKV_NATIVE_SPLIT3_QUANT": "1",
            "USE_TK_QKV_OVERLAP_RMS_WGRAD": "1",
            "USE_TK_QKV_PLAIN_BATCHED_ACCUM_DGRAD": "0",
            "USE_TK_FFN_PLAIN_BATCHED_ACCUM_DGRAD": "1",
            "USE_TK_FFN_FUSED_SUM_RMS": "1",
            "USE_TK_FFN_OVERLAP_RMS_WGRAD": "1",
        }
    )
    return env


def _localcta_v4_swiglu_env() -> dict[str, str]:
    env = _tk_v5_swiglu_env()
    env.update(
        {
            "USE_TK_LOCALCTA": "1",
            "USE_TK_LOCALCTA_VARIANT": "v4",
            "USE_TK_LOCALCTA_FUSED": "0",
            # The compiled default (1493) inflates 8B attention gradients,
            # while 774 leaves a small late-run loss bias. Matched attention
            # and 100-step trainer probes select the native E4M3 range, 448.
            "USE_TK_LOCALCTA_SCALE_NUM": os.environ.get(
                "USE_TK_LOCALCTA_SCALE_NUM", "448"
            ),
            "USE_TK_QKV_FUSED_SUM_RMS": "0",
            "FP4_ATTN_BACKEND": "localcta",
            "FP4_FFN_BACKEND": "localcta",
            # The fast prepared producer currently corrupts the 8B SwiGLU FFN
            # path. Keep it opt-in until the localCTA kernel contract is fixed.
            "USE_TK_LOCALCTA_V4_FAST_PREPARED_PRODUCER": "0",
            "USE_TK_LOCALCTA_V4_ROW_PREPARED_COL_OUTER": "1",
            "USE_TK_LOCALCTA_V4_FAST_FORWARD_GEMM": "1",
            "USE_TK_LOCALCTA_V4_FAST_SINGLE_DGRAD": "1",
            "USE_TK_LOCALCTA_V4_FAST_SINGLE_WGRAD": "1",
            "USE_TK_LOCALCTA_V4_FAST_FFN_RMSNORM_QUANT": "1",
            "USE_TK_LOCALCTA_V4_FAST_W2_WGRAD": "1",
            # Direct W13 gradients are caller-owned final-layout buffers. Keep
            # their GEMM on the caller stream so the returned storage cannot
            # race a side-stream writer.
            "USE_TK_FFN_DISABLE_WGRAD_STREAM": "1",
            "USE_TK_LOCALCTA_V4_FFN_DIRECT_GROUPED_WGRAD_LAYOUT": "1",
            "USE_TK_RMSNORM_BWD_SINGLE_OUT": "1",
            # The one-pass QKV dgrad path regresses 8B gradient parity and is
            # slower than the scale-preserving strided-sum implementation.
            "USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD": "0",
            "USE_TK_QKV_LOCALCTA_FAST_ACT": "1",
            "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "split3",
            "USE_TK_LOCALCTA_V4_QKV_DIRECT_GROUPED_WGRAD_LAYOUT": "1",
            "USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD": "0",
            "USE_TK_LOCALCTA_V4_FAST_WO_DGRAD": "1",
            "USE_TK_LOCALCTA_V4_FAST_WO_WGRAD": "1",
            "USE_TK_LOCALCTA_V4_FAST_DATA_SR": "1",
            "USE_TK_LOCALCTA_PERSISTENT_STEP_SCRATCH": "1",
            "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW": "1",
            "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_RAW_MULTIPLIER": "2.0",
        }
    )
    return env


def _localcta_v4_h13_delayed_env(no_collect: bool = False) -> dict[str, str]:
    env = _localcta_v4_swiglu_env()
    env.update(
        {
            "USE_TK_FFN_SPLIT_CACHE": "0",
            "USE_TK_FFN_H13_TILE_DELAYED_AMAX": "1",
            "USE_TK_FFN_H13_TILE_DELAYED_REFRESH_INTERVAL": _delayed_refresh_interval(),
            "USE_TK_FFN_H13_DELAYED_REFRESH_INTERVAL": _delayed_refresh_interval(),
        }
    )
    if no_collect:
        env["USE_TK_FFN_H13_TILE_DELAYED_NO_COLLECT"] = "1"
        env["USE_TK_FFN_H13_DELAYED_NO_COLLECT"] = "1"
    return env


def _localcta_v4_delayed_swiglu_env(no_collect: bool = False) -> dict[str, str]:
    env = _localcta_v4_swiglu_env()
    env.update(
        {
            "USE_TK_FFN_LOCALCTA_DELAYED_SPLIT": "1",
            "USE_TK_FFN_LOCALCTA_TILE_DELAYED_AMAX": "1",
            "USE_TK_FFN_LOCALCTA_DELAYED_REFRESH_INTERVAL": _delayed_refresh_interval(),
            "USE_TK_FFN_H13_TILE_DELAYED_AMAX": "1",
            "USE_TK_FFN_H13_TILE_DELAYED_REFRESH_INTERVAL": _delayed_refresh_interval(),
            "USE_TK_LOCALCTA_V4_DELAYED_COLLECT_AMAX": "1",
        }
    )
    if no_collect:
        env["USE_TK_FFN_LOCALCTA_DELAYED_NO_COLLECT"] = "1"
        env["USE_TK_FFN_H13_TILE_DELAYED_NO_COLLECT"] = "1"
        env["USE_TK_LOCALCTA_V4_DELAYED_COLLECT_AMAX"] = "0"
    return env


def _localcta_v4_highwater_env(
    *,
    delayed: bool = False,
    no_collect: bool = False,
) -> dict[str, str]:
    env = (
        _localcta_v4_delayed_swiglu_env(no_collect=no_collect)
        if delayed
        else _localcta_v4_swiglu_env()
    )
    env["LBT_LOCALCTA_V4_PROFILE"] = "highwater"
    env["USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER"] = "1"
    env["USE_TK_LOCALCTA_V4_FUSED_ATOMIC_INIT"] = "1"
    env["USE_TK_LOCALCTA_V4_FUSED_ATOMIC_INIT_THREADS"] = "64"
    env["USE_TK_LOCALCTA_V4_REUSE_ATOMIC_SCRATCH"] = "1"
    env["USE_TK_LOCALCTA_V4_NHSD_REDUCED_WARP_FINALIZE"] = "1"
    env["USE_TK_LOCALCTA_SKIP_SPLIT2_DGRAD_ZERO"] = "0"
    env["USE_TK_QKV_LOCALCTA_WEIGHT_OVERLAP"] = "1"
    return env


def _mxfp4_swiglu_env() -> dict[str, str]:
    return {
        "FP4_MXFP4_ROOT": os.environ.get("FP4_MXFP4_ROOT", str(DEFAULT_FP4_ROOT)),
        "FP4_MATMUL_GEMM_ROOT": os.environ.get(
            "FP4_MATMUL_GEMM_ROOT",
            os.environ.get("FP4_MXFP4_ROOT", str(DEFAULT_FP4_ROOT)),
        ),
        "FP4_CCE_ASSUME_NONEMPTY_LABELS": "1",
        "FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED": "1",
        "FP4_CCE_V4_NVFP4_G_CACHE": "0",
        "FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE": "1",
        "FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE": "0",
        "FP4_CCE_V4_NVFP4_P_DATA_SR": "1",
        "FP4_CCE_V4_NVFP4_P_SCALE_SR": "0",
        "FP4_CCE_V4_NVFP4_P_TARGET_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_P_TOP1_SPLIT": "1",
        "FP4_CCE_V4_NVFP4_STAGED_P_CACHE": "1",
        "FP4_CCE_V4_NVFP4_TILED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_TMA_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_FUSED_STAGED_P_CACHE": "0",
        "FP4_CCE_V4_NVFP4_DIRECT_SOFTMAX": "0",
        "FP4_CCE_V4_STRICT_FUSED_SPARSE": "1",
        "MXFP4_BACKEND_VERSION": "v4",
        "MXFP4_USE_RHT": "0",
        "MXFP4_RHT_ACTIVATION": "0",
        "MXFP4_RHT_GRAD": "0",
        "MXFP4_RHT_WEIGHT": "0",
        "MXFP4_RHT_AXES": "row",
        "MXFP4_RHT_RANDOM_SIGN_MASK": "0",
        "MXFP4_USE_STOCHASTIC_ROUNDING": "0",
        "MXFP4_SR_ACTIVATION": "0",
        "MXFP4_SR_GRAD": "0",
        "MXFP4_SR_WEIGHT": "0",
        "MXFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "MXFP4_SCALE_SR_ACTIVATION": "0",
        "MXFP4_SCALE_SR_GRAD": "0",
        "MXFP4_SCALE_SR_WEIGHT": "0",
        "MXFP4_USE_QKV_ROPE_EPILOGUE": "1",
        "MXFP4_USE_QKV_DIRECT_OUTPUTS": "1",
        "MXFP4_USE_QKV_RMSNORM_QUANT_FUSION": "1",
        # The 8B blog-shape no-AC route benefits from the combined QKV
        # backward path without the earlier lifetime failures.
        "MXFP4_USE_QKV_COMBINED_BWD": "1",
        "MXFP4_USE_SPLIT3_QKV_STAGE_COPY": "1",
        "MXFP4_SPLIT3_QKV_STAGE_COPY_MASK": "qkv",
        "MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD": "0",
        "MXFP4_QKV_BWD_STATE_SLOTS": "4",
        "MXFP4_USE_QKV_BF16_WGRAD": "0",
        # Config 10 is fastest in isolated sweeps for the 8B combined QKV
        # dgrad shape, but the end-to-end trainer showed a clear short-run loss
        # regression. Config 0 keeps the combined QKV path near the same MFU
        # while avoiding the worst numerics drift in the 20-step smoke.
        "MXFP4_QKV_GEMM_CONFIG_M32768_N6144_K4096": "0",
        # Preserve the original single-GPU 8B high-water route. The wait
        # fences help some Bridge lifetimes, but they move this no-AC local
        # run off the reproduced 115%+ MFU path.
        "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM": "0",
        "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM_DGAMMA": "0",
        "MXFP4_USE_QKV_FWD_WEIGHT_QUANT_OVERLAP": "0",
        # W2 quantization has an operation-scoped event and per-device stream;
        # The stream-safe opt-in path is retained for profiling, but it does
        # not beat the serialized schedule at the production 8B shape.
        "MXFP4_USE_FFN_FWD_W2_WEIGHT_QUANT_OVERLAP": "0",
        "MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD": "1",
        "MXFP4_USE_SPLIT2_FFN_INPLACE_QUANT": "1",
        "MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP": "0",
        "MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT": "1",
        "MXFP4_USE_SPLIT2_FFN_PRODUCER_SPLIT": "0",
        "MXFP4_USE_FFN_WGRAD_OVERLAP": "0",
        "MXFP4_FFN_WGRAD_OVERLAP_MIN_M": "32768",
        "MXFP4_USE_BWD_WGRAD_OVERLAP": "0",
        "MXFP4_USE_BWD_STATE_CACHE": "0",
        # Async RMSNorm outputs escape through autograd/FSDP and therefore
        # cannot share persistent launch storage across layers or steps.
        "USE_TK_TRANSIENT_RMSNORM_RETURNS": "1",
        "MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP": "19",
        "MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP_M4096_N2048": "199",
        "MXFP4_USE_RESIDUAL_FUSION": "1",
        "MXFP4_USE_RESIDUAL_FUSION_FFN": "1",
        "MXFP4_USE_RESIDUAL_FUSION_ATTN": "0",
        "MXFP4_ALLOW_UNSAFE_ATTN_FFN_RESIDUAL_OVERLAP": "0",
        "MXFP4_UNSAFE_RESIDUAL_FALLBACK": "prefer_ffn",
        "MXFP4_USE_GEMM_RESIDUAL_KERNEL": "1",
        "MXFP4_USE_FUSED_RMSNORM_QUANT_RHT": "1",
        "MXFP4_USE_FUSED_SILU_FFN_QUANT": "1",
        # Saved-sigmoid is useful for some Bridge/TP routes, but it adds an
        # FFN activation save that makes the local batch-4 8B run OOM and was
        # not part of the old 116% MFU high-water run.
        "MXFP4_USE_SAVED_SIGMOID_FFN": "0",
        "MXFP4_USE_SAVED_SIGMOID_FUSED_SPLIT2_FFN": "0",
        # The fused split2 producer is numerically valid but loses to the
        # existing CUDA derivative-plus-quant path at the 8B TP=2 shape.
        "MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_FFN": "0",
        "MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT": "0",
        "MXFP4_USE_SIMPLE_SQRELU_FUSED_W2": "0",
        "MXFP4_USE_WO_ATTN_LAYOUT": "0",
    }


def _mixed_localcta_mxfp4_env(
    profile: str = "",
    *,
    policy: str | None = None,
    tail_layers: str | None = None,
    model_flavor: str = DEFAULT_MODEL_FLAVOR,
) -> dict[str, str]:
    localcta_env = (
        _localcta_v4_highwater_env()
        if profile == "highwater"
        else _localcta_v4_swiglu_env()
    )
    env = {
        **localcta_env,
        **_mxfp4_swiglu_env(),
        "LBT_FP4_MIXED_POLICY": policy
        if policy is not None
        else os.environ.get("LBT_FP4_MIXED_POLICY", "front_localcta"),
    }
    if tail_layers is not None:
        env["LBT_FP4_MIXED_TAIL_LAYERS"] = str(tail_layers)
    elif os.environ.get("LBT_FP4_MIXED_TAIL_LAYERS"):
        env["LBT_FP4_MIXED_TAIL_LAYERS"] = os.environ["LBT_FP4_MIXED_TAIL_LAYERS"]
    if os.environ.get("LBT_FP4_MIXED_LAYERS"):
        env["LBT_FP4_MIXED_LAYERS"] = os.environ["LBT_FP4_MIXED_LAYERS"]
    if os.environ.get("LBT_FP4_MIXED_SPLIT_LAYER"):
        env["LBT_FP4_MIXED_SPLIT_LAYER"] = os.environ["LBT_FP4_MIXED_SPLIT_LAYER"]
    if profile:
        env["LBT_LOCALCTA_V4_PROFILE"] = profile
    if profile == "highwater":
        env = _localcta_highwater_coda_env(
            env,
            model_flavor=model_flavor,
        )
    return env


def _llama_final_coda_env(
    env: dict[str, str],
    *,
    exact_cde: bool = False,
    exact_cde_wo: bool = False,
) -> dict[str, str]:
    """Apply the retained Llama-8B policy without changing shared recipes."""

    merged = dict(env)
    merged.update(
        {
            "USE_FP4_CODA_EXACT_CDE": "1" if exact_cde else "0",
            "USE_FP4_CODA_H_TILE_RMS": "0",
            "USE_FP4_CODA_EXACT_CDE_WO": "1" if exact_cde_wo else "0",
        }
    )
    return merged


def _localcta_highwater_coda_env(
    env: dict[str, str],
    *,
    model_flavor: str,
) -> dict[str, str]:
    return _llama_final_coda_env(
        env,
        exact_cde=True,
        exact_cde_wo=model_flavor
        in {"8B", "8B_llama3_blog", "8B_nvfp4_paper_proxy"},
    )


def build_cases(model_flavor: str = DEFAULT_MODEL_FLAVOR) -> dict[str, Case]:
    return {
        "bf16": Case(
            name="bf16",
            config="train_configs/nvblog_llama3_8b/bf16.toml",
            env={**_disable_tk_env(), **_safe_reference_attention_env()},
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
        ),
        "bf16_native_nvfp4_cce": Case(
            name="bf16_native_nvfp4_cce",
            config="train_configs/nvblog_llama3_8b/bf16.toml",
            env={
                **_disable_tk_env(),
                **_safe_reference_attention_env(),
                **_native_nvfp4_cce_env(),
            },
            args=_native_nvfp4_cce_args(),
        ),
        "te_nvfp4_f0l4": Case(
            name="te_nvfp4_f0l4",
            config="train_configs/nvblog_llama3_8b/te_nvfp4_f0l4.toml",
            env={
                **_disable_tk_env(),
                **_safe_reference_attention_env(),
                "FP4_KEEP_LAST_N_LAYERS_BF16": "4",
            },
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
            needs_te213=True,
        ),
        "te_nvfp4_full": Case(
            name="te_nvfp4_full",
            config="train_configs/nvblog_llama3_8b/te_nvfp4_full.toml",
            env={**_disable_tk_env(), **_safe_reference_attention_env()},
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
            needs_te213=True,
        ),
        "te_nvfp4_full_native_nvfp4_cce": Case(
            name="te_nvfp4_full_native_nvfp4_cce",
            config="train_configs/nvblog_llama3_8b/te_nvfp4_full.toml",
            env={
                **_disable_tk_env(),
                **_safe_reference_attention_env(),
                **_native_nvfp4_cce_env(),
            },
            args=(
                "--training.local-batch-size",
                "2",
                *_native_nvfp4_cce_args(),
            ),
            needs_te213=True,
        ),
        "te_nvfp4_tail8_ffn": Case(
            name="te_nvfp4_tail8_ffn",
            config="train_configs/nvblog_llama3_8b/te_nvfp4_tail8_ffn.toml",
            env={
                **_disable_tk_env(),
                **_safe_reference_attention_env(),
                **_paper_tail_ffn_bf16_env(),
            },
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
            needs_te213=True,
        ),
        "nvfp4_tk_v5": Case(
            name="nvfp4_tk_v5",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_recipe_swiglu_env(),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_delayed": Case(
            name="nvfp4_tk_v5_delayed",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_recipe_delayed_swiglu_env(),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_split_delayed": Case(
            name="nvfp4_tk_v5_split_delayed",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_delayed_swiglu_env(direct_split=True),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_split_delayed_nocollect": Case(
            name="nvfp4_tk_v5_split_delayed_nocollect",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_delayed_swiglu_env(direct_split=True, no_collect=True),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_h13_delayed": Case(
            name="nvfp4_tk_v5_h13_delayed",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_h13_delayed_env(),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_h13_delayed_nocollect": Case(
            name="nvfp4_tk_v5_h13_delayed_nocollect",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_h13_delayed_env(no_collect=True),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_highwater": Case(
            name="nvfp4_tk_v5_highwater",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_llama_final_coda_env(_tk_v5_highwater_env()),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_highwater_delayed": Case(
            name="nvfp4_tk_v5_highwater_delayed",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_llama_final_coda_env(_tk_v5_highwater_env(delayed=True)),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_highwater_delayed_nocollect": Case(
            name="nvfp4_tk_v5_highwater_delayed_nocollect",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_highwater_env(delayed=True, no_collect=True),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_tk_v5_bf16cce": Case(
            name="nvfp4_tk_v5_bf16cce",
            config="train_configs/nvblog_llama3_8b/nvfp4_tk_v5.toml",
            env=_tk_v5_swiglu_env(),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
        ),
        "nvfp4_localcta_v4": Case(
            name="nvfp4_localcta_v4",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_v4_swiglu_env(),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_delayed": Case(
            name="nvfp4_localcta_v4_delayed",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_v4_delayed_swiglu_env(),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_delayed_nocollect": Case(
            name="nvfp4_localcta_v4_delayed_nocollect",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_v4_delayed_swiglu_env(no_collect=True),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_highwater": Case(
            name="nvfp4_localcta_v4_highwater",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_highwater_coda_env(
                _localcta_v4_highwater_env(),
                model_flavor=model_flavor,
            ),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_highwater_delayed": Case(
            name="nvfp4_localcta_v4_highwater_delayed",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_highwater_coda_env(
                _localcta_v4_highwater_env(delayed=True),
                model_flavor=model_flavor,
            ),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_highwater_delayed_nocollect": Case(
            name="nvfp4_localcta_v4_highwater_delayed_nocollect",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_v4_highwater_env(delayed=True, no_collect=True),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_h13_delayed": Case(
            name="nvfp4_localcta_v4_h13_delayed",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_v4_h13_delayed_env(),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_h13_delayed_nocollect": Case(
            name="nvfp4_localcta_v4_h13_delayed_nocollect",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_v4_h13_delayed_env(no_collect=True),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_fused_split": Case(
            name="nvfp4_localcta_v4_fused_split",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env={**_localcta_v4_swiglu_env(), "LBT_LOCALCTA_V4_PROFILE": "fused_split"},
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "nvfp4_localcta_v4_bf16cce": Case(
            name="nvfp4_localcta_v4_bf16cce",
            config="train_configs/nvblog_llama3_8b/nvfp4_localcta_v4.toml",
            env=_localcta_v4_swiglu_env(),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
        ),
        "mixed_localcta_mxfp4": Case(
            name="mixed_localcta_mxfp4",
            config="train_configs/nvblog_llama3_8b/mixed_localcta_mxfp4.toml",
            env=_mixed_localcta_mxfp4_env(),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "mixed_localcta_mxfp4_fused_split": Case(
            name="mixed_localcta_mxfp4_fused_split",
            config="train_configs/nvblog_llama3_8b/mixed_localcta_mxfp4.toml",
            env=_mixed_localcta_mxfp4_env("fused_split"),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "mixed_localcta_mxfp4_tail_mxfp4": Case(
            name="mixed_localcta_mxfp4_tail_mxfp4",
            config="train_configs/nvblog_llama3_8b/mixed_localcta_mxfp4.toml",
            env=_mixed_localcta_mxfp4_env(
                policy="tail_mxfp4",
                tail_layers=_mixed_tail_mxfp4_layers(),
            ),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "mixed_localcta_mxfp4_tail_mxfp4_highwater": Case(
            name="mixed_localcta_mxfp4_tail_mxfp4_highwater",
            config="train_configs/nvblog_llama3_8b/mixed_localcta_mxfp4.toml",
            env=_mixed_localcta_mxfp4_env(
                "highwater",
                policy="tail_mxfp4",
                tail_layers=_mixed_tail_mxfp4_layers(),
                model_flavor=model_flavor,
            ),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "mixed_localcta_mxfp4_highwater": Case(
            name="mixed_localcta_mxfp4_highwater",
            config="train_configs/nvblog_llama3_8b/mixed_localcta_mxfp4.toml",
            env=_mixed_localcta_mxfp4_env(
                "highwater",
                model_flavor=model_flavor,
            ),
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
        ),
        "mxfp4": Case(
            name="mxfp4",
            config="train_configs/nvblog_llama3_8b/mxfp4_tk.toml",
            env=_llama_final_coda_env(
                {
                    **_mxfp4_swiglu_env(),
                    **_native_nvfp4_cce_env(),
                },
                exact_cde=True,
            ),
            args=_native_nvfp4_cce_args(),
        ),
    }


def _logged_env_key(key: str) -> bool:
    prefixes = (
        "CUDA_VISIBLE_DEVICES",
        "FP4",
        "LOW_BITS",
        "LBT_",
        "MXFP4",
        "NVFP4",
        "NVTE",
        "PURE_TE",
        "TORCH_CUDNN_SDPA_ENABLED",
        "USE_",
        "WANDB_MODE",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
    )
    return any(key.startswith(prefix) for prefix in prefixes)


def _maybe_download_hf_assets(path: Path, *, download: bool) -> None:
    tokenizer_file = path / "tokenizer.json"
    if tokenizer_file.exists():
        return
    if not download:
        raise SystemExit(
            f"Llama 3 8B assets are missing at {path}. Populate that directory "
            "or rerun with --download-hf-assets after exporting HF_TOKEN."
        )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("--download-hf-assets requires HF_TOKEN in the environment.")
    from huggingface_hub import snapshot_download

    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="meta-llama/Meta-Llama-3-8B",
        local_dir=path,
        token=token,
        allow_patterns=[
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "config.json",
        ],
    )


def _rows_csv(rows: list[dict[str, float | int]]) -> str:
    if not rows:
        return ""
    fields = ["step", "loss", "grad_norm", "memory_gib", "tps", "tflops", "mfu"]
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _validation_rows_csv(rows: list[dict[str, float | int]]) -> str:
    if not rows:
        return ""
    fields = ["step", "loss", "memory_gib", "tps"]
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _build_train_command(
    case: Case,
    *,
    out_dir: Path,
    steps: int,
    log_freq: int,
    nproc_per_node: int,
    seed: int | None,
    config_overrides: tuple[str, ...],
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc_per_node),
        "--standalone",
        "train.py",
        "--job.config-file",
        case.config,
        "--job.dump-folder",
        str(out_dir / "dump"),
        "--training.steps",
        str(steps),
        "--metrics.log-freq",
        str(log_freq),
        "--metrics.disable-color-printing",
        *case.args,
        *config_overrides,
    ]
    if seed is not None:
        cmd.extend(["--debug.seed", str(seed)])
    return cmd


def run_case(
    case: Case,
    *,
    out_base: Path,
    gpu: str,
    steps: int,
    log_freq: int,
    nproc_per_node: int,
    seed: int | None,
    fp4_root: Path,
    te213_stage: Path,
    te213_lib: Path,
    steady_from: int,
    config_overrides: tuple[str, ...],
    extra_env: dict[str, str],
) -> dict[str, object]:
    out_dir = out_base / case.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    env = os.environ.copy()
    visible_devices = _visible_devices_for_nproc(str(gpu), nproc_per_node)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "WANDB_MODE": env.get("WANDB_MODE", "disabled"),
            "PYTHONUNBUFFERED": "1",
            "FP4_MATMUL_ROOT": str(fp4_root),
            "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
            "TORCH_CUDNN_SDPA_ENABLED": "1",
            "NVTE_FUSED_ATTN": "0",
        }
    )
    env.update(case.env)
    for root_key in ("FP4_MXFP4_ROOT", "FP4_MATMUL_GEMM_ROOT"):
        if root_key in case.env and root_key not in os.environ:
            env[root_key] = str(fp4_root)
    env.update(extra_env)
    if case.needs_te213:
        env["PYTHONPATH"] = f"{te213_stage}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["LD_LIBRARY_PATH"] = f"{te213_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    cmd = _build_train_command(
        case,
        out_dir=out_dir,
        steps=steps,
        log_freq=log_freq,
        nproc_per_node=nproc_per_node,
        seed=seed,
        config_overrides=config_overrides,
    )

    header = {
        "case": case.name,
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "env": {k: env[k] for k in sorted(env) if _logged_env_key(k)},
    }
    (out_dir / "run.json").write_text(json.dumps(header, indent=2) + "\n")

    print(f"\n=== {case.name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(header, indent=2) + "\n")
        log.flush()
        start = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        wall_s = time.time() - start

    log_text = log_path.read_text(errors="ignore")
    rows = parse_step_metrics(log_text)
    validation_rows = parse_validation_metrics(log_text)
    summary = summarize_rows(rows, steady_from)
    numerics_ok = bool(rows) and all(
        math.isfinite(float(row["loss"])) and math.isfinite(float(row["grad_norm"]))
        for row in rows
    )
    summary.update(summarize_validation(validation_rows))
    summary.update(
        {
            "case": case.name,
            "returncode": rc,
            "process_completed": rc == 0 and bool(rows) and int(rows[-1]["step"]) >= steps,
            "numerics_ok": numerics_ok,
            "completed": rc == 0 and bool(rows) and int(rows[-1]["step"]) >= steps and numerics_ok,
            "wall_s": wall_s,
            "log": str(log_path),
        }
    )
    (out_dir / "steps.csv").write_text(_rows_csv(rows))
    (out_dir / "validation.csv").write_text(_validation_rows_csv(validation_rows))
    return summary


def write_summary(out_base: Path, summaries: list[dict[str, object]], steady_from: int) -> None:
    (out_base / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fields = [
        "case",
        "completed",
        "process_completed",
        "numerics_ok",
        "returncode",
        "last_step",
        "last_loss",
        "peak_tps",
        "peak_mfu",
        "steady_tps",
        "steady_mfu",
        "validation_points",
        "last_validation_loss",
        "wall_s",
        "log",
    ]
    with (out_base / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})
    lines = [
        "# 8B NVFP4 Paper-Proxy Local Matrix",
        "",
        f"Steady window starts at logged step >= {steady_from}.",
        "",
        "| case | done | numerics | last step | train loss | val loss | peak MFU | steady MFU | steady tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {case} | {done} | {numerics} | {step} | {loss} | {val} | {peak} | {steady} | {tps} |".format(
                case=row["case"],
                done="yes" if row.get("completed") else "no",
                numerics="ok" if row.get("numerics_ok") else "bad",
                step=_fmt(row.get("last_step"), "{:.0f}"),
                loss=_fmt(row.get("last_loss"), "{:.4f}"),
                val=_fmt(row.get("last_validation_loss"), "{:.4f}"),
                peak=_fmt(row.get("peak_mfu"), "{:.2f}"),
                steady=_fmt(row.get("steady_mfu"), "{:.2f}"),
                tps=_fmt(row.get("steady_tps"), "{:.0f}"),
            )
        )
    lines.append("")
    lines.append("Logs:")
    for row in summaries:
        lines.append(f"- {row['case']}: `{row['log']}`")
    (out_base / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def _fmt(value: object, fmt: str) -> str:
    if value is None:
        return "-"
    return fmt.format(float(value))


def _same_resolved_path(lhs: Path, rhs: Path) -> bool:
    try:
        return lhs.expanduser().resolve() == rhs.expanduser().resolve()
    except OSError:
        return lhs.expanduser().absolute() == rhs.expanduser().absolute()


def _check_gpu_selection(gpu_spec: str, *, allow_gpu2: bool) -> None:
    selected = {item.strip() for item in gpu_spec.split(",") if item.strip()}
    blocked = selected & DEFAULT_BLACKLISTED_GPUS
    if blocked and not allow_gpu2:
        raise SystemExit(
            "Refusing to run on blacklisted GPU(s) "
            f"{','.join(sorted(blocked))}. GPU2 has inconsistent clocks for "
            "these MFU comparisons; use --allow-gpu2 only for explicit "
            "diagnostics."
        )


def _visible_devices_for_nproc(gpu_spec: str, nproc_per_node: int) -> str:
    selected = [item.strip() for item in gpu_spec.split(",") if item.strip()]
    if nproc_per_node <= 1 or len(selected) != 1:
        return ",".join(selected)

    try:
        start = int(selected[0])
    except ValueError:
        return ",".join(selected)
    return ",".join(str(start + offset) for offset in range(nproc_per_node))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--steady-from", type=int, default=20)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--allow-gpu2",
        action="store_true",
        help="Permit GPU2 despite the local benchmark blacklist.",
    )
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-debug-seed", action="store_true")
    parser.add_argument("--fp4-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument("--te213-stage", type=Path, default=DEFAULT_TE213_STAGE)
    parser.add_argument("--te213-lib", type=Path, default=DEFAULT_TE213_LIB)
    parser.add_argument("--hf-assets-path", type=Path, default=DEFAULT_HF_ASSETS)
    parser.add_argument(
        "--model-flavor",
        default=DEFAULT_MODEL_FLAVOR,
        help=(
            "TorchTitan model flavor. Defaults to the NVFP4 paper proxy; use "
            "8B_llama3_blog for the older Llama 3 8B throughput shape."
        ),
    )
    parser.add_argument("--download-hf-assets", action="store_true")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--validation-data", type=Path, default=DEFAULT_VALIDATION_DATA)
    parser.add_argument("--validation-enable", action="store_true")
    parser.add_argument("--validation-freq", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=2)
    parser.add_argument("--validation-local-batch-size", type=int, default=1)
    parser.add_argument(
        "--delayed-refresh-interval",
        type=int,
        default=int(os.environ.get("LBT_FP4_DELAYED_REFRESH_INTERVAL", "1")),
        help="Steps between delayed-scaling amax refreshes for v5/v4 delayed cases.",
    )
    parser.add_argument(
        "--mixed-tail-mxfp4-layers",
        type=int,
        default=int(os.environ.get("LBT_FP4_MIXED_TAIL_LAYERS", "8")),
        help="Number of final layers routed to MXFP4 in mixed tail-MXFP4 cases.",
    )
    parser.add_argument(
        "--activation-checkpoint-mode",
        default="none",
        help="Activation checkpoint mode passed to train.py; defaults to no AC for MFU runs.",
    )
    parser.add_argument(
        "--validation-load-dataset-kwargs",
        default='{"num_workers":0,"pin_memory":false,"repeat":false}',
    )
    parser.add_argument(
        "--load-dataset-kwargs",
        default='{"num_workers":8,"prefetch_factor":4,"pin_memory":false,"repeat":false,"require_full_run":true}',
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied after the selected case defaults. May be repeated.",
    )
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra train.py argument appended after the selected case defaults. May be repeated.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "te_nvfp4_tail8_ffn",
            "nvfp4_tk_v5",
            "nvfp4_tk_v5_delayed",
            "nvfp4_localcta_v4",
            "nvfp4_localcta_v4_delayed",
            "mxfp4",
            "mixed_localcta_mxfp4_tail_mxfp4",
        ],
    )
    parser.add_argument("--out-base", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.delayed_refresh_interval < 0:
        raise SystemExit("--delayed-refresh-interval must be >= 0")
    if args.mixed_tail_mxfp4_layers < 0:
        raise SystemExit("--mixed-tail-mxfp4-layers must be >= 0")
    os.environ["LBT_FP4_DELAYED_REFRESH_INTERVAL"] = str(args.delayed_refresh_interval)
    os.environ["LBT_FP4_MIXED_TAIL_LAYERS"] = str(args.mixed_tail_mxfp4_layers)
    visible_devices = _visible_devices_for_nproc(args.gpu, args.nproc_per_node)
    _check_gpu_selection(visible_devices, allow_gpu2=args.allow_gpu2)
    _maybe_download_hf_assets(args.hf_assets_path, download=args.download_hf_assets)
    if not args.train_data.exists():
        raise SystemExit(f"Training packed dataset not found: {args.train_data}")
    if args.validation_enable and not args.validation_data.exists():
        raise SystemExit(f"Validation packed dataset not found: {args.validation_data}")
    if args.validation_enable and _same_resolved_path(args.train_data, args.validation_data):
        raise SystemExit(
            "Validation dataset must be held out: --validation-data matches --train-data."
        )

    cases = build_cases(args.model_flavor)
    unknown = [case for case in args.cases if case not in cases]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    if any(cases[name].needs_te213 for name in args.cases):
        if not args.te213_stage.exists() or not args.te213_lib.exists():
            raise SystemExit(
                f"TE 2.13 stage/lib missing: {args.te213_stage} / {args.te213_lib}"
            )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = args.out_base or Path(f"/tmp/lbt_nvblog_llama3_8b_{args.steps}_{stamp}")
    out_base.mkdir(parents=True, exist_ok=True)

    config_overrides = [
        "--model.hf-assets-path",
        str(args.hf_assets_path),
        "--model.flavor",
        args.model_flavor,
        "--activation_checkpoint.mode",
        args.activation_checkpoint_mode,
        "--training.dataset",
        "packed-bin",
        "--training.dataset-path",
        str(args.train_data),
        "--training.load-dataset-kwargs",
        args.load_dataset_kwargs,
    ]
    config_overrides.extend(args.config_override)
    extra_env = {}
    if args.validation_enable:
        config_overrides.extend(
            [
                "--validation.enable",
                "--validation.dataset",
                "packed-bin",
                "--validation.dataset-path",
                str(args.validation_data),
                "--validation.local-batch-size",
                str(args.validation_local_batch_size),
                "--validation.seq-len",
                "8192",
                "--validation.freq",
                str(args.validation_freq),
                "--validation.steps",
                str(args.validation_steps),
            ]
        )
        extra_env["LBT_VALIDATION_LOAD_DATASET_KWARGS"] = (
            args.validation_load_dataset_kwargs
        )
    for item in args.extra_env:
        if "=" not in item:
            raise SystemExit(f"--extra-env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise SystemExit(f"--extra-env key cannot be empty: {item}")
        extra_env[key] = value

    summaries = []
    seed = None if args.no_debug_seed else args.seed
    for name in args.cases:
        summary = run_case(
            cases[name],
            out_base=out_base,
            gpu=str(args.gpu),
            steps=args.steps,
            log_freq=args.log_freq,
            nproc_per_node=args.nproc_per_node,
            seed=seed,
            fp4_root=args.fp4_root,
            te213_stage=args.te213_stage,
            te213_lib=args.te213_lib,
            steady_from=args.steady_from,
            config_overrides=tuple(config_overrides),
            extra_env=extra_env,
        )
        summaries.append(summary)
        write_summary(out_base, summaries, args.steady_from)
        print(
            f"[{name}] rc={summary['returncode']} last_step={summary.get('last_step')} "
            f"loss={summary.get('last_loss')} val={summary.get('last_validation_loss')} "
            f"peak_mfu={summary.get('peak_mfu')} steady_mfu={summary.get('steady_mfu')} "
            f"log={summary['log']}",
            flush=True,
        )
        if summary["returncode"] != 0:
            print(f"Stopping after failed case {name}", file=sys.stderr)
            return int(summary["returncode"])
    write_summary(out_base, summaries, args.steady_from)
    print(f"\nsummary: {out_base / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
