#!/usr/bin/env python3
"""Run local exact Nemotron-H 8B BF16/FP4 smoke comparisons.

This uses the TorchTitan wrapper registered as `nemotron_h_gc:8B_paper`.
It is intentionally separate from `run_nvblog_llama3_8b_matrix.py`, whose
optimized localCTA/MXFP4 cases are Llama-block fusions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HF_ASSETS = Path(
    os.environ.get(
        "LBT_NEMOTRON_H_HF_ASSETS",
        "./assets/hf/NVIDIA-Nemotron-Nano-12B-v2-Base",
    )
)
DEFAULT_TRAIN_DATA = Path(
    os.environ.get(
        "LBT_NEMOTRON_H_TRAIN_DATA",
        "/tmp/lbt_packed/slimpajama_1b_tokens_20260529",
    )
)
DEFAULT_TE213_STAGE = Path("/tmp/te213_stage")
DEFAULT_TE213_LIB = DEFAULT_TE213_STAGE / "nvidia" / "cu13" / "lib"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nvpaper_1p2b_numerics_500 import parse_step_metrics, summarize_rows  # noqa: E402
from run_nvblog_llama3_8b_matrix import (  # noqa: E402
    DEFAULT_BLACKLISTED_GPUS,
    _check_gpu_selection,
    _fmt,
    _localcta_v4_highwater_env,
    _mixed_localcta_mxfp4_env,
    _mixed_tail_mxfp4_layers,
    _mxfp4_swiglu_env,
    _tk_v5_swiglu_env,
    _visible_devices_for_nproc,
)


@dataclass(frozen=True)
class Case:
    name: str
    config: str
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()
    needs_te213: bool = False


def _native_nvfp4_cce_env(env: dict[str, str]) -> dict[str, str]:
    merged = dict(env)
    merged.setdefault("FP4_CCE_ASSUME_NONEMPTY_LABELS", "1")
    merged.setdefault("FP4_CCE_NVFP4_EXACT_NORM_QUANT", "0")
    merged.setdefault("FP4_CCE_V4_FUSED_X_PRODUCER", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_G_CACHE", "0")
    merged.setdefault("FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE", "0")
    merged.setdefault("FP4_CCE_V4_NVFP4_P_DATA_SR", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_P_SCALE_SR", "0")
    merged.setdefault("FP4_CCE_V4_NVFP4_P_TARGET_SPLIT", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_P_TOP1_SPLIT", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_STAGED_P_CACHE", "1")
    merged.setdefault("FP4_CCE_V4_NVFP4_TILED_P_CACHE", "0")
    merged.setdefault("FP4_CCE_V4_NVFP4_TMA_P_CACHE", "0")
    merged.setdefault("FP4_CCE_V4_NVFP4_FUSED_STAGED_P_CACHE", "0")
    merged.setdefault("FP4_CCE_V4_NVFP4_DIRECT_SOFTMAX", "0")
    merged.setdefault("FP4_CCE_V4_STRICT_FUSED_SPARSE", "1")
    return merged


def _nemotron_tail_fused_env(env: dict[str, str], **extra: str) -> dict[str, str]:
    merged = _native_nvfp4_cce_env(env)
    merged.setdefault("FP4_KEEP_LAST_N_LAYERS_BF16", "8")
    merged.setdefault("USE_FP4_SQRELU_FFN_TK", "1")
    merged.update(extra)
    return merged


def _nemotron_projection_env(
    env: dict[str, str],
    *,
    fp4_head: bool,
    mamba_in: bool = True,
    mamba_out: bool,
    keep_last_layers_bf16: int,
    fused_mamba_gated_out: bool = False,
    **extra: str,
) -> dict[str, str]:
    return _nemotron_tail_fused_env(
        env,
        FP4_KEEP_LAST_N_LAYERS_BF16=str(keep_last_layers_bf16),
        LBT_NEMOTRON_H_FP4_ATTENTION_PROJ="1",
        LBT_NEMOTRON_H_FP4_MAMBA_IN_PROJ="1" if mamba_in else "0",
        LBT_NEMOTRON_H_FP4_MAMBA_OUT_PROJ="1" if mamba_out else "0",
        LBT_NEMOTRON_H_FUSED_MAMBA_GATED_OUT_PROJ=(
            "1" if mamba_out and fused_mamba_gated_out else "0"
        ),
        LBT_NEMOTRON_H_REQUIRE_FUSED_MAMBA_GATED_OUT_PROJ=(
            "1" if mamba_out and fused_mamba_gated_out else "0"
        ),
        LBT_NEMOTRON_H_FP4_OUTPUT_HEAD="1" if fp4_head else "0",
        **extra,
    )


def _nemotron_native_ssd_env(env: dict[str, str], **extra: str) -> dict[str, str]:
    merged = dict(env)
    merged.update(
        {
            "LBT_NEMOTRON_H_CUTLASS_SSD": "1",
            "LBT_NEMOTRON_H_GROUPED_SSD_BWD": "1",
            "LBT_NEMOTRON_H_ADJACENT_CONV_GRADS": "1",
        }
    )
    merged.update(extra)
    return merged


def _nemotron_sqrelu_overlap_env(env: dict[str, str]) -> dict[str, str]:
    merged = dict(env)
    merged.update(
        {
            "USE_TK_SQRELU_FFN_OVERLAP_W1_WGRAD_RMS": "1",
            "USE_TK_SQRELU_FFN_OVERLAP_W2_WGRAD_DERIV": "0",
            "USE_TK_SQRELU_FFN_CACHED_RMS_BWD": "0",
        }
    )
    return merged


def _nvfp4_tk_v5_ssd_env(*, all_linear: bool = False) -> dict[str, str]:
    return _nemotron_native_ssd_env(
        _nemotron_sqrelu_overlap_env(
            _nemotron_projection_env(
                _tk_v5_swiglu_env(),
                fp4_head=True,
                mamba_in=all_linear,
                mamba_out=all_linear,
                fused_mamba_gated_out=all_linear,
                keep_last_layers_bf16=0,
                LBT_NEMOTRON_H_FUSED_MAMBA_RMS_IN_PROJ=(
                    "1" if all_linear else "0"
                ),
            )
        )
    )


def _localcta_v4_ssd_env(
    *,
    keep_last_layers_bf16: int,
    all_linear: bool = False,
    interlayer_cde: bool = False,
) -> dict[str, str]:
    return _nemotron_native_ssd_env(
        _nemotron_sqrelu_overlap_env(
            _nemotron_projection_env(
                _localcta_v4_highwater_env(),
                fp4_head=True,
                mamba_in=all_linear,
                mamba_out=all_linear,
                keep_last_layers_bf16=keep_last_layers_bf16,
                LBT_NEMOTRON_H_FUSED_MAMBA_RMS_IN_PROJ=(
                    "1" if all_linear else "0"
                ),
                USE_TK_LOCALCTA_V4_MAMBA_OUT_WEIGHT_QUANT_OVERLAP=(
                    "1" if all_linear else "0"
                ),
                USE_TK_LOCALCTA_V4_SQRELU_W2_WEIGHT_QUANT_OVERLAP=(
                    "1" if all_linear else "0"
                ),
                USE_FP4_NEMOTRON_INTERLAYER_CDE=(
                    "1" if interlayer_cde else "0"
                ),
            )
        )
    )


def _mxfp4_all_linear_ssd_env() -> dict[str, str]:
    return _nemotron_native_ssd_env(
        _nemotron_projection_env(
            _mxfp4_swiglu_env(),
            fp4_head=True,
            mamba_in=True,
            mamba_out=True,
            keep_last_layers_bf16=0,
            MXFP4_USE_RMS_BWD_SPLIT_DGAMMA="0",
            MXFP4_USE_BATCHED_NEMOTRON_PADDING="1",
        )
    )


def _mxfp4_all_linear_ssd_b4_env() -> dict[str, str]:
    return _fused_tiled_g_cache_env(_mxfp4_all_linear_ssd_env())


def _fused_tiled_g_cache_env(env: dict[str, str]) -> dict[str, str]:
    merged = dict(env)
    merged.update(
        {
            "FP4_CCE_V4_NVFP4_CHUNKED_LOGITS_G_CACHE": "0",
            "FP4_CCE_V4_NVFP4_FUSED_G_CACHE": "1",
            "FP4_CCE_V4_NVFP4_FUSED_G_CACHE_IMPL": "tiled",
        }
    )
    return merged


def _mixed_nemotron_ssd_env(
    *,
    policy: str,
    tail_layers: str | None = None,
    sqrelu_w1_overlap: bool = True,
) -> dict[str, str]:
    env = _mixed_localcta_mxfp4_env(
        "highwater",
        policy=policy,
        tail_layers=tail_layers,
    )
    env = _nemotron_projection_env(
        env,
        fp4_head=True,
        mamba_in=True,
        mamba_out=True,
        keep_last_layers_bf16=0,
        LBT_FP4_MIXED_MAMBA_BACKEND="mxfp4",
        LBT_FP4_MIXED_HEAD_BACKEND="mxfp4",
        LBT_NEMOTRON_H_FUSED_MAMBA_RMS_IN_PROJ="1",
        MXFP4_USE_RMS_BWD_SPLIT_DGAMMA="0",
        MXFP4_USE_BATCHED_NEMOTRON_PADDING="1",
    )
    env = _nemotron_native_ssd_env(_nemotron_sqrelu_overlap_env(env))
    if not sqrelu_w1_overlap:
        env["USE_TK_SQRELU_FFN_OVERLAP_W1_WGRAD_RMS"] = "0"
    return env


def build_cases() -> dict[str, Case]:
    return {
        "bf16": Case(
            name="bf16",
            config="train_configs/nvpaper_nemotron_h_8b_bf16.toml",
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
        ),
        "bf16_native_nvfp4_cce": Case(
            name="bf16_native_nvfp4_cce",
            config="train_configs/nvpaper_nemotron_h_8b_bf16.toml",
            env=_native_nvfp4_cce_env({}),
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
        "te_nvfp4": Case(
            name="te_nvfp4",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env=_native_nvfp4_cce_env(
                {
                    "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                    "TORCH_CUDNN_SDPA_ENABLED": "1",
                }
            ),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
            needs_te213=True,
        ),
        "te_nvfp4_triton_cce": Case(
            name="te_nvfp4_triton_cce",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env={
                "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "1",
            },
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "triton_bf16"),
            needs_te213=True,
        ),
        "te_mxfp4": Case(
            name="te_mxfp4",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env={
                "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "1",
            },
            args=(
                "--te-fp4.mlp-recipe",
                "MXFP4",
                "--te-fp4.attn-recipe",
                "MXFP4",
                "--te-fp4.mamba-recipe",
                "MXFP4",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
            needs_te213=True,
        ),
        "te_nvfp4_no_mamba": Case(
            name="te_nvfp4_no_mamba",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env={
                "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "1",
            },
            args=(
                "--te-fp4.mamba-recipe",
                "None",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
            needs_te213=True,
        ),
        "te_nvfp4_mamba_only": Case(
            name="te_nvfp4_mamba_only",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env={
                "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "1",
            },
            args=(
                "--te-fp4.mlp-recipe",
                "None",
                "--te-fp4.attn-recipe",
                "None",
                "--te-fp4.mamba-recipe",
                "NVFP4",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
            needs_te213=True,
        ),
        "te_nvfp4_mlp_only": Case(
            name="te_nvfp4_mlp_only",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env={
                "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "1",
            },
            args=(
                "--te-fp4.attn-recipe",
                "None",
                "--te-fp4.mamba-recipe",
                "None",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
            needs_te213=True,
        ),
        "te_nvfp4_attn_only": Case(
            name="te_nvfp4_attn_only",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env={
                "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "1",
            },
            args=(
                "--te-fp4.mlp-recipe",
                "None",
                "--te-fp4.mamba-recipe",
                "None",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
            needs_te213=True,
        ),
        "te_nvfp4_mamba_mxfp4": Case(
            name="te_nvfp4_mamba_mxfp4",
            config="train_configs/nvpaper_nemotron_h_8b_te_nvfp4.toml",
            env={
                "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
                "TORCH_CUDNN_SDPA_ENABLED": "1",
            },
            args=(
                "--te-fp4.mamba-recipe",
                "MXFP4",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
            needs_te213=True,
        ),
        "nvfp4_custom": Case(
            name="nvfp4_custom",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_custom.toml",
            env=_nemotron_tail_fused_env(
                _tk_v5_swiglu_env(),
                NVFP4_USE_RHT="0",
                NVFP4_USE_STOCHASTIC_ROUNDING="0",
            ),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "nvfp4_tk_v5_ssd": Case(
            name="nvfp4_tk_v5_ssd",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_tk_v5.toml",
            env=_nvfp4_tk_v5_ssd_env(),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "nvfp4_tk_v5_ssd_b2": Case(
            name="nvfp4_tk_v5_ssd_b2",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_tk_v5.toml",
            env=_nvfp4_tk_v5_ssd_env(),
            args=(
                "--training.local-batch-size",
                "2",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "nvfp4_tk_v5_all_linear_ssd_experimental": Case(
            name="nvfp4_tk_v5_all_linear_ssd_experimental",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_tk_v5.toml",
            env=_nvfp4_tk_v5_ssd_env(all_linear=True),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "nvfp4_tk_v5_all_linear_ssd_b2": Case(
            name="nvfp4_tk_v5_all_linear_ssd_b2",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_tk_v5.toml",
            env=_nvfp4_tk_v5_ssd_env(all_linear=True),
            args=(
                "--training.local-batch-size",
                "2",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "nvfp4_tk_v5_all_linear_ssd_b3": Case(
            name="nvfp4_tk_v5_all_linear_ssd_b3",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_tk_v5.toml",
            env=_nvfp4_tk_v5_ssd_env(all_linear=True),
            args=(
                "--training.local-batch-size",
                "3",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "nvfp4_custom_no_mamba": Case(
            name="nvfp4_custom_no_mamba",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_custom.toml",
            env={
                "NVFP4_USE_RHT": "0",
                "NVFP4_USE_STOCHASTIC_ROUNDING": "0",
            },
            args=(
                "--mxfp-custom.mamba-recipe",
                "None",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
        ),
        "nvfp4_custom_mamba_only": Case(
            name="nvfp4_custom_mamba_only",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_custom.toml",
            env={
                "NVFP4_USE_RHT": "0",
                "NVFP4_USE_STOCHASTIC_ROUNDING": "0",
            },
            args=(
                "--mxfp-custom.mlp-recipe",
                "None",
                "--mxfp-custom.attn-recipe",
                "None",
                "--mxfp-custom.mamba-recipe",
                "Custom",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
        ),
        "nvfp4_localcta_v4": Case(
            name="nvfp4_localcta_v4",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_localcta_v4.toml",
            env=_localcta_v4_ssd_env(keep_last_layers_bf16=8),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "nvfp4_localcta_v4_fp4_tail": Case(
            name="nvfp4_localcta_v4_fp4_tail",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_localcta_v4.toml",
            env=_localcta_v4_ssd_env(keep_last_layers_bf16=0),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "nvfp4_localcta_v4_fp4_tail_b2": Case(
            name="nvfp4_localcta_v4_fp4_tail_b2",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_localcta_v4.toml",
            env=_localcta_v4_ssd_env(keep_last_layers_bf16=0),
            args=(
                "--training.local-batch-size",
                "2",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "nvfp4_localcta_v4_all_linear": Case(
            name="nvfp4_localcta_v4_all_linear",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_localcta_v4.toml",
            env=_nemotron_projection_env(
                _localcta_v4_highwater_env(),
                fp4_head=True,
                mamba_out=True,
                keep_last_layers_bf16=0,
            ),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "nvfp4_localcta_v4_all_linear_ssd_b2": Case(
            name="nvfp4_localcta_v4_all_linear_ssd_b2",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_localcta_v4.toml",
            env=_localcta_v4_ssd_env(
                keep_last_layers_bf16=0,
                all_linear=True,
            ),
            args=(
                "--training.local-batch-size",
                "2",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "nvfp4_localcta_v4_all_linear_ssd_b3": Case(
            name="nvfp4_localcta_v4_all_linear_ssd_b3",
            config="train_configs/nvpaper_nemotron_h_8b_nvfp4_localcta_v4.toml",
            env=_localcta_v4_ssd_env(
                keep_last_layers_bf16=0,
                all_linear=True,
                interlayer_cde=True,
            ),
            args=(
                "--training.local-batch-size",
                "3",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "mxfp4_tk": Case(
            name="mxfp4_tk",
            config="train_configs/nvpaper_nemotron_h_8b_mxfp4_tk.toml",
            env=_nemotron_tail_fused_env(_mxfp4_swiglu_env()),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "mxfp4_tk_fp4_tail": Case(
            name="mxfp4_tk_fp4_tail",
            config="train_configs/nvpaper_nemotron_h_8b_mxfp4_tk.toml",
            env=_nemotron_projection_env(
                _mxfp4_swiglu_env(),
                fp4_head=True,
                mamba_out=False,
                keep_last_layers_bf16=0,
            ),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "mxfp4_tk_all_linear": Case(
            name="mxfp4_tk_all_linear",
            config="train_configs/nvpaper_nemotron_h_8b_mxfp4_tk.toml",
            env=_nemotron_projection_env(
                _mxfp4_swiglu_env(),
                fp4_head=True,
                mamba_out=True,
                keep_last_layers_bf16=0,
            ),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "mxfp4_tk_all_linear_ssd": Case(
            name="mxfp4_tk_all_linear_ssd",
            config="train_configs/nvpaper_nemotron_h_8b_mxfp4_tk.toml",
            env=_mxfp4_all_linear_ssd_env(),
            args=("--fp4-cce.enabled", "--fp4-cce.backend", "nvfp4"),
        ),
        "mxfp4_tk_all_linear_ssd_b2": Case(
            name="mxfp4_tk_all_linear_ssd_b2",
            config="train_configs/nvpaper_nemotron_h_8b_mxfp4_tk.toml",
            env=_mxfp4_all_linear_ssd_env(),
            args=(
                "--training.local-batch-size",
                "2",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "mxfp4_tk_all_linear_ssd_b3": Case(
            name="mxfp4_tk_all_linear_ssd_b3",
            config="train_configs/nvpaper_nemotron_h_8b_mxfp4_tk.toml",
            env=_mxfp4_all_linear_ssd_env(),
            args=(
                "--training.local-batch-size",
                "3",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "mxfp4_tk_all_linear_ssd_b4": Case(
            name="mxfp4_tk_all_linear_ssd_b4",
            config="train_configs/nvpaper_nemotron_h_8b_mxfp4_tk.toml",
            env=_mxfp4_all_linear_ssd_b4_env(),
            args=(
                "--training.local-batch-size",
                "4",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "mixed_localcta_mxfp4_mamba_mx_b2": Case(
            name="mixed_localcta_mxfp4_mamba_mx_b2",
            config="train_configs/nvpaper_nemotron_h_8b_mixed_localcta_mxfp4.toml",
            env=_mixed_nemotron_ssd_env(policy="all_localcta"),
            args=(
                "--training.local-batch-size",
                "2",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "mixed_localcta_mxfp4_tail_mxfp4_b2": Case(
            name="mixed_localcta_mxfp4_tail_mxfp4_b2",
            config="train_configs/nvpaper_nemotron_h_8b_mixed_localcta_mxfp4.toml",
            env=_mixed_nemotron_ssd_env(
                policy="tail_mxfp4",
                tail_layers=_mixed_tail_mxfp4_layers(),
            ),
            args=(
                "--training.local-batch-size",
                "2",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
        "mixed_localcta_mxfp4_tail_mxfp4_b3": Case(
            name="mixed_localcta_mxfp4_tail_mxfp4_b3",
            config="train_configs/nvpaper_nemotron_h_8b_mixed_localcta_mxfp4.toml",
            env=_mixed_nemotron_ssd_env(
                policy="tail_mxfp4",
                tail_layers=_mixed_tail_mxfp4_layers(),
            ),
            args=(
                "--training.local-batch-size",
                "3",
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
            ),
        ),
    }


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


def _build_train_command(
    case: Case,
    *,
    out_dir: Path,
    hf_assets_path: Path,
    train_data: Path,
    steps: int,
    global_batch_size: int,
    log_freq: int,
    nproc_per_node: int,
    extra_overrides: tuple[str, ...],
) -> list[str]:
    return [
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
        "--model.hf-assets-path",
        str(hf_assets_path),
        "--training.dataset",
        "packed-bin",
        "--training.dataset-path",
        str(train_data),
        "--training.load-dataset-kwargs",
        '{"num_workers":8,"prefetch_factor":4,"pin_memory":false,"repeat":false,"require_full_run":true}',
        "--training.steps",
        str(steps),
        "--training.global-batch-size",
        str(global_batch_size),
        "--metrics.log-freq",
        str(log_freq),
        "--metrics.disable-color-printing",
        *case.args,
        *extra_overrides,
    ]


def run_case(
    case: Case,
    *,
    out_base: Path,
    gpu: str,
    steps: int,
    log_freq: int,
    global_batch_size: int,
    nproc_per_node: int,
    steady_from: int,
    hf_assets_path: Path,
    train_data: Path,
    te213_stage: Path,
    te213_lib: Path,
    extra_overrides: tuple[str, ...],
    extra_env: dict[str, str],
) -> dict[str, object]:
    out_dir = out_base / case.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": _visible_devices_for_nproc(gpu, nproc_per_node),
            "WANDB_MODE": env.get("WANDB_MODE", "disabled"),
            "PYTHONUNBUFFERED": "1",
            "NVTE_FUSED_ATTN": "0",
        }
    )
    env.update(case.env)
    env.update(extra_env)
    if case.needs_te213:
        env["PYTHONPATH"] = f"{te213_stage}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["LD_LIBRARY_PATH"] = f"{te213_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    cmd = _build_train_command(
        case,
        out_dir=out_dir,
        hf_assets_path=hf_assets_path,
        train_data=train_data,
        steps=steps,
        global_batch_size=global_batch_size,
        log_freq=log_freq,
        nproc_per_node=nproc_per_node,
        extra_overrides=extra_overrides,
    )

    header = {"case": case.name, "cmd": cmd, "env": {**case.env, **extra_env}}
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

    rows = parse_step_metrics(log_path.read_text(errors="ignore"))
    summary = summarize_rows(rows, steady_from)
    numerics_ok = bool(rows) and all(
        math.isfinite(float(row["loss"])) and math.isfinite(float(row["grad_norm"]))
        for row in rows
    )
    summary.update(
        {
            "case": case.name,
            "returncode": rc,
            "completed": rc == 0 and bool(rows) and int(rows[-1]["step"]) >= steps and numerics_ok,
            "numerics_ok": numerics_ok,
            "wall_s": wall_s,
            "log": str(log_path),
        }
    )
    (out_dir / "steps.csv").write_text(_rows_csv(rows))
    return summary


def write_summary(out_base: Path, summaries: list[dict[str, object]], steady_from: int) -> None:
    (out_base / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Exact Nemotron-H 8B Local Matrix",
        "",
        f"Steady window starts at logged step >= {steady_from}.",
        "",
        "| case | done | numerics | last step | loss | peak MFU | steady MFU |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {case} | {done} | {numerics} | {step} | {loss} | {peak} | {steady} |".format(
                case=row["case"],
                done="yes" if row.get("completed") else "no",
                numerics="ok" if row.get("numerics_ok") else "bad",
                step=_fmt(row.get("last_step"), "{:.0f}"),
                loss=_fmt(row.get("last_loss"), "{:.4f}"),
                peak=_fmt(row.get("peak_mfu"), "{:.2f}"),
                steady=_fmt(row.get("steady_mfu"), "{:.2f}"),
            )
        )
    (out_base / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--log-freq", type=int, default=5)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=-1,
        help="Local single-GPU runs default to no gradient accumulation.",
    )
    parser.add_argument("--steady-from", type=int, default=10)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--allow-gpu2", action="store_true")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--hf-assets-path", type=Path, default=DEFAULT_HF_ASSETS)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--te213-stage", type=Path, default=DEFAULT_TE213_STAGE)
    parser.add_argument("--te213-lib", type=Path, default=DEFAULT_TE213_LIB)
    parser.add_argument(
        "--cuda-site-packages",
        type=Path,
        default=(
            Path(os.environ["LBT_NEMOTRON_H_CUDA_SITE_PACKAGES"])
            if os.environ.get("LBT_NEMOTRON_H_CUDA_SITE_PACKAGES")
            else None
        ),
        help="Site-packages containing selective_scan_cuda and causal_conv1d.",
    )
    parser.add_argument(
        "--fp4-root",
        type=Path,
        default=(
            Path(os.environ["FP4_MATMUL_ROOT"])
            if os.environ.get("FP4_MATMUL_ROOT")
            else None
        ),
        help="Pinned fp4_matmul checkout containing all native extensions.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["nvfp4_tk_v5_all_linear_ssd_b3"],
    )
    parser.add_argument("--config-override", action="append", default=[])
    parser.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out-base", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _check_gpu_selection(
        _visible_devices_for_nproc(args.gpu, args.nproc_per_node),
        allow_gpu2=args.allow_gpu2,
    )
    if not (args.hf_assets_path / "config.json").exists():
        raise SystemExit(f"Nemotron-H HF assets missing at {args.hf_assets_path}")
    if not args.train_data.exists():
        raise SystemExit(f"Training packed dataset not found: {args.train_data}")
    if args.cuda_site_packages is None or not args.cuda_site_packages.is_dir():
        raise SystemExit(
            "Nemotron native CUDA site-packages is missing; pass "
            "--cuda-site-packages or set LBT_NEMOTRON_H_CUDA_SITE_PACKAGES"
        )
    if not list(args.cuda_site_packages.glob("selective_scan_cuda*.so")):
        raise SystemExit(
            f"selective_scan_cuda extension missing from {args.cuda_site_packages}"
        )
    if args.fp4_root is None or not args.fp4_root.is_dir():
        raise SystemExit(
            "fp4_matmul checkout is missing; pass --fp4-root or set FP4_MATMUL_ROOT"
        )
    if not list(
        (args.fp4_root / "TK_quantisation" / "mamba_cuda").glob(
            "_nemotron_mamba_cuda*.so"
        )
    ):
        raise SystemExit(
            f"built Nemotron CUDA extension missing under {args.fp4_root}"
        )

    cases = build_cases()
    unknown = [case for case in args.cases if case not in cases]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    if any(cases[name].needs_te213 for name in args.cases):
        if not args.te213_stage.exists() or not args.te213_lib.exists():
            raise SystemExit(
                f"TE 2.13 stage/lib missing: {args.te213_stage} / {args.te213_lib}"
            )

    extra_env = {}
    for item in args.extra_env:
        if "=" not in item:
            raise SystemExit(f"--extra-env must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        extra_env[key] = value
    extra_env.setdefault(
        "LBT_NEMOTRON_H_CUDA_SITE_PACKAGES", str(args.cuda_site_packages)
    )
    extra_env.setdefault("FP4_MATMUL_ROOT", str(args.fp4_root))
    extra_env.setdefault("FP4_MXFP4_ROOT", str(args.fp4_root))
    extra_env.setdefault("FP4_MATMUL_GEMM_ROOT", str(args.fp4_root))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = args.out_base or Path(f"/tmp/lbt_nvpaper_nemotron_h_8b_{args.steps}_{stamp}")
    out_base.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.cases:
        summary = run_case(
            cases[name],
            out_base=out_base,
            gpu=str(args.gpu),
            steps=args.steps,
            log_freq=args.log_freq,
            global_batch_size=args.global_batch_size,
            nproc_per_node=args.nproc_per_node,
            steady_from=args.steady_from,
            hf_assets_path=args.hf_assets_path,
            train_data=args.train_data,
            te213_stage=args.te213_stage,
            te213_lib=args.te213_lib,
            extra_overrides=tuple(args.config_override),
            extra_env=extra_env,
        )
        summaries.append(summary)
        write_summary(out_base, summaries, args.steady_from)
        print(
            f"[{name}] rc={summary['returncode']} last_step={summary.get('last_step')} "
            f"loss={summary.get('last_loss')} peak_mfu={summary.get('peak_mfu')} "
            f"steady_mfu={summary.get('steady_mfu')} log={summary['log']}",
            flush=True,
        )
        if summary["returncode"] != 0:
            return int(summary["returncode"])
    print(f"\nsummary: {out_base / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
