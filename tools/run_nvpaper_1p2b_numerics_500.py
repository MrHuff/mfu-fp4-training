#!/usr/bin/env python3
"""Run and summarize 1.2B paper-model BF16/TE/TK/localCTA numerics.

The default cases are intentionally explicit:
  - bf16_ref: BF16 model with the BF16 CCE loss path
  - te_ref: unedited native TE FP4 internal linears plus NVFP4 CCE final loss
  - tk_v5: regular TK/v5 path with deterministic no-RHT/no-SR quantization
  - localcta_v4: localCTA v4 path with the same deterministic quantization

Materializing full BF16/TE logits for batch 8, sequence 8192, vocab 131072 does
not fit in 184GiB, so the reference routes use internal CCE loss.

The fp4_matmul checkout currently does not ship NVFP4 CCE v5 entrypoints, so
"tk_v5" means the v5 regular-TK quantizer/GEMM route with the shared v4 CCE
final layer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as stats
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FP4_ROOT = Path("/opt/mfu/EXTERNAL_PATH")
DEFAULT_TE213_STAGE = Path("/tmp/te213_stage")
DEFAULT_TE213_LIB = DEFAULT_TE213_STAGE / "nvidia" / "cu13" / "lib"
DEFAULT_BLACKLISTED_GPUS = {"2"}
DEFAULT_LOCALCTA_PAIR_ROUTE = (
    REPO_ROOT
    / "analysis_outputs"
    / "gb200_localcta_rht_step36000_w32_s2000_p10000_r14_20260828"
    / "localcta_route.env"
)

ANSI_RE = re.compile(r"\x1b\[[0-9;:]*m")
STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[-+0-9.eEnNaAiIfF]+)\s+"
    r"grad_norm:\s*(?P<grad_norm>[-+0-9.eEnNaAiIfF]+)\s+"
    r"memory:\s*(?P<memory_gib>[-+0-9.eE]+)GiB"
    r".*?tps:\s*(?P<tps>[0-9,]+)\s+"
    r"tflops:\s*(?P<tflops>[0-9.,]+)\s+"
    r"mfu:\s*(?P<mfu>[-+0-9.eE]+)%"
)
VALIDATION_RE = re.compile(
    r"validate step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[-+0-9.eE]+)\s+"
    r"memory:\s*(?P<memory_gib>[-+0-9.eE]+)GiB"
    r".*?tps:\s*(?P<tps>[0-9,]+)"
)


@dataclass(frozen=True)
class Case:
    name: str
    config: str
    env: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()
    needs_te213: bool = False


def _load_literal_export_env(path: Path) -> dict[str, str]:
    """Load the capsule's literal ``export KEY='VALUE'`` route without sourcing it."""

    result: dict[str, str] = {}
    pattern = re.compile(r"^export ([A-Za-z_][A-Za-z0-9_]*)='([^']*)'$")
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError(f"{path}:{line_number}: unsupported route line: {line!r}")
        result[match.group(1)] = match.group(2)
    return result


def _fp4_det_env() -> dict[str, str]:
    return {
        "NVTE_NVFP4_DISABLE_RHT": "1",
        "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
        "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
        "NVTE_NVFP4_ENCODE_CENTRIC": "0",
        "NVFP4_USE_RHT": "0",
        "NVFP4_RHT_ACTIVATION": "0",
        "NVFP4_RHT_GRAD": "0",
        "NVFP4_RHT_WEIGHT": "0",
        "NVFP4_RHT_AXES": "col",
        "NVFP4_RHT_RANDOM_SIGNS": "0",
        "NVFP4_USE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SR_ACTIVATION": "0",
        "NVFP4_SR_GRAD": "0",
        "NVFP4_SR_WEIGHT": "0",
        "NVFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SCALE_SR_ACTIVATION": "0",
        "NVFP4_SCALE_SR_GRAD": "0",
        "NVFP4_SCALE_SR_WEIGHT": "0",
    }


def _fp4_te_style_numeric_env() -> dict[str, str]:
    """Low-bits NVFP4 knobs for the TE-style training recipe.

    TE 2.13's native NVFP4 recipe applies RHT to fwd inputs and bwd grads,
    stochastic rounding to bwd grads, encode-centric quantization, and 2D
    quantization to weights. The TK bridge cannot exactly express TE's
    weight-2D path today, so the model rows use the stable low-bits col-RHT +
    grad data-SR recipe. Regular TK v5 overrides the axis to row-RHT so it can
    keep its fused RMSNorm producer.
    """

    return {
        "NVTE_NVFP4_ENCODE_CENTRIC": "1",
        "NVFP4_USE_RHT": "1",
        "NVFP4_RHT_ACTIVATION": "1",
        "NVFP4_RHT_GRAD": "1",
        "NVFP4_RHT_WEIGHT": "0",
        "NVFP4_RHT_AXES": "col",
        "NVFP4_RHT_RANDOM_SIGNS": "0",
        "NVFP4_USE_STOCHASTIC_ROUNDING": "1",
        "NVFP4_SR_ACTIVATION": "0",
        "NVFP4_SR_GRAD": "1",
        "NVFP4_SR_WEIGHT": "0",
        "NVFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "NVFP4_SCALE_SR_ACTIVATION": "0",
        "NVFP4_SCALE_SR_GRAD": "0",
        "NVFP4_SCALE_SR_WEIGHT": "0",
    }


def _tk_sqrelu_env() -> dict[str, str]:
    return {
        "USE_FP4_SQRELU_FFN_TK": "1",
        "USE_TK_SQRELU_FFN_OVERLAP_W1_WGRAD_RMS": "1",
        "USE_TK_SQRELU_FFN_OVERLAP_W2_WGRAD_DERIV": "1",
        "USE_TK_SQRELU_FFN_CACHED_RMS_BWD": "1",
    }


def _localcta_v4_recovered_env() -> dict[str, str]:
    return {
        "USE_TK_LOCALCTA_V4_FAST_PREPARED_PRODUCER": "1",
        "USE_TK_LOCALCTA_V4_ROW_PREPARED_COL_OUTER": "1",
        "USE_TK_LOCALCTA_V4_RAW_OUTER_TMA_GRAD": "0",
        "USE_TK_LOCALCTA_V4_FUSED_SQRELU_QUANT": "1",
        "USE_TK_LOCALCTA_V4_FUSED_SQRELU_DERIV_QUANT": "0",
        "USE_TK_LOCALCTA_V4_FAST_FORWARD_GEMM": "1",
        "USE_TK_LOCALCTA_V4_FAST_SINGLE_DGRAD": "1",
        "USE_TK_LOCALCTA_V4_FAST_SINGLE_WGRAD": "1",
        "USE_TK_LOCALCTA_V4_FAST_QKV_ONEPASS_DGRAD": "1",
        "USE_TK_LOCALCTA_V4_FULLCOL_QKV_DGRAD": "1",
        "USE_TK_LOCALCTA_V4_FAST_WO_DGRAD": "1",
        "USE_TK_LOCALCTA_V4_FAST_WO_WGRAD": "1",
        "USE_TK_SQRELU_FFN_OVERLAP_W1_WGRAD_RMS": "1",
        "USE_TK_SQRELU_FFN_OVERLAP_W2_WGRAD_DERIV": "1",
        "USE_TK_SQRELU_FFN_CACHED_RMS_BWD": "1",
        "USE_TK_LOCALCTA_V4_FAST_DATA_SR": "1",
        "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_FROM_RAW": "1",
        "USE_TK_LOCALCTA_V4_COL_RHT_AMAX_RAW_MULTIPLIER": "2.0",
    }


def _disable_tk_env() -> dict[str, str]:
    return {
        "USE_TK_GEMM": "0",
        "USE_TK_QUANT": "0",
        "USE_TK_LOCALCTA": "0",
        "USE_TK_LOCALCTA_FUSED": "0",
        "USE_MXFP4_TK_BACKEND": "0",
        "USE_MXFP4_TK_FUSED": "0",
    }


def _nvpaper_1p2b_final_cce_env() -> dict[str, str]:
    """Stable NVFP4 CCE final-layer defaults for the 1.2B paper shape.

    Isolated benchmark shape: M=65536, K=2048, V=131072. On H100 GPU1 this
    moved nv-v4-enc from ~41.9 ms to ~36.1 ms while preserving dH/dW cosine.
    """

    return {
        "FP4_CCE_ASSUME_NONEMPTY_LABELS": "1",
        "FP4_CCE_NVFP4_EXACT_NORM_QUANT": "0",
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


def _mxfp4_highwater_env() -> dict[str, str]:
    """Stable MXFP4 1.2B high-water defaults from run_mxfp4_highwater_repro.sh."""

    return {
        "FP4_MXFP4_ROOT": os.environ.get("FP4_MXFP4_ROOT", str(DEFAULT_FP4_ROOT)),
        "FP4_MATMUL_GEMM_ROOT": os.environ.get(
            "FP4_MATMUL_GEMM_ROOT",
            "/opt/mfu/EXTERNAL_PATH",
        ),
        "FP4_CCE_ASSUME_NONEMPTY_LABELS": "0",
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
        "MXFP4_USE_RHT": "0",
        "MXFP4_RHT_ACTIVATION": "0",
        "MXFP4_RHT_GRAD": "0",
        "MXFP4_RHT_WEIGHT": "0",
        "MXFP4_RHT_RANDOM_SIGN_MASK": "0",
        "MXFP4_USE_STOCHASTIC_ROUNDING": "0",
        "MXFP4_SR_ACTIVATION": "0",
        "MXFP4_SR_GRAD": "0",
        "MXFP4_SR_WEIGHT": "0",
        "MXFP4_USE_SCALE_STOCHASTIC_ROUNDING": "0",
        "MXFP4_SCALE_SR_ACTIVATION": "0",
        "MXFP4_SCALE_SR_GRAD": "0",
        "MXFP4_SCALE_SR_WEIGHT": "0",
        "MXFP4_BACKEND_VERSION": "v4",
        "MXFP4_USE_QKV_ROPE_EPILOGUE": "1",
        "MXFP4_USE_QKV_DIRECT_OUTPUTS": "1",
        "MXFP4_USE_QKV_RMSNORM_QUANT_FUSION": "1",
        "MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD": "0",
        "MXFP4_QKV_BWD_STATE_SLOTS": "4",
        "MXFP4_USE_QKV_BF16_WGRAD": "0",
        "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM": "1",
        "MXFP4_QKV_WGRAD_WAIT_BEFORE_RMSNORM_DGAMMA": "1",
        "MXFP4_USE_QKV_FWD_WEIGHT_QUANT_OVERLAP": "0",
        "MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD": "1",
        "MXFP4_USE_SPLIT2_FFN_INPLACE_QUANT": "1",
        "MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP": "0",
        "MXFP4_USE_SPLIT2_FFN_ROW_OVERLAP_RHT": "1",
        "MXFP4_USE_SPLIT2_FFN_PRODUCER_SPLIT": "0",
        "MXFP4_USE_BWD_WGRAD_OVERLAP": "0",
        "MXFP4_USE_BWD_STATE_CACHE": "0",
        "MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP": "-1",
        "MXFP4_EARLY_WGRAD_CONFIG_MAX_STEP_M4096_N2048": "199",
        "MXFP4_USE_RESIDUAL_FUSION": "1",
        "MXFP4_USE_RESIDUAL_FUSION_FFN": "1",
        "MXFP4_USE_RESIDUAL_FUSION_ATTN": "0",
        "MXFP4_ALLOW_UNSAFE_ATTN_FFN_RESIDUAL_OVERLAP": "0",
        "MXFP4_UNSAFE_RESIDUAL_FALLBACK": "prefer_ffn",
        "MXFP4_USE_GEMM_RESIDUAL_KERNEL": "1",
        "USE_TK_LOCALCTA_V4_FINAL_SG_PRODUCER": "1",
        "MXFP4_RHT_AXES": "row",
        "MXFP4_USE_FUSED_RMSNORM_QUANT_RHT": "1",
        "MXFP4_USE_FUSED_SILU_FFN_QUANT": "0",
        "MXFP4_USE_FUSED_SILU_DERIV_SPLIT2_RHT": "0",
        "MXFP4_USE_FUSED_SQRELU_QUANT": "1",
        "MXFP4_USE_TMA_SQRELU_QUANT": "1",
        "MXFP4_USE_FUSED_SQRELU_DERIV_QUANT": "0",
        "MXFP4_USE_SQRELU_FUSED_RMS_W1": "0",
        "MXFP4_USE_SIMPLE_SQRELU_FUSED_W2": "1",
        "MXFP4_USE_SQRELU_SPLIT_COL_OVERLAP": "0",
        "MXFP4_USE_SQRELU_SPLIT_COL_QUANT": "0",
        "MXFP4_USE_SQRELU_SPLIT_COL_WAIT_FORWARD": "1",
        "MXFP4_USE_SQRELU_W2_WGRAD_OVERLAP": "0",
        "MXFP4_USE_SQRELU_W2_WGRAD_AFTER_DGRAD_OVERLAP": "0",
        "MXFP4_USE_SQRELU_DERIV_GEMM_EPILOGUE": "0",
        "MXFP4_USE_WO_ATTN_LAYOUT": "0",
    }


def build_cases() -> dict[str, Case]:
    fp4_det = _fp4_det_env()
    fp4_te_style = _fp4_te_style_numeric_env()
    tk_sqrelu = _tk_sqrelu_env()
    localcta_v4_recovered = _localcta_v4_recovered_env()
    cce_final = _nvpaper_1p2b_final_cce_env()
    localcta_pair_route = _load_literal_export_env(DEFAULT_LOCALCTA_PAIR_ROUTE)
    localcta_sr_pair_control = {
        **localcta_pair_route,
        "NVFP4_RHT_AXES": "col",
        # The common carrier requires the sealed sign geometry in both arms.
        # The control does not apply it because all three RHT enable bits are off.
        "NVFP4_RHT_RANDOM_SIGNS": "1",
        "NVFP4_USE_RHT": "0",
        "NVFP4_RHT_ACTIVATION": "0",
        "NVFP4_RHT_GRAD": "0",
        "NVFP4_RHT_WEIGHT": "0",
    }
    localcta_sr_pair_rht = {
        **localcta_sr_pair_control,
        "NVFP4_USE_RHT": "1",
        "NVFP4_RHT_ACTIVATION": "1",
        "NVFP4_RHT_GRAD": "1",
    }
    compiled_bf16_head_args = (
        "--fp4-cce.enabled",
        "--fp4-cce.backend",
        "torch_compile_bf16",
        "--fp4-cce.implementation",
        "v2",
        "--fp4-cce.quant-mode",
        "enc",
    )
    localcta_v4_env = {
        **fp4_det,
        **tk_sqrelu,
        **localcta_v4_recovered,
        "USE_TK_GEMM": "1",
        "USE_TK_QUANT": "1",
        "USE_TK_LOCALCTA": "1",
        "USE_TK_LOCALCTA_VARIANT": "v4",
        "USE_TK_LOCALCTA_FUSED": "0",
        "NVTE_CUSTOM_QUANT": "1",
        "USE_TK_LOCALCTA_V4_FFN_RESIDUAL_EPILOGUE": "1",
        "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
        "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
        **cce_final,
    }
    mixed_env = {
        **localcta_v4_env,
        **_mxfp4_highwater_env(),
        "LBT_FP4_MIXED_POLICY": os.environ.get("LBT_FP4_MIXED_POLICY", "front_localcta"),
    }
    if os.environ.get("LBT_FP4_MIXED_LAYERS"):
        mixed_env["LBT_FP4_MIXED_LAYERS"] = os.environ["LBT_FP4_MIXED_LAYERS"]
    if os.environ.get("LBT_FP4_MIXED_SPLIT_LAYER"):
        mixed_env["LBT_FP4_MIXED_SPLIT_LAYER"] = os.environ["LBT_FP4_MIXED_SPLIT_LAYER"]
    return {
        "bf16_ref": Case(
            name="bf16_ref",
            config="train_configs/nvpaper_transformer_1p2b_bf16_matrix.toml",
            env={**_disable_tk_env()},
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "triton_bf16",
            ),
        ),
        "te_ref": Case(
            name="te_ref",
            config="train_configs/nvpaper_transformer_1p2b_pure_te_fp4_matrix.toml",
            env={
                **_disable_tk_env(),
                "FP4_KEEP_LAST_N_LAYERS_BF16": "4",
                **cce_final,
            },
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
            needs_te213=True,
        ),
        "te_det_ref": Case(
            name="te_det_ref",
            config="train_configs/nvpaper_transformer_1p2b_pure_te_fp4_matrix.toml",
            env={**fp4_det, **_disable_tk_env(), **cce_final},
            args=(
                "--fp4-cce.enabled",
                "--fp4-cce.backend",
                "nvfp4",
                "--fp4-cce.implementation",
                "v4",
                "--fp4-cce.quant-mode",
                "enc",
            ),
            needs_te213=True,
        ),
        "te_full_logits": Case(
            name="te_full_logits",
            config="train_configs/nvpaper_transformer_1p2b_pure_te_fp4_matrix.toml",
            env={**fp4_det, **_disable_tk_env()},
            args=("--fp4-cce.no-enabled",),
            needs_te213=True,
        ),
        "tk_v5": Case(
            name="tk_v5",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_tk_v5_matrix.toml",
            env={
                **fp4_det,
                **tk_sqrelu,
                "USE_TK_GEMM": "1",
                "USE_TK_QUANT": "1",
                "USE_TK_LOCALCTA": "0",
                "USE_TK_LOCALCTA_FUSED": "0",
                "FP4_ATTN_BACKEND": "tk",
                "FP4_FFN_BACKEND": "tk",
                "NVTE_CUSTOM_QUANT": "1",
                "USE_TK_QKV_ROPE_EPILOGUE": "1",
                "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
                **cce_final,
            },
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
        "tk_v5_te_recipe": Case(
            name="tk_v5_te_recipe",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_tk_v5_matrix.toml",
            env={
                **fp4_te_style,
                **tk_sqrelu,
                "NVFP4_RHT_AXES": "row",
                "USE_TK_GEMM": "1",
                "USE_TK_QUANT": "1",
                "USE_TK_LOCALCTA": "0",
                "USE_TK_LOCALCTA_FUSED": "0",
                "FP4_ATTN_BACKEND": "tk",
                "FP4_FFN_BACKEND": "tk",
                "NVTE_CUSTOM_QUANT": "1",
                "USE_TK_QKV_ROPE_EPILOGUE": "1",
                "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
                **cce_final,
            },
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
        "tk_v5_noextras": Case(
            name="tk_v5_noextras",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_tk_v5_matrix.toml",
            env={
                **fp4_det,
                **tk_sqrelu,
                "USE_TK_GEMM": "1",
                "USE_TK_QUANT": "1",
                "USE_TK_LOCALCTA": "0",
                "USE_TK_LOCALCTA_FUSED": "0",
                "FP4_ATTN_BACKEND": "tk",
                "FP4_FFN_BACKEND": "tk",
                "NVTE_CUSTOM_QUANT": "1",
                "USE_TK_QKV_ROPE_EPILOGUE": "1",
                "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
                **cce_final,
            },
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
        "localcta_v4": Case(
            name="localcta_v4",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_localcta_v4_matrix.toml",
            env=localcta_v4_env,
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
        "localcta_v4_sr_s0_compiled_ce": Case(
            name="localcta_v4_sr_s0_compiled_ce",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_localcta_v4_matrix.toml",
            env=localcta_sr_pair_control,
            args=compiled_bf16_head_args,
        ),
        "localcta_v4_sr_rht_s0_compiled_ce": Case(
            name="localcta_v4_sr_rht_s0_compiled_ce",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_localcta_v4_matrix.toml",
            env=localcta_sr_pair_rht,
            args=compiled_bf16_head_args,
        ),
        "localcta_v4_fused_split": Case(
            name="localcta_v4_fused_split",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_localcta_v4_matrix.toml",
            env={**localcta_v4_env, "LBT_LOCALCTA_V4_PROFILE": "fused_split"},
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
        "localcta_v4_te_recipe": Case(
            name="localcta_v4_te_recipe",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_localcta_v4_matrix.toml",
            env={
                **fp4_te_style,
                **tk_sqrelu,
                **localcta_v4_recovered,
                "USE_TK_GEMM": "1",
                "USE_TK_QUANT": "1",
                "USE_TK_LOCALCTA": "1",
                "USE_TK_LOCALCTA_VARIANT": "v4",
                "USE_TK_LOCALCTA_FUSED": "0",
                "NVTE_CUSTOM_QUANT": "1",
                "USE_TK_LOCALCTA_V4_FFN_RESIDUAL_EPILOGUE": "1",
                "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
                **cce_final,
            },
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
        "localcta_v4_noextras": Case(
            name="localcta_v4_noextras",
            config="train_configs/nvpaper_transformer_1p2b_nvfp4_localcta_v4_matrix.toml",
            env={
                **fp4_det,
                **tk_sqrelu,
                **localcta_v4_recovered,
                "USE_TK_GEMM": "1",
                "USE_TK_QUANT": "1",
                "USE_TK_LOCALCTA": "1",
                "USE_TK_LOCALCTA_VARIANT": "v4",
                "USE_TK_LOCALCTA_FUSED": "0",
                "NVTE_CUSTOM_QUANT": "1",
                "USE_TK_LOCALCTA_V4_FFN_RESIDUAL_EPILOGUE": "1",
                "FP4_CCE_V4_FUSED_X_PRODUCER": "1",
                "FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER": "1",
                **cce_final,
            },
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
        "mixed_localcta_mxfp4": Case(
            name="mixed_localcta_mxfp4",
            config="train_configs/nvpaper_transformer_1p2b_mixed_localcta_mxfp4_matrix.toml",
            env=mixed_env,
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
            config="train_configs/nvpaper_transformer_1p2b_mixed_localcta_mxfp4_matrix.toml",
            env={**mixed_env, "LBT_LOCALCTA_V4_PROFILE": "fused_split"},
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
        "mxfp4_highwater": Case(
            name="mxfp4_highwater",
            config="train_configs/nvpaper_transformer_1p2b_mxfp4_tk_matrix.toml",
            env={**_mxfp4_highwater_env()},
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
    }


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_step_metrics(log_text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in strip_ansi(log_text).splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad_norm")),
                "memory_gib": float(match.group("memory_gib")),
                "tps": float(match.group("tps").replace(",", "")),
                "tflops": float(match.group("tflops").replace(",", "")),
                "mfu": float(match.group("mfu")),
            }
        )
    return rows


def parse_validation_metrics(log_text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for line in strip_ansi(log_text).splitlines():
        match = VALIDATION_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "memory_gib": float(match.group("memory_gib")),
                "tps": float(match.group("tps").replace(",", "")),
            }
        )
    return rows


def summarize_rows(rows: list[dict[str, float | int]], steady_from: int) -> dict[str, object]:
    if not rows:
        return {
            "completed": False,
            "logged_steps": 0,
            "last_step": None,
            "last_loss": None,
            "last_grad_norm": None,
            "peak_tps": None,
            "peak_mfu": None,
            "steady_tps": None,
            "steady_mfu": None,
        }
    steady = [row for row in rows if int(row["step"]) >= steady_from] or rows
    last = rows[-1]
    return {
        "completed": False,
        "logged_steps": len(rows),
        "first_step": rows[0]["step"],
        "last_step": last["step"],
        "last_loss": last["loss"],
        "last_grad_norm": last["grad_norm"],
        "last_tps": last["tps"],
        "last_mfu": last["mfu"],
        "peak_tps": max(float(row["tps"]) for row in rows),
        "peak_mfu": max(float(row["mfu"]) for row in rows),
        "steady_tps": stats.mean(float(row["tps"]) for row in steady),
        "steady_mfu": stats.mean(float(row["mfu"]) for row in steady),
    }


def summarize_validation(rows: list[dict[str, float | int]]) -> dict[str, object]:
    if not rows:
        return {
            "validation_points": 0,
            "last_validation_step": None,
            "last_validation_loss": None,
        }
    last = rows[-1]
    return {
        "validation_points": len(rows),
        "last_validation_step": last["step"],
        "last_validation_loss": last["loss"],
    }


def run_case(
    case: Case,
    *,
    out_base: Path,
    gpu: str,
    steps: int,
    log_freq: int,
    nproc_per_node: int,
    nnodes: int,
    node_rank: int,
    master_addr: str,
    master_port: str,
    rdzv_id: str,
    seed: Optional[int],
    fp4_root: Path,
    te213_stage: Path,
    te213_lib: Path,
    steady_from: int,
    config_overrides: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
) -> dict[str, object]:
    out_dir = out_base / case.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "WANDB_MODE": env.get("WANDB_MODE", "disabled"),
            "PYTHONUNBUFFERED": "1",
            "FP4_MATMUL_ROOT": str(fp4_root),
            "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
            "TORCH_CUDNN_SDPA_ENABLED": "1",
            "NVTE_FUSED_ATTN": "0",
        }
    )
    env.update(case.env)
    if extra_env:
        env.update(extra_env)
    if case.needs_te213:
        env["PYTHONPATH"] = f"{te213_stage}:{env.get('PYTHONPATH', '')}".rstrip(":")
        env["LD_LIBRARY_PATH"] = f"{te213_lib}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    torchrun_args = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(nproc_per_node),
    ]
    if nnodes > 1:
        torchrun_args.extend(
            [
                "--nnodes",
                str(nnodes),
                "--node_rank",
                str(node_rank),
                "--rdzv_id",
                rdzv_id,
                "--rdzv_backend",
                "c10d",
                "--rdzv_endpoint",
                f"{master_addr}:{master_port}",
                "--rdzv-conf",
                "timeout=3600",
            ]
        )
    else:
        torchrun_args.append("--standalone")

    cmd = [
        *torchrun_args,
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
        *config_overrides,
        *case.args,
    ]
    if seed is not None:
        cmd.extend(["--debug.seed", str(seed)])

    header = {
        "case": case.name,
        "cmd": cmd,
        "cwd": str(REPO_ROOT),
        "distributed": {
            "nproc_per_node": nproc_per_node,
            "nnodes": nnodes,
            "node_rank": node_rank,
            "master_addr": master_addr,
            "master_port": master_port,
            "rdzv_id": rdzv_id,
        },
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
    summary.update(summarize_validation(validation_rows))
    summary.update(
        {
            "case": case.name,
            "returncode": rc,
            "completed": rc == 0 and bool(rows) and int(rows[-1]["step"]) >= steps,
            "wall_s": wall_s,
            "log": str(log_path),
        }
    )
    (out_dir / "steps.csv").write_text(_rows_csv(rows), encoding="utf-8")
    (out_dir / "validation.csv").write_text(
        _validation_rows_csv(validation_rows), encoding="utf-8"
    )
    return summary


def _logged_env_key(key: str) -> bool:
    prefixes = (
        "CUDA_VISIBLE_DEVICES",
        "FP4",
        "LOW_BITS",
        "NVFP4",
        "NVTE",
        "PURE_TE",
        "TORCH_CUDNN_SDPA_ENABLED",
        "USE_",
        "WANDB_MODE",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "MXFP4",
        "LBT_",
    )
    return any(key.startswith(prefix) for prefix in prefixes)


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


def write_summary(out_base: Path, summaries: list[dict[str, object]], steady_from: int) -> None:
    summary_json = out_base / "summary.json"
    summary_json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    fields = [
        "case",
        "completed",
        "returncode",
        "last_step",
        "last_loss",
        "last_grad_norm",
        "peak_tps",
        "peak_mfu",
        "steady_tps",
        "steady_mfu",
        "validation_points",
        "last_validation_step",
        "last_validation_loss",
        "wall_s",
        "log",
    ]
    with (out_base / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})

    by_name = {str(row["case"]): row for row in summaries}
    lines = [
        f"# 1.2B Numerics Summary",
        "",
        f"Steady window starts at logged step >= {steady_from}.",
        "",
        "| case | done | last step | loss | grad norm | peak tps | peak MFU | steady tps | steady MFU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {case} | {done} | {step} | {loss} | {grad} | {peak_tps} | {peak_mfu} | {steady_tps} | {steady_mfu} |".format(
                case=row["case"],
                done="yes" if row.get("completed") else "no",
                step=_fmt(row.get("last_step"), "{:.0f}"),
                loss=_fmt(row.get("last_loss"), "{:.4f}"),
                grad=_fmt(row.get("last_grad_norm"), "{:.4f}"),
                peak_tps=_fmt(row.get("peak_tps"), "{:.0f}"),
                peak_mfu=_fmt(row.get("peak_mfu"), "{:.2f}"),
                steady_tps=_fmt(row.get("steady_tps"), "{:.0f}"),
                steady_mfu=_fmt(row.get("steady_mfu"), "{:.2f}"),
            )
        )

    lines.append("")
    lines.extend(_comparison_lines(by_name))
    lines.append("")
    lines.append("Logs:")
    for row in summaries:
        lines.append(f"- {row['case']}: `{row['log']}`")
    (out_base / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_curve_manifest(out_base, summaries)


def _write_curve_manifest(out_base: Path, summaries: list[dict[str, object]]) -> None:
    display_rows = {
        "bf16_ref": (
            "baseline",
            "BF16 + BF16 CCE",
            "All BF16 with internal BF16 CCE",
        ),
        "te_ref": (
            "baseline",
            "TE 2.13 NVFP4 original-style",
            "Native TE defaults; output head BF16; last 4 transformer blocks BF16 if FP4_KEEP_LAST_N_LAYERS_BF16=4 is set",
        ),
        "tk_v5_noextras": (
            "fast",
            "NVFP4 TK v5 no extras",
            "Regular TK v5 square-ReLU route",
        ),
        "tk_v5": (
            "fast",
            "NVFP4 TK v5 no extras",
            "Regular TK v5 square-ReLU route",
        ),
        "localcta_v4_noextras": (
            "fast",
            "NVFP4 localCTA v4 no extras",
            "Recovered localCTA v4 square-ReLU route",
        ),
        "localcta_v4": (
            "fast",
            "NVFP4 localCTA v4 no extras",
            "Recovered localCTA v4 square-ReLU route",
        ),
        "mxfp4_highwater": (
            "fast",
            "MXFP4 high-water",
            "Stable high-water MXFP4 route; no RHT/SR extras",
        ),
    }
    fields = [
        "group",
        "run",
        "recipe",
        "compute_log",
        "loss_refresh_log",
        "validation_log",
    ]
    with (out_base / "curves_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            case = str(summary["case"])
            if case not in display_rows:
                continue
            group, run, recipe = display_rows[case]
            log = str(summary["log"])
            writer.writerow(
                {
                    "group": group,
                    "run": run,
                    "recipe": recipe,
                    "compute_log": log,
                    "loss_refresh_log": log,
                    "validation_log": log,
                }
            )


def _comparison_lines(by_name: dict[str, dict[str, object]]) -> list[str]:
    lines = ["Comparisons:"]
    for lhs, rhs, label in [
        ("tk_v5", "te_ref", "tk_v5 vs te_ref"),
        ("localcta_v4", "tk_v5", "localCTA v4 vs TK v5"),
    ]:
        if lhs not in by_name or rhs not in by_name:
            continue
        a = by_name[lhs].get("last_loss")
        b = by_name[rhs].get("last_loss")
        if a is None or b is None:
            continue
        delta = float(a) - float(b)
        rel = delta / max(abs(float(b)), 1e-12)
        lines.append(f"- {label} final-loss delta: {delta:+.6f} ({rel:+.4%})")

    if "localcta_v4" in by_name and "tk_v5" in by_name:
        local = by_name["localcta_v4"]
        tk = by_name["tk_v5"]
        if local.get("steady_mfu") is not None and tk.get("steady_mfu") is not None:
            lines.append(
                "- localCTA v4 steady speedup vs TK v5: "
                f"{float(local['steady_mfu']) - float(tk['steady_mfu']):+.2f} MFU, "
                f"{float(local['steady_tps']) / max(float(tk['steady_tps']), 1e-12):.3f}x TPS"
            )
    return lines


def _fmt(value: object, fmt: str) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return fmt.format(float(value))


def _same_resolved_path(lhs: str | Path | None, rhs: str | Path | None) -> bool:
    if lhs is None or rhs is None:
        return False
    lhs_path = Path(lhs).expanduser()
    rhs_path = Path(rhs).expanduser()
    try:
        return lhs_path.resolve() == rhs_path.resolve()
    except OSError:
        return lhs_path.absolute() == rhs_path.absolute()


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no-debug-seed",
        action="store_true",
        help="Do not pass --debug.seed; use the training stack default RNG/data order.",
    )
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--allow-gpu2",
        action="store_true",
        help="Permit GPU2 despite the local benchmark blacklist.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=int(os.environ.get("NPROC_PER_NODE", os.environ.get("GPUS_PER_NODE", "1"))),
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=int(os.environ.get("NNODES", os.environ.get("PET_NNODES", "1"))),
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=int(os.environ.get("NODE_RANK", os.environ.get("PET_NODE_RANK", "0"))),
    )
    parser.add_argument(
        "--master-addr",
        default=os.environ.get("MASTER_ADDR", os.environ.get("PET_MASTER_ADDR", "127.0.0.1")),
    )
    parser.add_argument(
        "--master-port",
        default=os.environ.get("MASTER_PORT", os.environ.get("PET_MASTER_PORT", "29500")),
    )
    parser.add_argument("--rdzv-id", default=os.environ.get("RDZV_ID", "nvpaper-1p2b"))
    parser.add_argument(
        "--no-auto-global-batch",
        action="store_true",
        help=(
            "Do not override training.global_batch_size to -1 for distributed "
            "runs. The stock 1.2B configs set global_batch_size=8, which only "
            "works for one rank with local_batch_size=8."
        ),
    )
    parser.add_argument("--fp4-root", type=Path, default=DEFAULT_FP4_ROOT)
    parser.add_argument("--te213-stage", type=Path, default=DEFAULT_TE213_STAGE)
    parser.add_argument("--te213-lib", type=Path, default=DEFAULT_TE213_LIB)
    parser.add_argument("--out-base", type=Path, default=None)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["bf16_ref", "te_ref", "tk_v5", "localcta_v4"],
    )
    parser.add_argument("--steady-from", type=int, default=50)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--load-dataset-kwargs", default=None)
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment override applied after the selected case defaults. May be repeated.",
    )
    parser.add_argument("--validation-enable", action="store_true")
    parser.add_argument("--validation-dataset", default=None)
    parser.add_argument("--validation-dataset-path", default=None)
    parser.add_argument("--validation-local-batch-size", type=int, default=None)
    parser.add_argument("--validation-seq-len", type=int, default=None)
    parser.add_argument("--validation-freq", type=int, default=None)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument(
        "--validation-load-dataset-kwargs",
        default=None,
        help=(
            "JSON dataloader kwargs for packed-bin validation. Stored in "
            "LBT_VALIDATION_LOAD_DATASET_KWARGS because Torchtitan has no "
            "validation.load_dataset_kwargs config field."
        ),
    )
    args = parser.parse_args(argv)
    _check_gpu_selection(args.gpu, allow_gpu2=args.allow_gpu2)

    cases = build_cases()
    unknown = [name for name in args.cases if name not in cases]
    if unknown:
        raise SystemExit(f"Unknown case(s): {', '.join(unknown)}")
    if any(cases[name].needs_te213 for name in args.cases):
        if not args.te213_stage.exists() or not args.te213_lib.exists():
            raise SystemExit(
                f"TE 2.13 stage/lib missing: {args.te213_stage} / {args.te213_lib}"
            )
    if args.validation_enable and _same_resolved_path(
        args.dataset_path,
        args.validation_dataset_path,
    ):
        raise SystemExit(
            "Validation dataset must be held out: --validation-dataset-path "
            "matches --dataset-path."
        )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_base = args.out_base or Path(f"/tmp/lbt_nvpaper_1p2b_numerics_{args.steps}_{stamp}")
    out_base.mkdir(parents=True, exist_ok=True)

    seed = None if args.no_debug_seed else args.seed
    config_overrides = []
    if args.dataset is not None:
        config_overrides.extend(["--training.dataset", args.dataset])
    if args.dataset_path is not None:
        config_overrides.extend(["--training.dataset-path", args.dataset_path])
    if args.load_dataset_kwargs is not None:
        config_overrides.extend(
            ["--training.load-dataset-kwargs", args.load_dataset_kwargs]
        )
    if (
        args.nnodes * args.nproc_per_node > 1
        and not args.no_auto_global_batch
    ):
        config_overrides.extend(["--training.global-batch-size", "-1"])
    if args.validation_enable:
        config_overrides.append("--validation.enable")
    if args.validation_dataset is not None:
        config_overrides.extend(["--validation.dataset", args.validation_dataset])
    if args.validation_dataset_path is not None:
        config_overrides.extend(
            ["--validation.dataset-path", args.validation_dataset_path]
        )
    if args.validation_local_batch_size is not None:
        config_overrides.extend(
            ["--validation.local-batch-size", str(args.validation_local_batch_size)]
        )
    if args.validation_seq_len is not None:
        config_overrides.extend(["--validation.seq-len", str(args.validation_seq_len)])
    if args.validation_freq is not None:
        config_overrides.extend(["--validation.freq", str(args.validation_freq)])
    if args.validation_steps is not None:
        config_overrides.extend(["--validation.steps", str(args.validation_steps)])
    extra_env = {}
    if args.validation_load_dataset_kwargs is not None:
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
    for name in args.cases:
        summary = run_case(
            cases[name],
            out_base=out_base,
            gpu=str(args.gpu),
            steps=args.steps,
            log_freq=args.log_freq,
            nproc_per_node=args.nproc_per_node,
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=str(args.master_port),
            rdzv_id=args.rdzv_id,
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
            f"loss={summary.get('last_loss')} peak_mfu={summary.get('peak_mfu')} "
            f"steady_mfu={summary.get('steady_mfu')} log={summary['log']}",
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
