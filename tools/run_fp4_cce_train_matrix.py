#!/usr/bin/env python3
"""Run a short single-rank training matrix for internal-loss CCE backends."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "train_configs"
    / "ablations"
    / "fp4_cce_slimpajama"
    / "8b"
    / "final_layer_cce"
    / "nvfp4_v4_pcache_matrix.toml"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "low_bits_training" / "cce"
FAST_FP4_CCE_V4_ROOT = Path("/tmp/fp4_matmul_v4_pcache")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[-+0-9.eE]+)\s+"
    r"grad_norm:\s*(?P<grad_norm>[-+0-9.eE]+)\s+"
    r"memory:\s*(?P<memory_gib>[-+0-9.eE]+)GiB"
    r".*?tps:\s*(?P<tps>[0-9,]+)\s+"
    r"tflops:\s*(?P<tflops>[0-9.,]+)\s+"
    r"mfu:\s*(?P<mfu>[-+0-9.eE]+)%"
)
BF16_EVAL_RE = re.compile(r"eval_bf16/loss:\s*(?P<loss>[-+0-9.eE]+)")


@dataclass(frozen=True)
class Variant:
    label: str
    cli_args: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


NVFP4_V4_PCACHE_ENV = (
    ("FP4_CCE_ASSUME_NONEMPTY_LABELS", "1"),
    ("FP4_CCE_NVFP4_EXACT_NORM_QUANT", "0"),
    ("FP4_CCE_V4_FUSED_X_PRODUCER", "1"),
    ("FP4_CCE_V4_NVFP4_FUSED_X_PRODUCER", "1"),
    ("FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED", "1"),
    ("FP4_CCE_V4_NVFP4_G_CACHE", "0"),
    ("FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE", "1"),
    ("FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE", "1"),
)
MXFP4_V4_PCACHE_ENV = (
    ("FP4_CCE_ASSUME_NONEMPTY_LABELS", "1"),
    ("FP4_CCE_V4_FUSED_X_PRODUCER", "1"),
    ("FP4_CCE_V4_MXFP4_FUSED_X_PRODUCER", "1"),
    ("FP4_CCE_V4_NVFP4_GEMM_CONFIG_TUNED", "1"),
    ("FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE", "1"),
    ("FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE", "1"),
)
NVFP4_V4_PCACHE_DYNAMIC_P_ENV = (
    *NVFP4_V4_PCACHE_ENV,
    ("FP4_CCE_V4_NVFP4_P_CONSTANT_SCALE", "0"),
)
NVFP4_V4_PCACHE_DYNAMIC_P_SR_ENV = (
    *NVFP4_V4_PCACHE_DYNAMIC_P_ENV,
    ("FP4_CCE_V4_NVFP4_P_DATA_SR", "1"),
    ("FP4_CCE_V4_NVFP4_P_SCALE_SR", "0"),
)
NVFP4_V4_PCACHE_TARGET_SPLIT_ENV = (
    *NVFP4_V4_PCACHE_DYNAMIC_P_ENV,
    ("FP4_CCE_V4_NVFP4_P_TARGET_SPLIT", "1"),
    ("FP4_CCE_V4_STRICT_FUSED_SPARSE", "1"),
)
NVFP4_V4_PCACHE_TARGET_SPLIT_SR_ENV = (
    *NVFP4_V4_PCACHE_TARGET_SPLIT_ENV,
    ("FP4_CCE_V4_NVFP4_P_DATA_SR", "1"),
    ("FP4_CCE_V4_NVFP4_P_SCALE_SR", "0"),
)
NVFP4_V4_PCACHE_TARGET_TOP1_SPLIT_SR_ENV = (
    *NVFP4_V4_PCACHE_TARGET_SPLIT_SR_ENV,
    ("FP4_CCE_V4_NVFP4_P_TOP1_SPLIT", "1"),
)
NVFP4_V4_GCACHE_UNIT_BOUND_ENV = (
    *NVFP4_V4_PCACHE_ENV,
    ("FP4_CCE_V4_NVFP4_G_CACHE", "1"),
)
NVFP4_V4_GCACHE_DYNAMIC_ENV = (
    *NVFP4_V4_GCACHE_UNIT_BOUND_ENV,
    ("FP4_CCE_V4_NVFP4_G_CONSTANT_SCALE", "0"),
)


HISTORICAL_VARIANTS = [
    Variant("triton-bf16", ("--fp4_cce.enabled", "--fp4_cce.backend=triton_bf16")),
    Variant(
        "triton-bf16-auto",
        ("--fp4_cce.enabled", "--fp4_cce.backend=triton_bf16", "--fp4_cce.filter_eps=auto"),
    ),
    Variant(
        "bf16-torch-compile",
        ("--fp4_cce.enabled", "--fp4_cce.backend=torch_compile_bf16"),
    ),
    Variant(
        "nv-v2-enc",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
        ),
    ),
    Variant(
        "nv-v2-dec",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=dec",
        ),
    ),
    Variant(
        "nv-v2-auto",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
    ),
    Variant(
        "nv-v2-enc-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
        ),
        env=(("LBT_NV_TRUE_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "nv-v3-enc",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
        ),
    ),
    Variant(
        "nv-v3-dec",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=dec",
        ),
    ),
    Variant(
        "nv-v3-auto",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
    ),
    Variant(
        "nv-v4-enc",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
    ),
    Variant(
        "nv-v3-enc-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
        ),
        env=(("LBT_NV_TRUE_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "mx-v2-enc",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
        ),
    ),
    Variant(
        "mx-v2-dec",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=dec",
        ),
    ),
    Variant(
        "mx-v2-auto",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
    ),
    Variant(
        "mx-v2-enc-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
        ),
        env=(("LBT_MX_TRUE_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "mx-v3-enc",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
        ),
    ),
    Variant(
        "mx-v3-dec",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=dec",
        ),
    ),
    Variant(
        "mx-v3-auto",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
    ),
    Variant(
        "mx-v4-enc",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
    ),
    Variant(
        "mx-v4-dec",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=dec",
        ),
    ),
    Variant(
        "mx-v3-enc-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
        ),
        env=(("LBT_MX_TRUE_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "nv-v2-auto-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_NV_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "nv-v3-auto-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_NV_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "mx-v2-auto-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_MX_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "mx-v3-auto-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_MX_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "nv-v2-auto-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_NV_TRUE_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "nv-v3-auto-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_NV_TRUE_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "mx-v2-auto-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v2",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_MX_TRUE_NUCLEAR_BWD", "1"),),
    ),
    Variant(
        "mx-v3-auto-true-nuclear",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v3",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.filter_eps=auto",
        ),
        env=(("LBT_MX_TRUE_NUCLEAR_BWD", "1"),),
    ),
]
NATIVE_PRECISION_VARIANTS = [
    Variant(
        "native-bf16-fwd-bf16-bwd",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=native_mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.forward_precision=bf16",
            "--fp4_cce.backward_precision=bf16",
        ),
    ),
    Variant(
        "native-fp4-fwd-bf16-bwd",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=native_mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.forward_precision=fp4",
            "--fp4_cce.backward_precision=bf16",
        ),
    ),
    Variant(
        "native-bf16-fwd-fp4-bwd",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=native_mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.forward_precision=bf16",
            "--fp4_cce.backward_precision=fp4",
        ),
    ),
    Variant(
        "native-fp4-fwd-fp4-bwd",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=native_mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.forward_precision=fp4",
            "--fp4_cce.backward_precision=fp4",
        ),
    ),
]
DEFAULT_VARIANTS = [
    Variant(
        "native-bf16-control",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=native_mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
            "--fp4_cce.forward_precision=bf16",
            "--fp4_cce.backward_precision=bf16",
        ),
    ),
    Variant(
        "nv-v4-pcache",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_PCACHE_ENV,
    ),
    Variant(
        "mx-v4-pcache",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=mxfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=MXFP4_V4_PCACHE_ENV,
    ),
]
PCACHE_ABLATION_VARIANTS = [
    Variant(
        "nv-v4-pcache-dynamic-p",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_PCACHE_DYNAMIC_P_ENV,
    ),
    Variant(
        "nv-v4-pcache-dynamic-p-sr",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_PCACHE_DYNAMIC_P_SR_ENV,
    ),
    Variant(
        "nv-v4-pcache-target-split",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_PCACHE_TARGET_SPLIT_ENV,
    ),
    Variant(
        "nv-v4-pcache-target-split-sr",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_PCACHE_TARGET_SPLIT_SR_ENV,
    ),
    Variant(
        "nv-v4-pcache-target-top1-split-sr",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_PCACHE_TARGET_TOP1_SPLIT_SR_ENV,
    ),
    Variant(
        "nv-v4-gcache-unit-bound",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_GCACHE_UNIT_BOUND_ENV,
    ),
    Variant(
        "nv-v4-gcache-dynamic",
        (
            "--fp4_cce.enabled",
            "--fp4_cce.backend=nvfp4",
            "--fp4_cce.implementation=v4",
            "--fp4_cce.quant_mode=enc",
        ),
        env=NVFP4_V4_GCACHE_DYNAMIC_ENV,
    ),
]
VARIANT_BY_LABEL = {
    variant.label: variant
    for variant in [
        *HISTORICAL_VARIANTS,
        *NATIVE_PRECISION_VARIANTS,
        *DEFAULT_VARIANTS,
        *PCACHE_ABLATION_VARIANTS,
    ]
}


def default_fp4_matmul_root() -> str:
    env_root = os.environ.get("FP4_MATMUL_ROOT")
    if env_root:
        return str(Path(env_root).expanduser().resolve())

    candidates = [
        REPO_ROOT.parent / "fp4_matmul",
        Path("/opt/mfu/EXTERNAL_PATH"),
        Path("/opt/mfu/EXTERNAL_PATH"),
        FAST_FP4_CCE_V4_ROOT,
        REPO_ROOT.parent / "cce" / "fp4_matmul",
        Path("/opt/mfu/EXTERNAL_PATH"),
        Path("/opt/mfu/EXTERNAL_PATH"),
        Path("/opt/mfu/EXTERNAL_PATH"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((REPO_ROOT.parent / "fp4_matmul").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--variants",
        default=",".join(variant.label for variant in DEFAULT_VARIANTS),
        help="Comma-separated variant labels.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--master-port", type=int, default=29541)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label", default="debug")
    parser.add_argument(
        "--common-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log a common native CUDA BF16 eval loss on the same batch.",
    )
    parser.add_argument(
        "--common-eval-filter-eps",
        default=None,
        help="Legacy option; native CUDA BF16 common eval ignores filter_eps.",
    )
    parser.add_argument(
        "--common-eval-every",
        type=int,
        default=50,
        help="Optional frequency for inline common BF16 eval logging.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional train.py CLI override. May be passed multiple times.",
    )
    return parser.parse_args()


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_step_metrics(log_text: str) -> list[dict]:
    rows = []
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


def parse_eval_bf16_metrics(log_text: str) -> list[float]:
    rows = []
    for line in strip_ansi(log_text).splitlines():
        match = BF16_EVAL_RE.search(line)
        if not match:
            continue
        rows.append(float(match.group("loss")))
    return rows


def render_markdown(args: argparse.Namespace, results: list[dict]) -> str:
    lines = []
    lines.append(f"# FP4 CCE Training Matrix ({args.label})")
    lines.append("")
    lines.append(f"- Date: {date.today().isoformat()}")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Steps: `{args.steps}`")
    lines.append(f"- Variants: `{', '.join(result['label'] for result in results)}`")
    lines.append("")
    lines.append("| Variant | Status | Last Step | Last Loss | Last BF16 Eval Loss | Mean TPS | Steady TPS | Last TPS | Steady TFLOPS | Steady MFU (%) | Last MFU (%) | Wall Time (s) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        lines.append(
            "| {label} | {status} | {last_step} | {last_loss} | {last_eval_bf16_loss} | {mean_tps} | {steady_tps} | {last_tps} | {steady_tflops} | {steady_mfu} | {last_mfu} | {wall_s} |".format(
                label=result["label"],
                status=result["status"],
                last_step=result.get("last_step", "-"),
                last_loss=result.get("last_loss_fmt", "-"),
                last_eval_bf16_loss=result.get("last_eval_bf16_loss_fmt", "-"),
                mean_tps=result.get("mean_tps_fmt", "-"),
                steady_tps=result.get("steady_tps_fmt", "-"),
                last_tps=result.get("last_tps_fmt", "-"),
                steady_tflops=result.get("steady_tflops_fmt", "-"),
                steady_mfu=result.get("steady_mfu_fmt", "-"),
                last_mfu=result.get("last_mfu_fmt", "-"),
                wall_s=result.get("wall_time_s_fmt", "-"),
            )
        )
    lines.append("")
    for result in results:
        lines.append(f"## {result['label']}")
        lines.append("")
        lines.append(f"- Status: `{result['status']}`")
        lines.append(f"- Command: `{result['command']}`")
        if result["status"] == "OK":
            lines.append(f"- Last step: `{result['last_step']}`")
            lines.append(f"- Last loss: `{result['last_loss_fmt']}`")
            if "last_eval_bf16_loss_fmt" in result:
                lines.append(f"- Last BF16 eval loss: `{result['last_eval_bf16_loss_fmt']}`")
            lines.append(f"- Mean TPS: `{result['mean_tps_fmt']}`")
            lines.append(f"- Steady TPS: `{result['steady_tps_fmt']}`")
            lines.append(f"- Last TPS: `{result['last_tps_fmt']}`")
            lines.append(f"- Steady TFLOPS: `{result['steady_tflops_fmt']}`")
            lines.append(f"- Mean MFU: `{result['mean_mfu_fmt']}%`")
            lines.append(f"- Steady MFU: `{result['steady_mfu_fmt']}%`")
            lines.append(f"- Last MFU: `{result['last_mfu_fmt']}%`")
            lines.append(f"- Wall time: `{result['wall_time_s_fmt']} s`")
        else:
            lines.append(f"- Return code: `{result.get('returncode', '-')}`")
            lines.append(f"- Error: `{result.get('error', '').strip()}`")
        lines.append(f"- Log: [{Path(result['log_path']).name}]({result['log_path']})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    for label in [item.strip() for item in args.variants.split(",") if item.strip()]:
        if label not in VARIANT_BY_LABEL:
            raise SystemExit(f"Unknown variant label: {label}")
        selected.append(VARIANT_BY_LABEL[label])

    results = []
    for variant in selected:
        run_name = f"{args.label}_{variant.label}"
        run_dump = output_dir / f"train_run_{run_name}"
        run_dump.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"FP4_CCE_TRAIN_{run_name}_{date.today().isoformat()}.log"
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            f"--master_port={args.master_port}",
            "train.py",
            f"--job.config_file={config_path}",
            f"--job.dump_folder={run_dump}",
            f"--job.description=FP4 CCE train matrix {variant.label}",
            f"--training.steps={args.steps}",
            f"--job.steps={args.steps}",
            "--metrics.log_freq=1",
            *args.extra_arg,
            *variant.cli_args,
        ]
        env = os.environ.copy()
        if args.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("PARI_NO_SIGNAL", "1")
        env.setdefault("USE_LBT_SAFE_FAST_EXIT", "1")
        env.setdefault("FP4_MATMUL_ROOT", default_fp4_matmul_root())
        if args.common_eval:
            env["LBT_FP4_CCE_COMMON_EVAL"] = "1"
            if args.common_eval_filter_eps is not None:
                env["LBT_FP4_CCE_COMMON_EVAL_FILTER_EPS"] = args.common_eval_filter_eps
            if args.common_eval_every is not None:
                env["LBT_FP4_CCE_COMMON_EVAL_EVERY"] = str(args.common_eval_every)
        for key, value in variant.env:
            env[key] = value

        start = time.perf_counter()
        log_path.write_text("")
        try:
            with log_path.open("w") as log_handle:
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.timeout_sec,
                )
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            with log_path.open("a") as log_handle:
                log_handle.write(
                    "\n[run_fp4_cce_train_matrix] timed out after "
                    f"{args.timeout_sec}s\n"
                )
            returncode = 124
            timed_out = True
        wall_time_s = time.perf_counter() - start
        stdout = log_path.read_text(errors="replace")

        result = {
            "label": variant.label,
            "command": " ".join(cmd),
            "log_path": str(log_path),
            "returncode": returncode,
            "wall_time_s": wall_time_s,
            "wall_time_s_fmt": f"{wall_time_s:.2f}",
        }

        if returncode != 0:
            result["status"] = "TIMEOUT" if timed_out else "FAIL"
            result["error"] = strip_ansi(stdout)[-2000:]
            results.append(result)
            continue

        rows = parse_step_metrics(stdout)
        if not rows:
            result["status"] = "NO_METRICS"
            result["error"] = "No training step metrics found in log output."
            results.append(result)
            continue
        eval_bf16_losses = parse_eval_bf16_metrics(stdout)

        mean_tps = sum(row["tps"] for row in rows) / len(rows)
        mean_tflops = sum(row["tflops"] for row in rows) / len(rows)
        mean_mfu = sum(row["mfu"] for row in rows) / len(rows)
        steady_rows = rows[1:] if len(rows) > 1 else rows
        steady_tps = sum(row["tps"] for row in steady_rows) / len(steady_rows)
        steady_tflops = sum(row["tflops"] for row in steady_rows) / len(steady_rows)
        steady_mfu = sum(row["mfu"] for row in steady_rows) / len(steady_rows)
        last = rows[-1]
        result.update(
            {
                "status": "OK",
                "last_step": last["step"],
                "last_loss": last["loss"],
                "last_loss_fmt": f"{last['loss']:.4f}",
                "last_tps": last["tps"],
                "last_tps_fmt": f"{last['tps']:.0f}",
                "mean_tps": mean_tps,
                "mean_tps_fmt": f"{mean_tps:.0f}",
                "steady_tps": steady_tps,
                "steady_tps_fmt": f"{steady_tps:.0f}",
                "mean_tflops": mean_tflops,
                "steady_tflops": steady_tflops,
                "steady_tflops_fmt": f"{steady_tflops:.2f}",
                "mean_mfu": mean_mfu,
                "mean_mfu_fmt": f"{mean_mfu:.2f}",
                "steady_mfu": steady_mfu,
                "steady_mfu_fmt": f"{steady_mfu:.2f}",
                "last_mfu": last["mfu"],
                "last_mfu_fmt": f"{last['mfu']:.2f}",
                "rows": rows,
            }
        )
        if eval_bf16_losses:
            result.update(
                {
                    "last_eval_bf16_loss": eval_bf16_losses[-1],
                    "last_eval_bf16_loss_fmt": f"{eval_bf16_losses[-1]:.4f}",
                    "eval_bf16_losses": eval_bf16_losses,
                }
            )
        results.append(result)

    json_path = output_dir / f"FP4_CCE_TRAIN_MATRIX_{args.label}_{date.today().isoformat()}.json"
    md_path = output_dir / f"FP4_CCE_TRAIN_MATRIX_{args.label}_{date.today().isoformat()}.md"
    json_path.write_text(json.dumps(results, indent=2))
    md_path.write_text(render_markdown(args, results))

    print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")
    for result in results:
        print(
            f"{result['label']}: {result['status']} "
            f"loss={result.get('last_loss_fmt', '-')} "
            f"eval_bf16={result.get('last_eval_bf16_loss_fmt', '-')} "
            f"steady_tps={result.get('steady_tps_fmt', '-')} "
            f"wall_s={result['wall_time_s_fmt']}"
        )
    return 0 if all(result["status"] == "OK" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
