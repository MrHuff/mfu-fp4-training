#!/usr/bin/env python3
"""Run shape-correct FP4 attribution for synthetic 1B block benchmarks.

This tool keeps all comparisons on the canonical harness and combines:
- low-bits isolated/full block timings
- low-bits profiler summaries and copy reports
- parent exact-shape localCTA / MXFP4 microbenches
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FP4_MATMUL_ROOT = REPO_ROOT.parent / "fp4_matmul"
BENCH = REPO_ROOT / "tools" / "bench_synth_1b_fp4.py"
PROFILE = REPO_ROOT / "tools" / "profile_synth_1b_fp4.py"
PARENT_BENCH = FP4_MATMUL_ROOT / "ThunderKittens" / "kernels" / "gemm" / "bench_fp4_model_shapes.py"

MODES = [
    "fp4_fused_te",
    "fp4_tk",
    "fp4_localcta",
    "fp4_localcta_fused",
    "mxfp4_tk_fused",
]
BLOCKS = ["qkv", "wo", "ffn", "full"]
PROFILE_BLOCKS = ["qkv", "ffn"]

COPY_KEYS = {"aten::copy_", "aten::contiguous", "aten::to", "aten::_to_copy", "aten::cat", "cudaMalloc"}
GEMM_HINTS = ("gemm", "grouped_gemm", "batched_gemm", "onepass", "mxfp4", "nvfp4")
QUANT_HINTS = ("quant", "fused_norm", "rmsnorm", "reconstruct")


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return proc.stdout


def _extract_last_json_blob(text: str):
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise RuntimeError(f"no trailing JSON object found in output:\n{text}")


def _copy_score(profile: dict) -> float:
    total = 0.0
    for row in profile.get("copy_rows") or []:
        total += float(row.get("duration_ms", 0.0))
    for event in profile.get("top_events", []):
        if event["key"] in COPY_KEYS:
            total += event.get("self_cpu_us", 0.0) / 1000.0
            total += event.get("self_cuda_us", 0.0) / 1000.0
    return total


def _gemm_score(profile: dict) -> float:
    total = 0.0
    for event in profile.get("top_events", []):
        key = event["key"].lower()
        if any(hint in key for hint in GEMM_HINTS):
            total += event.get("self_cuda_us", 0.0) / 1000.0
            total += event.get("self_cpu_us", 0.0) / 1000.0
    return total


def _quant_score(profile: dict) -> float:
    total = 0.0
    for event in profile.get("top_events", []):
        key = event["key"].lower()
        if any(hint in key for hint in QUANT_HINTS):
            total += event.get("self_cuda_us", 0.0) / 1000.0
            total += event.get("self_cpu_us", 0.0) / 1000.0
    return total


def _classify(profile: dict | None, bench_total_ms: float, parent_rows: list[dict] | None) -> dict[str, object]:
    if profile is None:
        return {"classification": "unknown", "reason": "no profile summary"}

    copy_ms = _copy_score(profile)
    gemm_ms = _gemm_score(profile)
    quant_ms = _quant_score(profile)

    parent_fast = None
    if parent_rows:
        fast_values = []
        for row in parent_rows:
            if "fast_ms" in row:
                fast_values.append(float(row["fast_ms"]))
            elif "default_ms" in row:
                fast_values.append(float(row["default_ms"]))
        if fast_values:
            parent_fast = sum(fast_values)

    if copy_ms >= max(gemm_ms, quant_ms) and copy_ms >= 0.5:
        classification = "copy/wrapper"
        reason = f"copy-like ops/callers ~= {copy_ms:.3f} ms"
    elif gemm_ms >= max(copy_ms, quant_ms):
        classification = "gemm/config"
        reason = f"gemm-like ops dominate ~= {gemm_ms:.3f} ms"
    else:
        classification = "quant/prep"
        reason = f"quant/norm-like ops dominate ~= {quant_ms:.3f} ms"

    if parent_fast is not None and bench_total_ms > parent_fast * 2.5 and classification == "gemm/config":
        classification = "copy/wrapper"
        reason = (
            f"block total {bench_total_ms:.3f} ms is far above parent fast path "
            f"{parent_fast:.3f} ms; likely wrapper/control-plane overhead"
        )

    return {
        "classification": classification,
        "reason": reason,
        "copy_ms": round(copy_ms, 6),
        "gemm_ms": round(gemm_ms, 6),
        "quant_ms": round(quant_ms, 6),
        "parent_fast_sum_ms": round(parent_fast, 6) if parent_fast is not None else None,
    }


def _parent_rows_for_block(parent_report: dict, family: str, block: str) -> list[dict]:
    rows = [row for row in parent_report.get("rows", []) if row.get("family") == family]
    if family == "localcta":
        if block == "qkv":
            return [row for row in rows if row["case"] == "qkv_grouped_fwd"]
        if block == "wo":
            return [row for row in rows if row["case"] == "wo_fwd"]
        if block == "ffn":
            return [row for row in rows if row["case"] in {"ffn_w13_fwd", "ffn_w2_fwd"}]
    if family == "mxfp4":
        if block == "wo":
            return [row for row in rows if row["case"] == "wo_fwd"]
        if block == "ffn":
            return [row for row in rows if row["case"] in {"ffn_w2_fwd", "ffn_split2_onepass_dgrad"}]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--flavors", nargs="+", default=["1B", "1B_legacy"], choices=["1B", "1B_legacy"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--isolation-m", type=int, default=65536)
    parser.add_argument("--row-limit", type=int, default=12)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base_env = {"USE_TK_LOCALCTA_VARIANT": "v1"}
    report = {
        "device_index": args.device_index,
        "warmup": args.warmup,
        "steps": args.steps,
        "isolation_m": args.isolation_m,
        "shapes": {},
    }

    with tempfile.TemporaryDirectory(prefix="fp4_shape_attr_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for flavor in args.flavors:
            flavor_report = {
                "bench": {},
                "profiles": {},
                "parent": {},
                "attribution": {},
            }

            for mode in MODES:
                mode_report = {}
                for block in BLOCKS:
                    out = _run(
                        [
                            sys.executable,
                            "-u",
                            str(BENCH),
                            "--mode",
                            mode,
                            "--flavor",
                            flavor,
                            "--block",
                            block,
                            "--warmup",
                            str(args.warmup),
                            "--steps",
                            str(args.steps),
                            "--device-index",
                            str(args.device_index),
                            "--isolation-m",
                            str(args.isolation_m),
                        ],
                        REPO_ROOT,
                        base_env,
                    )
                    mode_report[block] = _extract_last_json_blob(out)
                flavor_report["bench"][mode] = mode_report

            for family in ("localcta", "mxfp4"):
                out = _run(
                    [
                        sys.executable,
                        "-u",
                        str(PARENT_BENCH),
                        "--flavor",
                        flavor,
                        "--family",
                        family,
                        "--localcta-variant",
                        "v1",
                        "--warmup",
                        "1",
                        "--iters",
                        "3",
                        "--device",
                        f"cuda:{args.device_index}",
                    ],
                    FP4_MATMUL_ROOT / "ThunderKittens" / "kernels" / "gemm",
                )
                flavor_report["parent"][family] = json.loads(out)

            for mode in MODES:
                prof_mode = {}
                for block in PROFILE_BLOCKS:
                    trace_path = tmp_root / f"{flavor}_{mode}_{block}.json.gz"
                    summary_path = tmp_root / f"{flavor}_{mode}_{block}_summary.json"
                    _run(
                        [
                            sys.executable,
                            "-u",
                            str(PROFILE),
                            "--mode",
                            mode,
                            "--flavor",
                            flavor,
                            "--block",
                            block,
                            "--device-index",
                            str(args.device_index),
                            "--activities",
                            "cpu,cuda",
                            "--row-limit",
                            str(args.row_limit),
                            "--export-trace",
                            str(trace_path),
                            "--copy-report",
                            "--summary-json",
                            str(summary_path),
                        ],
                        REPO_ROOT,
                        base_env,
                    )
                    with open(summary_path, "r", encoding="utf-8") as f:
                        prof_mode[block] = json.load(f)
                flavor_report["profiles"][mode] = prof_mode

            for mode in MODES:
                attr_mode = {}
                for block in BLOCKS:
                    bench_row = flavor_report["bench"][mode][block]
                    profile_row = flavor_report["profiles"][mode].get(block)
                    family = None
                    if mode in {"fp4_localcta", "fp4_localcta_fused"}:
                        family = "localcta"
                    elif mode == "mxfp4_tk_fused":
                        family = "mxfp4"
                    parent_rows = _parent_rows_for_block(flavor_report["parent"].get(family, {}), family, block) if family else None
                    attr_mode[block] = {
                        "total_ms": bench_row["total_ms"],
                        "shape": {
                            "q_dim": bench_row["q_dim"],
                            "k_dim": bench_row["k_dim"],
                            "v_dim": bench_row["v_dim"],
                            "hidden_dim": bench_row["hidden_dim"],
                        },
                        "parent_rows": parent_rows,
                        **_classify(profile_row, bench_row["total_ms"], parent_rows),
                    }
                flavor_report["attribution"][mode] = attr_mode

            report["shapes"][flavor] = flavor_report

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
