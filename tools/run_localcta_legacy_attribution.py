#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "tools" / "run_fp4_training_matrix.py"

_spec = importlib.util.spec_from_file_location("run_fp4_training_matrix", RUNNER_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Unable to import matrix runner from {RUNNER_PATH}")
matrix = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = matrix
_spec.loader.exec_module(matrix)


BACKEND_TRACE_MARKER = "[TK BACKEND]"
LOCALCTA_PARAM_MARKER = "[LBT LOCALCTA PARAM]"
LOCALCTA_FUNC_MARKER = "[LBT LOCALCTA FUNC]"
DEFAULT_FP4_ROOT = "/tmp/fp4_matmul_main_0406"


@dataclass(frozen=True)
class Scenario:
    name: str
    env: dict[str, str]
    steps: int = 10


SCENARIOS: dict[str, Scenario] = {
    "baseline_fast": Scenario(
        name="baseline_fast",
        env={},
    ),
    "baseline_fast_wo_prepared": Scenario(
        name="baseline_fast_wo_prepared",
        env={"USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1"},
    ),
    "attn_only": Scenario(
        name="attn_only",
        env={
            "FP4_ATTN_BACKEND": "localcta",
            "FP4_FFN_BACKEND": "te",
        },
    ),
    "attn_only_wo_prepared": Scenario(
        name="attn_only_wo_prepared",
        env={
            "FP4_ATTN_BACKEND": "localcta",
            "FP4_FFN_BACKEND": "te",
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
        },
    ),
    "ffn_only": Scenario(
        name="ffn_only",
        env={
            "FP4_ATTN_BACKEND": "te",
            "FP4_FFN_BACKEND": "localcta",
        },
    ),
    "full_split2": Scenario(
        name="full_split2",
        env={"USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2": "1"},
    ),
    "full_split2_wo_prepared": Scenario(
        name="full_split2_wo_prepared",
        env={
            "USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2": "1",
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
        },
    ),
    "full_split2_fused_row": Scenario(
        name="full_split2_fused_row",
        env={
            "USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2": "1",
            "USE_TK_LOCALCTA_FFN_FUSED_ROW_PRODUCER": "1",
        },
    ),
    "ffn_only_split2": Scenario(
        name="ffn_only_split2",
        env={
            "FP4_ATTN_BACKEND": "te",
            "FP4_FFN_BACKEND": "localcta",
            "USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2": "1",
        },
    ),
    "full_saved_sig_overlap": Scenario(
        name="full_saved_sig_overlap",
        env={
            "USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2": "1",
            "USE_TK_LOCALCTA_FFN_EXPERIMENT": "saved_sigmoid_overlap",
        },
    ),
    "full_saved_sig_overlap_w2highacc": Scenario(
        name="full_saved_sig_overlap_w2highacc",
        env={
            "USE_TK_LOCALCTA_FFN_ENABLE_DIRECT_SPLIT2": "1",
            "USE_TK_LOCALCTA_FFN_EXPERIMENT": "saved_sigmoid_overlap_w2highacc",
        },
    ),
    "bf16_wgrad": Scenario(
        name="bf16_wgrad",
        env={"USE_TK_QKV_BF16_WGRAD": "1"},
    ),
    "wo_prepared_bf16_wgrad": Scenario(
        name="wo_prepared_bf16_wgrad",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_BF16_WGRAD": "1",
        },
    ),
    "attn_only_bf16_dgrad": Scenario(
        name="attn_only_bf16_dgrad",
        env={
            "FP4_ATTN_BACKEND": "localcta",
            "FP4_FFN_BACKEND": "te",
            "USE_TK_QKV_BF16_DGRAD": "1",
        },
    ),
    "wo_prepared_bf16_dgrad": Scenario(
        name="wo_prepared_bf16_dgrad",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_BF16_DGRAD": "1",
        },
    ),
    "wo_prepared_bf16_rmsnorm_bwd": Scenario(
        name="wo_prepared_bf16_rmsnorm_bwd",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_BF16_RMSNORM_BWD": "1",
        },
    ),
    "wo_prepared_bf16_dgrad_rmsnorm_bwd": Scenario(
        name="wo_prepared_bf16_dgrad_rmsnorm_bwd",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_BF16_DGRAD": "1",
            "USE_TK_QKV_BF16_RMSNORM_BWD": "1",
        },
    ),
    "wo_prepared_tk_prepared_act": Scenario(
        name="wo_prepared_tk_prepared_act",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_LOCALCTA_TK_PREPARED_ACT": "1",
        },
    ),
    "wo_prepared_dgrad_strided_sum": Scenario(
        name="wo_prepared_dgrad_strided_sum",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "strided_sum",
        },
    ),
    "wo_prepared_dgrad_split3": Scenario(
        name="wo_prepared_dgrad_split3",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "split3",
        },
    ),
    "wo_prepared_dgrad_batched_accum": Scenario(
        name="wo_prepared_dgrad_batched_accum",
        env={
            "USE_TK_LOCALCTA_WO_PREPARED_SPLIT2_BWD": "1",
            "USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "batched_accum",
        },
    ),
    "safe_qkv_fwd_sync": Scenario(
        name="safe_qkv_fwd_sync",
        env={"USE_TK_ATTN_SAFE_QKV_FWD_SYNC": "1"},
    ),
    "bf16_wgrad_sync": Scenario(
        name="bf16_wgrad_sync",
        env={
            "USE_TK_QKV_BF16_WGRAD": "1",
            "USE_TK_ATTN_SAFE_QKV_FWD_SYNC": "1",
        },
    ),
    "dgrad_strided_sum": Scenario(
        name="dgrad_strided_sum",
        env={"USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "strided_sum"},
    ),
    "dgrad_split3": Scenario(
        name="dgrad_split3",
        env={"USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "split3"},
    ),
    "dgrad_batched_accum": Scenario(
        name="dgrad_batched_accum",
        env={"USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "batched_accum"},
    ),
    "dgrad_direct_split": Scenario(
        name="dgrad_direct_split",
        env={"USE_TK_QKV_LOCALCTA_DGRAD_BACKEND": "direct_split"},
    ),
}


def _localcta_case_names() -> list[str]:
    return [c.name for c in matrix.MFU_LEGACY_CASES if c.name in {"localcta_legacy", "localcta_fused_legacy"}]


def _load_localcta_case(name: str) -> matrix.Case:
    return next(c for c in matrix.MFU_LEGACY_CASES if c.name == name)


def _localcta_backend_name(base: matrix.Case) -> str:
    return "localcta_fused" if base.env.get("USE_TK_LOCALCTA_FUSED") == "1" else "localcta"


def _tail_median_mfu(steps: list[dict[str, Any]]) -> float | None:
    usable = [row["mfu"] for row in steps if row["step"] >= 2]
    if not usable:
        return None
    return float(matrix.statistics.median(usable))


def _extract_backend_trace_lines(log_path: Path) -> list[str]:
    traces: list[str] = []
    if not log_path.exists():
        return traces
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if BACKEND_TRACE_MARKER in line:
            traces.append(line.strip())
    return traces


def _extract_json_marker_events(log_path: Path, marker: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not log_path.exists():
        return events
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        payload = line.split(marker, 1)[1].strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _summarize_localcta_param_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_kind.setdefault(str(event.get("kind", "unknown")), []).append(event)
    final_by_kind = {kind: items[-1] for kind, items in by_kind.items() if items}
    return {
        "events": events,
        "event_count": len(events),
        "kinds": sorted(by_kind),
        "final_by_kind": final_by_kind,
    }


def _scenario_env_for_base(base: matrix.Case, scenario: Scenario) -> dict[str, str]:
    env = dict(scenario.env)
    backend = _localcta_backend_name(base)
    for key in ("FP4_ATTN_BACKEND", "FP4_FFN_BACKEND"):
        if env.get(key) == "localcta":
            env[key] = backend
    return env


def _build_case(base: matrix.Case, scenario: Scenario, steps: int, seed: int) -> matrix.Case:
    env = dict(base.env)
    env.pop("USE_TK_QKV_LOCALCTA_TK_PREPARED_ACT", None)
    env.update(_scenario_env_for_base(base, scenario))
    if env.get("USE_TK_LOCALCTA") == "1":
        env["USE_TK_LOCALCTA_DIRECT_CONTRACT"] = "0"
    env["USE_TK_LOCALCTA_BACKEND_TRACE"] = "1"
    env["USE_TK_DEBUG_LOG_LOCALCTA_PARAM_GRADS"] = "1"
    env["USE_TK_DEBUG_LOG_LOCALCTA_FUNCTION_GRADS"] = "1"
    return matrix.Case(
        name=f"{base.name}_{scenario.name}",
        family=base.family,
        config=base.config,
        env=env,
        overrides=["--training.steps", str(steps), "--debug.seed", str(seed)],
        allow_failure=True,
    )


def _summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    steps = result["steps"]
    log_path = Path(result["log_path"])
    localcta_param_events = _extract_json_marker_events(log_path, LOCALCTA_PARAM_MARKER)
    localcta_func_events = _extract_json_marker_events(log_path, LOCALCTA_FUNC_MARKER)
    summary = {
        "name": result["name"],
        "completed": result["completed"],
        "returncode": result["returncode"],
        "timed_out": result["timed_out"],
        "final_step": result["final_step"],
        "first_nonfinite_step": result["first_nonfinite_step"],
        "duration_s": result["duration_s"],
        "log_path": result["log_path"],
        "backend_trace": _extract_backend_trace_lines(log_path),
        "tail_median_mfu": _tail_median_mfu(steps),
        "localcta_param_debug": _summarize_localcta_param_events(localcta_param_events),
        "localcta_function_debug": _summarize_localcta_param_events(localcta_func_events),
    }
    if steps:
        summary["step1"] = steps[0]
        summary["final"] = steps[-1]
        summary["loss_delta"] = steps[-1]["loss"] - steps[0]["loss"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-case",
        default="localcta_legacy",
        choices=_localcta_case_names(),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["baseline_fast", "attn_only", "ffn_only", "full_split2"],
        choices=sorted(SCENARIOS.keys()),
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--fp4-root",
        default=DEFAULT_FP4_ROOT,
    )
    parser.add_argument(
        "--out-root",
        default=f"/tmp/localcta_legacy_attr_{matrix.time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_case = _load_localcta_case(args.base_case)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    matrix.BASE_ENV["FP4_MATMUL_ROOT"] = args.fp4_root
    results: list[dict[str, Any]] = []
    for scenario_name in args.scenarios:
        scenario = SCENARIOS[scenario_name]
        case = _build_case(base_case, scenario, args.steps, args.seed)
        raw = matrix.run_case(case, out_root)
        summary = _summarize_result(raw)
        summary["scenario"] = scenario_name
        summary["fp4_root"] = args.fp4_root
        summary["env_overrides"] = _scenario_env_for_base(base_case, scenario)
        results.append(summary)

    payload = {
        "base_case": args.base_case,
        "fp4_root": args.fp4_root,
        "steps": args.steps,
        "seed": args.seed,
        "out_root": str(out_root),
        "results": results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
