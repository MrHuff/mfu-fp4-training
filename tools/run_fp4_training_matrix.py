#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import select
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = REPO_ROOT / "train.py"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+).*?"
    r"loss:\s*(?P<loss>[0-9.]+|nan).*?"
    r"grad_norm:\s*(?P<grad_norm>[0-9.]+|nan).*?"
    r"mfu:\s*(?P<mfu>[0-9.]+)%"
)


def _default_fp4_matmul_root() -> str:
    env_root = os.environ.get("FP4_MATMUL_ROOT")
    if env_root:
        return str(Path(env_root).expanduser().resolve())

    sibling_root = (REPO_ROOT.parent / "fp4_matmul").resolve()
    if sibling_root.exists():
        return str(sibling_root)

    fallback_root = Path("/tmp/fp4_matmul_main_0406")
    if fallback_root.exists():
        return str(fallback_root.resolve())

    return str(sibling_root)


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    config: str
    env: dict[str, str]
    overrides: list[str]
    allow_failure: bool = False


BASE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "CUDA_VISIBLE_DEVICES": "3",
    "WANDB_MODE": "disabled",
    "NVTE_FUSED_ATTN": "0",
    "TORCH_CUDNN_SDPA_ENABLED": "1",
    "LOW_BITS_DISABLE_ATEN_FLASH_PATCH": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_ENCODE_CENTRIC": "0",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "USE_TK_LOCALCTA_VARIANT": "v1",
    "FUSED_TE_QUANT": "0",
    "FP4_MATMUL_ROOT": _default_fp4_matmul_root(),
    "USE_LBT_SAFE_FAST_EXIT": "1",
    "LBT_ENABLE_SIGUSR1_FAULTHANDLER": "1",
    # Canonical matrix runs stay on the real no-fallback baseline.
    "USE_TK_QKV_BF16_WGRAD": "0",
    "USE_TK_QKV_BF16_DGRAD": "0",
    "USE_TK_QKV_BF16_RMSNORM_BWD": "0",
    "USE_TK_ATTN_SAFE_QKV_FWD_SYNC": "0",
    "USE_TK_WO_BF16_WGRAD": "0",
    "USE_TK_WO_ROWONLY_INPUT_QUANT": "0",
    "USE_TK_FFN_BWD_SAFE_PRODUCER": "0",
    "USE_TK_FFN_SAFE_INPUT_QUANT": "0",
    "USE_TK_FFN_FWD_SAFE_PRODUCER": "0",
    "USE_TK_FFN_SPLIT_QUANT_EAGER": "0",
    "USE_TK_FFN_SPLIT_DGRAD_EAGER": "0",
    "USE_TK_FFN_SPLIT_WGRAD_EAGER": "0",
}

MFU_CASE_TIMEOUT_S = 3 * 60
CONVERGENCE_CASE_TIMEOUT_S = 10 * 60


def _config(path: str) -> str:
    return str((REPO_ROOT / path).resolve())


MFU_1B_CASES = [
    Case(
        name="te_unfused_1b",
        family="mfu_1b",
        config=_config("train_configs/llama3_1B_fp4_te_unfused_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="te_fused_1b",
        family="mfu_1b",
        config=_config("train_configs/llama3_1B_fp4_fused_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="fp4_tk_1b",
        family="mfu_1b",
        config=_config("train_configs/llama3_1B_fp4_fused_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="localcta_fused_1b",
        family="mfu_1b",
        config=_config("train_configs/llama3_1B_fp4_fused_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "1",
            "USE_TK_LOCALCTA_FUSED": "1",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="mxfp4_1b",
        family="mfu_1b",
        config=_config("train_configs/llama3_1B_mxfp4_tk_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "1",
            "USE_MXFP4_TK_BACKEND": "0",
            "MXFP4_BACKEND_VERSION": "v4",
            "MXFP4_USE_QKV_DIRECT_OUTPUTS": "1",
            "MXFP4_USE_SPLIT3_QKV_STAGE_COPY": "0",
            "MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD": "0",
            "MXFP4_QKV_BWD_STATE_SLOTS": "4",
            "MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD": "1",
            "MXFP4_USE_FUSED_RMSNORM_QUANT_FFN": "1",
        },
        overrides=[],
        allow_failure=True,
    ),
]


MFU_LEGACY_CASES = [
    Case(
        name="te_fp4_legacy",
        family="mfu_legacy",
        config=_config("train_configs/llama3_1B_legacy_te_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="fp4_fused_ref_legacy",
        family="mfu_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="fp4_tk_legacy",
        family="mfu_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="localcta_legacy",
        family="mfu_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "1",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="localcta_fused_legacy",
        family="mfu_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "1",
            "USE_TK_LOCALCTA_FUSED": "1",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=[],
    ),
    Case(
        name="mxfp4_legacy",
        family="mfu_legacy",
        config=_config("train_configs/llama3_1B_legacy_mxfp4_tk_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "1",
            "USE_MXFP4_TK_BACKEND": "0",
            "MXFP4_BACKEND_VERSION": "v4",
            "MXFP4_USE_QKV_DIRECT_OUTPUTS": "1",
            "MXFP4_USE_SPLIT3_QKV_STAGE_COPY": "0",
            "MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD": "0",
            "MXFP4_QKV_BWD_STATE_SLOTS": "4",
            "MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD": "1",
            "MXFP4_USE_FUSED_RMSNORM_QUANT_FFN": "1",
        },
        overrides=[],
    ),
]


CONVERGENCE_LEGACY_CASES = [
    Case(
        name="te_fp4_legacy",
        family="convergence_legacy",
        config=_config("train_configs/llama3_1B_legacy_te_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="fp4_fused_ref_legacy",
        family="convergence_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="fp4_tk_legacy",
        family="convergence_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="localcta_legacy",
        family="convergence_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "1",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="localcta_fused_legacy",
        family="convergence_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "1",
            "USE_TK_LOCALCTA_FUSED": "1",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="mxfp4_legacy",
        family="convergence_legacy",
        config=_config("train_configs/llama3_1B_legacy_mxfp4_tk_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "1",
            "USE_MXFP4_TK_BACKEND": "0",
            "MXFP4_BACKEND_VERSION": "v4",
            "MXFP4_USE_QKV_DIRECT_OUTPUTS": "1",
            "MXFP4_USE_SPLIT3_QKV_STAGE_COPY": "0",
            "MXFP4_USE_SPLIT3_QKV_ONEPASS_DGRAD": "0",
            "MXFP4_QKV_BWD_STATE_SLOTS": "4",
            "MXFP4_USE_SPLIT2_FFN_ONEPASS_DGRAD": "1",
            "MXFP4_USE_FUSED_RMSNORM_QUANT_FFN": "1",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
]


CONVERGENCE_TK_SPLIT_LEGACY_CASES = [
    Case(
        name="te_fp4_legacy",
        family="convergence_tk_split_legacy",
        config=_config("train_configs/llama3_1B_legacy_te_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "0",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="tk_attn_te_ffn_legacy",
        family="convergence_tk_split_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
            "FP4_ATTN_BACKEND": "tk",
            "FP4_FFN_BACKEND": "te",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
    Case(
        name="te_attn_tk_ffn_legacy",
        family="convergence_tk_split_legacy",
        config=_config("train_configs/llama3_1B_legacy_fp4_matrix.toml"),
        env={
            "USE_TK_GEMM": "1",
            "USE_TK_LOCALCTA": "0",
            "USE_TK_LOCALCTA_FUSED": "0",
            "USE_MXFP4_TK_FUSED": "0",
            "USE_MXFP4_TK_BACKEND": "0",
            "FP4_ATTN_BACKEND": "te",
            "FP4_FFN_BACKEND": "tk",
        },
        overrides=["--training.steps", "100", "--debug.seed", "1234"],
    ),
]


TRACKS = {
    "mfu_1b": MFU_1B_CASES,
    "mfu_legacy": MFU_LEGACY_CASES,
    "convergence_legacy": CONVERGENCE_LEGACY_CASES,
    "convergence_tk_split_legacy": CONVERGENCE_TK_SPLIT_LEGACY_CASES,
}


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_steps(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in strip_ansi(log_path.read_text(errors="replace")).splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group("step"))
        loss_str = match.group("loss")
        grad_str = match.group("grad_norm")
        mfu_str = match.group("mfu")
        rows.append(
            {
                "step": step,
                "loss": None if loss_str == "nan" else float(loss_str),
                "grad_norm": None if grad_str == "nan" else float(grad_str),
                "mfu": float(mfu_str),
                "nonfinite": loss_str == "nan" or grad_str == "nan",
            }
        )
    return rows


def cleanup_stale_processes(marker: str) -> None:
    try:
        proc = subprocess.run(
            ["pgrep", "-af", marker],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def case_timeout_s(case: Case) -> int:
    return CONVERGENCE_CASE_TIMEOUT_S if case.family.startswith("convergence") else MFU_CASE_TIMEOUT_S


def _drain_output(proc: subprocess.Popen[Any], logf, timeout_s: float) -> None:
    if proc.stdout is None:
        return
    while True:
        ready, _, _ = select.select([proc.stdout], [], [], timeout_s)
        if not ready:
            return
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            return
        logf.buffer.write(chunk)
        logf.flush()
        timeout_s = 0.0


def _terminate_process_tree(proc: subprocess.Popen[Any], logf) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGUSR1)
    except (OSError, ProcessLookupError):
        try:
            proc.send_signal(signal.SIGUSR1)
        except (OSError, ProcessLookupError):
            pass
    deadline = time.time() + 5.0
    while time.time() < deadline:
        _drain_output(proc, logf, timeout_s=0.2)
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        try:
            proc.terminate()
        except ProcessLookupError:
            return
    deadline = time.time() + 5.0
    while time.time() < deadline:
        _drain_output(proc, logf, timeout_s=0.2)
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        try:
            proc.kill()
        except ProcessLookupError:
            return


def run_case(case: Case, out_root: Path) -> dict[str, Any]:
    family_root = out_root / case.family
    family_root.mkdir(parents=True, exist_ok=True)
    dump_dir = family_root / f"{case.name}_dump"
    log_path = family_root / f"{case.name}.log"
    cleanup_stale_processes("/tmp/fp4_matrix_gpu3_")
    cleanup_stale_processes(str(dump_dir))
    if dump_dir.exists():
        shutil.rmtree(dump_dir)
    if log_path.exists():
        log_path.unlink()

    env = os.environ.copy()
    env.pop("FP4_ATTN_BACKEND", None)
    env.pop("FP4_FFN_BACKEND", None)
    env.update(BASE_ENV)
    env.update(case.env)
    use_torchrun = env.get("LBT_MATRIX_FORCE_TORCHRUN", "0") == "1"
    if use_torchrun:
        cmd = [
            "torchrun",
            "--nproc_per_node=1",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            str(TRAIN_PY),
            "--job.config_file",
            case.config,
            "--job.dump_folder",
            str(dump_dir),
            *case.overrides,
        ]
    else:
        # Single-rank matrix cases are more stable when launched directly and given
        # an explicit world-size-1 distributed contract, instead of going through
        # torchrun's elastic wrapper.
        env.setdefault("WORLD_SIZE", "1")
        env.setdefault("RANK", "0")
        env.setdefault("LOCAL_RANK", "0")
        env.setdefault("LOCAL_WORLD_SIZE", "1")
        env.setdefault("GROUP_RANK", "0")
        env.setdefault("ROLE_RANK", "0")
        env.setdefault("ROLE_WORLD_SIZE", "1")
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", str(29500 + (os.getpid() % 1000)))
        cmd = [
            "python3",
            "-u",
            str(TRAIN_PY),
            "--job.config_file",
            case.config,
            "--job.dump_folder",
            str(dump_dir),
            *case.overrides,
        ]
    started = time.time()
    timeout_hit = False
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            start_new_session=True,
        )
        deadline = started + case_timeout_s(case)
        while True:
            _drain_output(proc, logf, timeout_s=1.0)
            if proc.poll() is not None:
                break
            if time.time() >= deadline:
                timeout_hit = True
                _terminate_process_tree(proc, logf)
                break
        _drain_output(proc, logf, timeout_s=0.0)
    cleanup_stale_processes(str(dump_dir))
    duration_s = time.time() - started
    steps = parse_steps(log_path)
    completed = "Training completed" in strip_ansi(log_path.read_text(errors="replace"))
    first_nonfinite = next((row["step"] for row in steps if row["nonfinite"]), None)
    returncode = proc.returncode if proc.returncode is not None else -signal.SIGKILL
    result = {
        "name": case.name,
        "family": case.family,
        "config": case.config,
        "log_path": str(log_path),
        "dump_dir": str(dump_dir),
        "returncode": returncode,
        "completed": completed,
        "duration_s": round(duration_s, 3),
        "steps": steps,
        "final_step": steps[-1]["step"] if steps else 0,
        "first_nonfinite_step": first_nonfinite,
        "timed_out": timeout_hit,
        "supported": returncode == 0 or not case.allow_failure,
    }
    if case.allow_failure and returncode != 0:
        result["supported"] = False
    return result


def tail_median_mfu(rows: list[dict[str, Any]]) -> tuple[float | None, str]:
    if not rows:
        return None, "n/a"
    tail = [row["mfu"] for row in rows if row["step"] >= 11]
    if not tail:
        tail = [row["mfu"] for row in rows if row["step"] >= 2]
        if not tail:
            return None, f"step {rows[0]['step']}"
        window = f"steps {rows[1]['step']}-{rows[-1]['step']}" if len(rows) > 1 else "step 1"
    else:
        window = f"steps 11-{rows[-1]['step']}"
    return round(statistics.median(tail), 4), window


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2))


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_mfu(track: str, results: list[dict[str, Any]], out_root: Path) -> None:
    summary_rows: list[dict[str, Any]] = []
    for result in results:
        median_mfu, window = tail_median_mfu(result["steps"])
        summary_rows.append(
            {
                "case": result["name"],
                "final_step": result["final_step"],
                "completed": result["completed"],
                "returncode": result["returncode"],
                "timed_out": result.get("timed_out", False),
                "first_nonfinite_step": result["first_nonfinite_step"] or "",
                "tail_median_mfu": median_mfu if median_mfu is not None else "",
                "tail_window": window,
                "log_path": result["log_path"],
            }
        )

    track_root = out_root / track
    write_json(track_root / "summary.json", summary_rows)
    write_csv(
        track_root / "summary.csv",
        summary_rows,
        [
            "case",
            "final_step",
            "completed",
            "returncode",
            "timed_out",
            "first_nonfinite_step",
            "tail_median_mfu",
            "tail_window",
            "log_path",
        ],
    )
    lines = [
        f"# {track} MFU Summary",
        "",
        "| case | final step | completed | timed out | first non-finite | tail median MFU | tail window |",
        "| --- | ---: | :---: | :---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['case']} | {row['final_step']} | "
            f"{'yes' if row['completed'] else 'no'} | "
            f"{'yes' if row['timed_out'] else 'no'} | "
            f"{row['first_nonfinite_step'] or '-'} | "
            f"{row['tail_median_mfu'] or '-'} | {row['tail_window']} |"
        )
    (track_root / "summary.md").write_text("\n".join(lines) + "\n")


def convergence_divergence_step(
    te_steps: list[dict[str, Any]], backend_steps: list[dict[str, Any]]
) -> tuple[int | None, list[dict[str, Any]]]:
    te_by_step = {row["step"]: row for row in te_steps}
    aligned: list[dict[str, Any]] = []
    for row in backend_steps:
        te_row = te_by_step.get(row["step"])
        if te_row is None:
            continue
        loss_delta = None
        if te_row["loss"] is not None and row["loss"] is not None:
            loss_delta = row["loss"] - te_row["loss"]
        aligned.append(
            {
                "step": row["step"],
                "backend_loss": row["loss"],
                "te_loss": te_row["loss"],
                "loss_delta_vs_te": loss_delta,
                "backend_grad_norm": row["grad_norm"],
                "te_grad_norm": te_row["grad_norm"],
                "backend_mfu": row["mfu"],
                "te_mfu": te_row["mfu"],
                "nonfinite": row["nonfinite"],
            }
        )
    first_nonfinite = next((row["step"] for row in aligned if row["nonfinite"]), None)
    if first_nonfinite is not None:
        return first_nonfinite, aligned
    for idx in range(len(aligned) - 4):
        window = aligned[idx : idx + 5]
        if all(
            row["loss_delta_vs_te"] is not None and abs(row["loss_delta_vs_te"]) > 0.5
            for row in window
        ):
            return window[0]["step"], aligned
    return None, aligned


def summarize_convergence(track: str, results: list[dict[str, Any]], out_root: Path) -> None:
    track_root = out_root / track
    te_result = next((result for result in results if result["name"] == "te_fp4_legacy"), None)
    per_case_summary: list[dict[str, Any]] = []
    aligned_payload: dict[str, Any] = {}
    first_divergent_case: str | None = None
    first_divergent_step: int | None = None

    for result in results:
        if te_result is not None:
            divergence_step, aligned = convergence_divergence_step(te_result["steps"], result["steps"])
        else:
            divergence_step, aligned = (None, [])
        if divergence_step is None and result["name"] != "te_fp4_legacy" and not result["completed"]:
            divergence_step = max(1, result["final_step"] + 1)
        aligned_payload[result["name"]] = aligned
        if result["name"] == "te_fp4_legacy" or te_result is None:
            max_abs_loss_delta = 0.0
        else:
            deltas = [abs(row["loss_delta_vs_te"]) for row in aligned if row["loss_delta_vs_te"] is not None]
            max_abs_loss_delta = max(deltas) if deltas else None
        stay_in_family = result["completed"] and divergence_step is None
        summary = {
            "case": result["name"],
            "final_step": result["final_step"],
            "completed": result["completed"],
            "first_nonfinite_step": result["first_nonfinite_step"] or "",
            "timed_out": result.get("timed_out", False),
            "first_divergence_step": divergence_step or "",
            "max_abs_loss_delta_vs_te": round(max_abs_loss_delta, 6) if max_abs_loss_delta is not None else "",
            "stay_in_family": stay_in_family,
            "log_path": result["log_path"],
        }
        per_case_summary.append(summary)
        if result["name"] != "te_fp4_legacy" and divergence_step is not None:
            if first_divergent_step is None or divergence_step < first_divergent_step:
                first_divergent_case = result["name"]
                first_divergent_step = divergence_step

    payload = {
        "summary": per_case_summary,
        "aligned_metrics": aligned_payload,
        "first_divergent_backend": first_divergent_case,
        "first_divergent_step": first_divergent_step,
    }
    write_json(track_root / "summary.json", payload)
    write_csv(
        track_root / "summary.csv",
        per_case_summary,
        [
            "case",
            "final_step",
            "completed",
            "first_nonfinite_step",
            "timed_out",
            "first_divergence_step",
            "max_abs_loss_delta_vs_te",
            "stay_in_family",
            "log_path",
        ],
    )
    lines = [
        f"# {track} Summary",
        "",
        f"- first divergent backend: {first_divergent_case or 'none'}",
        f"- first divergent step: {first_divergent_step or 'none'}",
        "",
        "| case | final step | completed | first non-finite | timed out | first divergence | max abs loss delta vs TE | stays in-family |",
        "| --- | ---: | :---: | ---: | :---: | ---: | ---: | :---: |",
    ]
    for row in per_case_summary:
        lines.append(
            f"| {row['case']} | {row['final_step']} | "
            f"{'yes' if row['completed'] else 'no'} | "
            f"{row['first_nonfinite_step'] or '-'} | "
            f"{'yes' if row['timed_out'] else 'no'} | "
            f"{row['first_divergence_step'] or '-'} | "
            f"{row['max_abs_loss_delta_vs_te'] or '-'} | "
            f"{'yes' if row['stay_in_family'] else 'no'} |"
        )
    (track_root / "summary.md").write_text("\n".join(lines) + "\n")


def run_track(track: str, cases: list[Case], out_root: Path) -> list[dict[str, Any]]:
    case_filter = os.environ.get("FP4_MATRIX_CASE_FILTER", "").strip()
    if case_filter:
        wanted = {name.strip() for name in case_filter.split(",") if name.strip()}
        cases = [case for case in cases if case.name in wanted]
        if not cases:
            raise SystemExit(
                f"FP4_MATRIX_CASE_FILTER matched no cases for track {track!r}: {case_filter}"
            )
    results = []
    for case in cases:
        print(f"=== RUN {track}:{case.name} ===", flush=True)
        result = run_case(case, out_root)
        print(
            json.dumps(
                {
                    "case": case.name,
                    "returncode": result["returncode"],
                    "final_step": result["final_step"],
                    "completed": result["completed"],
                    "first_nonfinite_step": result["first_nonfinite_step"],
                    "timed_out": result.get("timed_out", False),
                    "log_path": result["log_path"],
                }
            ),
            flush=True,
        )
        results.append(result)
    if track.startswith("mfu_"):
        summarize_mfu(track, results, out_root)
    else:
        summarize_convergence(track, results, out_root)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["mfu_1b", "mfu_legacy", "convergence_legacy"],
        choices=list(TRACKS.keys()),
    )
    parser.add_argument(
        "--out-root",
        default=f"/tmp/fp4_matrix_gpu3_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "repo_root": str(REPO_ROOT),
        "out_root": str(out_root),
        "tracks": args.tracks,
        "gpu": "3",
        "base_env": BASE_ENV,
    }
    write_json(out_root / "manifest.json", manifest)

    all_results: dict[str, Any] = {}
    for track in args.tracks:
        all_results[track] = run_track(track, TRACKS[track], out_root)
    write_json(out_root / "all_results.json", all_results)
    print(f"Results written to {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
