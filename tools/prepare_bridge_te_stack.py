#!/usr/bin/env python3
"""Prepare a matched Transformer Engine stack for Bridge FP4 runs.

The local system CUDA runtime can be older than the Transformer Engine wheel
expects. This helper stages TE plus the matching cuBLAS runtime in a target
directory and prints shell exports that must be applied before launching Python.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


DEFAULT_TE_VERSION = "2.14.1"
DEFAULT_CUBLAS_VERSION = "13.5.1.27"
DEFAULT_STAGE_ROOT = Path(os.environ.get("LBT_BRIDGE_TE_ROOT", "/tmp/lbt_bridge_te"))


def _tag(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


def _pip(python: Path) -> list[str]:
    return [str(python), "-m", "pip"]


def _one_wheel(wheel_dir: Path, prefix: str, version: str) -> Path:
    matches = sorted(wheel_dir.glob(f"{prefix}-{version}*.whl"))
    if not matches:
        raise SystemExit(f"missing staged wheel matching {prefix}-{version}*.whl in {wheel_dir}")
    return matches[-1]


def _stage_env(stage_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(stage_dir)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    ld_parts = [
        str(stage_dir / "nvidia" / "cu13" / "lib"),
        str(stage_dir / "transformer_engine" / "wheel_lib"),
    ]
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)
    env["LBT_BRIDGE_TE_STAGE"] = str(stage_dir)
    return env


def _verify_stage(python: Path, stage_dir: Path, te_version: str) -> bool:
    env = _stage_env(stage_dir)
    code = f"""
import transformer_engine
import transformer_engine_torch
from transformer_engine.pytorch.tensor import utils as te_tensor_utils
version = getattr(transformer_engine, "__version__", "?")
if version != {te_version!r}:
    raise SystemExit(f"expected TE {te_version}, got {{version}}")
if not hasattr(te_tensor_utils, "quantize_master_weights"):
    raise SystemExit("TE tensor utils does not expose quantize_master_weights")
print("TE", version, "ok", transformer_engine.__file__)
print("TE torch", getattr(transformer_engine_torch, "__file__", transformer_engine_torch))
"""
    result = subprocess.run(
        [str(python), "-c", code],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode == 0:
        print(result.stdout.strip(), file=sys.stderr)
        return True
    print(result.stdout, file=sys.stderr)
    return False


def prepare_stack(args: argparse.Namespace) -> Path:
    stage_dir = args.stage_dir or (
        DEFAULT_STAGE_ROOT / f"te_{_tag(args.te_version)}_cublas_{_tag(args.cublas_version)}"
    )
    wheel_dir = args.wheel_dir or (
        DEFAULT_STAGE_ROOT / "wheels" / f"te_{_tag(args.te_version)}_cublas_{_tag(args.cublas_version)}"
    )
    python = args.python.expanduser()

    if stage_dir.exists() and not args.force and _verify_stage(python, stage_dir, args.te_version):
        return stage_dir

    wheel_dir.mkdir(parents=True, exist_ok=True)
    _run(
        _pip(python)
        + [
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "-d",
            str(wheel_dir),
            f"transformer-engine=={args.te_version}",
            f"transformer-engine-cu13=={args.te_version}",
            f"nvidia-cublas=={args.cublas_version}",
        ]
    )

    if not list(wheel_dir.glob(f"transformer_engine_torch-{args.te_version}*.whl")):
        build_env = os.environ.copy()
        build_env["NVTE_PYTORCH_FORCE_BUILD"] = "TRUE"
        build_env["MAX_JOBS"] = str(args.max_jobs)
        _run(
            _pip(python)
            + [
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "-w",
                str(wheel_dir),
                f"transformer-engine-torch=={args.te_version}",
            ],
            env=build_env,
        )

    tmp_stage = stage_dir.with_name(f"{stage_dir.name}.tmp.{os.getpid()}")
    if tmp_stage.exists():
        shutil.rmtree(tmp_stage)
    tmp_stage.mkdir(parents=True)

    wheels = [
        _one_wheel(wheel_dir, "nvidia_cublas", args.cublas_version),
        _one_wheel(wheel_dir, "transformer_engine", args.te_version),
        _one_wheel(wheel_dir, "transformer_engine_cu13", args.te_version),
        _one_wheel(wheel_dir, "transformer_engine_torch", args.te_version),
    ]
    _run(_pip(python) + ["install", "--no-deps", "--target", str(tmp_stage), *map(str, wheels)])

    if not _verify_stage(python, tmp_stage, args.te_version):
        shutil.rmtree(tmp_stage, ignore_errors=True)
        raise SystemExit(f"staged TE {args.te_version} failed import verification")

    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    tmp_stage.rename(stage_dir)
    return stage_dir


def _print_env(stage_dir: Path, mode: str) -> None:
    env = _stage_env(stage_dir)
    keys = ("LBT_BRIDGE_TE_STAGE", "PYTHONPATH", "LD_LIBRARY_PATH")
    if mode == "json":
        print(json.dumps({key: env[key] for key in keys}, indent=2))
    elif mode == "sh":
        for key in keys:
            print(f"export {key}={shlex.quote(env[key])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--te-version", default=DEFAULT_TE_VERSION)
    parser.add_argument("--cublas-version", default=DEFAULT_CUBLAS_VERSION)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage-dir", type=Path, default=None)
    parser.add_argument("--wheel-dir", type=Path, default=None)
    parser.add_argument("--max-jobs", type=int, default=int(os.environ.get("MAX_JOBS", "12")))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--print-env", choices=("none", "sh", "json"), default="sh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stage_dir = prepare_stack(args)
    _print_env(stage_dir, args.print_env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
