"""Run a rendered MXFP4 H32 plan without cluster-specific dependencies."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from .launcher import LaunchGeometry, torchrun_command
from .route_contract import (
    file_sha256,
    load_spec,
    scrub_scientific_environment,
    validate_environment,
)


HERE = Path(__file__).resolve().parent


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported effective-plan schema")
    expected = plan.get("plan_sha256")
    unsealed = deepcopy(plan)
    unsealed.pop("plan_sha256", None)
    actual = sha256(
        json.dumps(unsealed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("effective-plan SHA-256 mismatch")
    return plan


def validate_plan(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Path]]:
    spec = plan["benchmark_spec"]
    canonical_spec = load_spec(HERE / "benchmark.json")
    if spec != canonical_spec:
        raise ValueError("embedded benchmark spec differs from this checkout")
    source = plan["source"]
    if source["benchmark_spec_sha256"] != file_sha256(HERE / "benchmark.json"):
        raise ValueError("benchmark spec hash differs from this checkout")
    if source["route_contract_sha256"] != file_sha256(HERE / "route_contract.py"):
        raise ValueError("route contract hash differs from this checkout")
    bindings = {key: Path(value).resolve() for key, value in plan["bindings"].items()}
    for name in ("repository", "runtime", "model_assets", "dataset", "output_dir"):
        if "://" in str(plan["bindings"][name]):
            raise ValueError(f"remote URI is forbidden for {name}")
    for name in ("repository", "runtime", "model_assets", "dataset"):
        if not bindings[name].is_dir():
            raise ValueError(f"bound {name} directory disappeared")
    expected_heads = {
        "repository": source["repository_commit"],
        "runtime": source["runtime_commit"],
        "torchtitan": source["torchtitan_commit"],
    }
    actual_heads = {
        "repository": _git_head(bindings["repository"]),
        "runtime": _git_head(bindings["runtime"]),
        "torchtitan": _git_head(bindings["repository"] / "torchtitan_submodule"),
    }
    if actual_heads != expected_heads:
        raise ValueError(f"source pins changed after render: {actual_heads}")
    return spec, bindings


def train_arguments(spec: dict[str, Any], bindings: dict[str, Path], node_rank: int) -> list[str]:
    model = spec["model"]
    batch = spec["batch"]
    measurement = spec["measurement"]
    training = spec["training"]
    node_output = bindings["output_dir"] / f"node-{node_rank}"
    return [
        "--job.config_file",
        str(bindings["repository"] / model["config"]),
        f"--model.hf_assets_path={bindings['model_assets']}",
        "--training.dataset=packed-bin",
        f"--training.dataset_path={bindings['dataset']}",
        "--training.dataset_node_distribution=shard",
        f"--training.steps={training['scheduler_horizon']}",
        f"--job.steps={measurement['updates']}",
        f"--training.global_batch_size={batch['global_sequences']}",
        f"--training.local_batch_size={batch['local_sequences']}",
        "--training.mixed_precision_param=bfloat16",
        "--training.no-enable-cce",
        "--training.no-compile",
        "--fp4-cce.no-enabled",
        "--compile.enable",
        "--compile.components=loss",
        "--compile.backend=inductor",
        "--model.converters=bfloat16,mxfp4_tk,mxfp4_h32_benchmark_contract,fp32_master",
        "--job.experimental-modules=mxfp4_h32_benchmark_contract",
        f"--optimizer.lr={training['learning_rate']}",
        f"--lr-scheduler.warmup-steps={training['warmup_updates']}",
        "--lr-scheduler.decay-type=cosine",
        "--lr-scheduler.decay-ratio=0.3333333333333333",
        f"--lr-scheduler.min-lr-factor={training['minimum_learning_rate_factor']}",
        "--activation_checkpoint.mode=selective",
        f"--activation_checkpoint.selective_ac_option={training['activation_checkpoint_option']}",
        "--parallelism.fsdp_reshard_after_forward=default",
        "--parallelism.data_parallel_replicate_degree=1",
        "--parallelism.data_parallel_shard_degree=-1",
        f"--model.flavor={model['flavor']}",
        f"--debug.seed={model['seed']}",
        "--checkpoint.no-enable",
        "--metrics.log_freq=1",
        "--metrics.distributed_mode=all",
        f"--job.dump-folder={node_output / 'dump'}",
        "--wandb.mode=disabled",
    ]


def run(
    plan_path: Path,
    *,
    node_rank: int,
    master_addr: str,
    master_port: int,
    dry_run: bool,
    memory_mode: str,
) -> int:
    plan = load_plan(plan_path)
    spec, bindings = validate_plan(plan)
    topology = spec["topology"]
    environment = scrub_scientific_environment(os.environ)
    environment.update(spec["environment"])
    environment.update(
        {
            "FP4_MATMUL_ROOT": str(bindings["runtime"]),
            "FP4_MATMUL_GEMM_ROOT": str(bindings["runtime"]),
            "FP4_MXFP4_ROOT": str(bindings["runtime"]),
            "TORCH_EXTENSIONS_DIR": str(bindings["output_dir"] / "torch-extensions"),
            "PYTHONPATH": os.pathsep.join(
                (
                    str(bindings["runtime"]),
                    str(bindings["repository"] / "torchtitan_submodule"),
                    str(bindings["repository"]),
                    environment.get("PYTHONPATH", ""),
                )
            ).rstrip(os.pathsep),
        }
    )
    validate_environment(spec, environment)
    geometry = LaunchGeometry(
        nodes=topology["nodes"],
        processes_per_node=topology["processes_per_node"],
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
        rendezvous_id=f"mxfp4-h32-{plan['plan_sha256'][:12]}",
    )
    command = torchrun_command(
        geometry,
        bindings["repository"] / "train.py",
        train_arguments(spec, bindings, node_rank),
    )
    if dry_run:
        print(shlex.join(command))
        return 0

    node_output = bindings["output_dir"] / f"node-{node_rank}"
    node_output.mkdir(parents=False, exist_ok=False)
    orientation = [
        sys.executable,
        "-m",
        "benchmarks.mxfp4_h32.orientation_gate",
        "--device",
        "cuda:0",
    ]
    subprocess.run(
        orientation,
        cwd=bindings["repository"],
        env=environment,
        check=True,
    )
    log_path = node_output / "train.log"
    stop_file = node_output / "memory.stop"
    violation_file = node_output / "memory.violation"
    telemetry_file = node_output / "memory.tsv"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=bindings["repository"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        guard: subprocess.Popen[str] | None = None
        if memory_mode != "off":
            guard = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.mxfp4_h32.memory_guard",
                    "--pgid",
                    str(process.pid),
                    "--expected-local-gpus",
                    str(topology["processes_per_node"]),
                    "--cap-mib",
                    str(spec["runtime"]["recommended_memory_cap_mib"]),
                    "--mode",
                    memory_mode,
                    "--stop-file",
                    str(stop_file),
                    "--violation-file",
                    str(violation_file),
                    "--telemetry-file",
                    str(telemetry_file),
                ],
                cwd=bindings["repository"],
                env=environment,
                text=True,
            )
        status = process.wait()
        stop_file.touch(exist_ok=True)
        guard_status = guard.wait() if guard is not None else 0
    if status != 0 or guard_status != 0 or violation_file.exists():
        raise RuntimeError(
            f"benchmark failed: train={status} guard={guard_status} "
            f"violation={violation_file.exists()}"
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--memory-mode", choices=("off", "monitor", "enforce"), default="monitor"
    )
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.plan,
            node_rank=args.node_rank,
            master_addr=args.master_addr,
            master_port=args.master_port,
            dry_run=args.dry_run,
            memory_mode=args.memory_mode,
        )
    )


if __name__ == "__main__":
    main()
