#!/usr/bin/env python3
"""Plan or execute one hash-bound, local-input paper recipe.

This launcher is deliberately independent of Kubernetes, Slurm, object stores,
and tracking services. A scheduler only needs to provide standard torchrun
topology values. The child process receives a scrubbed numerical-route
environment assembled from versioned files in ``configs/paper/env``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import release_capsule


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_PATH = ROOT / "configs" / "paper" / "route_execution.json"
ROUTE_ENV_RE = re.compile(
    r"^(?:MXFP4_|NVFP4_|USE_TK_|USE_FP4_|FP4_(?:ATTN|FFN|KEEP)|"
    r"NVTE_(?:NVFP4|CUSTOM_QUANT)|LBT_(?:LOCALCTA|FP4_MIXED|MXFP4|NEMOTRON|"
    r"REQUIRE_V5|REQUIRE_FRESH)|TORCHTITAN_FSDP_)"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _route(route_id: str) -> dict[str, Any]:
    document = _load_json(EXECUTION_PATH)
    matches = [route for route in document["routes"] if route["id"] == route_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate route: {route_id}")
    return matches[0]


def _route_environment(path: Path) -> dict[str, str]:
    """Evaluate a trusted tracked env preset in an otherwise empty shell."""
    process = subprocess.run(
        (
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'set -euo pipefail; source "$1"; env -0',
            "paper-route-env",
            str(path),
        ),
        check=True,
        stdout=subprocess.PIPE,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )
    result: dict[str, str] = {}
    for item in process.stdout.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode()] = value.decode()
    result.pop("PATH", None)
    result.pop("PWD", None)
    result.pop("SHLVL", None)
    result.pop("_", None)
    return result


def _canonical_environment_sha256(values: dict[str, str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_plan(args: argparse.Namespace) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    inputs_path = Path(args.inputs).expanduser().resolve()
    inputs = _load_json(inputs_path)
    route = _route(args.route)
    findings = release_capsule.validate_inputs(inputs_path)
    findings.extend(release_capsule.validate_route(args.route, args.resume))
    if findings:
        rendered = "; ".join(f"{item.rule}:{item.subject}" for item in findings)
        raise ValueError(f"release input/route contract blocked: {rendered}")

    expected_world = int(route["default_world_size"])
    world = int(args.nnodes) * int(args.nproc_per_node)
    if world != expected_world:
        raise ValueError(
            f"route {args.route} requires world size {expected_world}; "
            f"got {args.nnodes}x{args.nproc_per_node}={world}. Create a new route "
            "identity for a topology change."
        )
    if not 0 <= int(args.node_rank) < int(args.nnodes):
        raise ValueError("node rank must be in [0, nnodes)")

    checkpoint = inputs.get("checkpoint")
    if args.resume:
        if checkpoint is None:
            raise ValueError("resume requires an externally supplied checkpoint binding")
        if checkpoint["route_id"] != args.route:
            raise ValueError("checkpoint route_id does not match the requested route")
        if int(checkpoint["step"]) >= 38147:
            raise ValueError("checkpoint is already at or beyond the paper terminal step")
    elif checkpoint is not None:
        raise ValueError("fresh launch rejects a checkpoint binding; use --resume")

    env_path = (ROOT / route["environment"]).resolve()
    if not env_path.is_relative_to(ROOT) or not env_path.is_file():
        raise ValueError("route environment escaped or is missing")
    route_env = _route_environment(env_path)
    child_env = {key: value for key, value in os.environ.items() if not ROUTE_ENV_RE.match(key)}
    child_env.update(route_env)
    child_env["LBT_PAPER_ROUTE_ID"] = args.route
    child_env["LBT_PAPER_ROUTE_ENV_SHA256"] = _canonical_environment_sha256(route_env)
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    child_env["FP4_MATMUL_ROOT"] = str(runtime_root)
    child_env["FP4_MATMUL_GEMM_ROOT"] = str(runtime_root)
    child_env["FP4_MXFP4_ROOT"] = str(runtime_root)
    child_env["LBT_REQUIRE_FRESH_START"] = "0" if args.resume else "1"

    model_assets = Path(inputs["tokenizer"]["directory"]).expanduser().resolve()
    dataset_manifest = Path(inputs["dataset"]["manifest"]["path"]).expanduser().resolve()
    output = Path(inputs["output"]["directory"]).expanduser().resolve()
    child_env["TORCH_EXTENSIONS_DIR"] = str(output / ".torch_extensions")

    config = (ROOT / _load_json(EXECUTION_PATH)["base_config"]).resolve()
    dataset_kwargs = json.dumps(
        {
            "manifest": str(dataset_manifest),
            "repeat": False,
            "require_full_run": True,
            "num_workers": 1,
            "prefetch_factor": 4,
            "pin_memory": False,
        },
        separators=(",", ":"),
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={args.nnodes}",
        f"--nproc-per-node={args.nproc_per_node}",
        f"--node-rank={args.node_rank}",
        f"--master-addr={args.master_addr}",
        f"--master-port={args.master_port}",
        str(ROOT / "train.py"),
        f"--job.config-file={config}",
        f"--job.dump-folder={output}",
        f"--model.hf-assets-path={model_assets}",
        "--training.dataset=packed-bin",
        f"--training.dataset-path={dataset_manifest}",
        f"--training.load-dataset-kwargs={dataset_kwargs}",
        f"--training.local-batch-size={route['local_batch_size']}",
        "--training.global-batch-size=512",
        f"--model.converters={','.join(route['converters'])}",
        "--job.experimental-modules=low_bits_training.reproduction.paper_contracts",
        f"--wandb.mode={args.wandb_mode}",
    ]
    for key, value in sorted(route.get("overrides", {}).items()):
        command.append(f"--{key.replace('_', '-')}={_cli_value(value)}")
    if args.resume:
        command.extend(
            (
                f"--checkpoint.initial-load-path={Path(checkpoint['directory']).expanduser().resolve()}",
                "--checkpoint.no-initial-load-model-only",
            )
        )

    summary = {
        "schema_version": 1,
        "route_id": args.route,
        "mode": "resume" if args.resume else "fresh",
        "world_size": world,
        "local_batch_size": route["local_batch_size"],
        "gradient_accumulation": route["gradient_accumulation"],
        "global_batch_size": 512,
        "route_environment_sha256": child_env["LBT_PAPER_ROUTE_ENV_SHA256"],
        "execution_status": route["execution_status"],
        "paper_evidence_status": route["paper_evidence_status"],
        "external_values_redacted": True,
    }
    return command, child_env, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--runtime-root", default=str(ROOT / "fp4_runtime"))
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default="disabled")
    args = parser.parse_args()

    try:
        command, child_env, summary = build_plan(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if not args.execute:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    output = Path(_load_json(Path(args.inputs))["output"]["directory"]).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    os.execvpe(command[0], command, child_env)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
