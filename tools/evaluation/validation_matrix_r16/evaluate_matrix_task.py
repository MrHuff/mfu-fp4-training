#!/usr/bin/env python3
"""Evaluate one local r16 matrix cell with training-lineage scaled RoPE."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch


MATRIX_SCHEMA = "mfu_llama_fixed_independent_validation_matrix_r16_v1"
MATRIX_ID = "fixed-independent-r16-exact-checkpoints-20260902"
RESULT_SCHEMA = "mfu_llama_fixed_independent_scaledrope_validation_result_r19_v1"
TRAINING_TORCHTITAN_COMMIT = "20b3de7585696c327bd5aa9f9627f0300abdbf9d"
TRAINING_ROPE = {
    "rope_theta": 500000.0,
    "scaling_factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
TASK_FIELDS = (
    "semantic_route_key",
    "route_label",
    "training_recipe",
    "lineage_note",
    "converter_route",
    "checkpoint_key",
    "step",
    "expected_ntokens_seen",
    "metadata_sha256",
    "shard_count",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def task_identity(task: dict) -> str:
    return sha256_bytes(canonical_bytes({field: task[field] for field in TASK_FIELDS}))


def validate_matrix(matrix: dict) -> None:
    unsigned = dict(matrix)
    observed_seal = unsigned.pop("matrix_seal_sha256", None)
    if observed_seal != sha256_bytes(canonical_bytes(unsigned)):
        raise RuntimeError("r16 matrix seal drift")
    if (
        matrix.get("schema") != MATRIX_SCHEMA
        or matrix.get("matrix_id") != MATRIX_ID
        or matrix.get("claim") != "fixed-independent-not-proven-held-out"
        or matrix.get("scope", {}).get("task_count") != 44
        or len(matrix.get("tasks", [])) != 44
    ):
        raise RuntimeError("r16 matrix identity drift")
    identities = set()
    for index, task in enumerate(matrix["tasks"]):
        if task.get("index") != index or task.get("task_identity_sha256") != task_identity(task):
            raise RuntimeError(f"r16 task identity drift: {index}")
        if task["task_identity_sha256"] in identities:
            raise RuntimeError("duplicate r16 task identity")
        identities.add(task["task_identity_sha256"])
        if task.get("shard_count") not in {32, 64}:
            raise RuntimeError(f"r16 task shard geometry drift: {index}")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_lineage_torchtitan_types(torchtitan_root: Path, parity_modules):
    """Replace the native builder's unscaled RoPE with the producer semantics."""

    types = parity_modules.load_pinned_torchtitan(torchtitan_root)
    BaseArgs, Transformer, Adapter, Attention, commit = types
    if commit != TRAINING_TORCHTITAN_COMMIT:
        raise RuntimeError("training TorchTitan gitlink drift")
    from torchtitan.models.llama3.model.args import RoPEScalingArgs

    class TrainingLineageModelArgs(BaseArgs):
        def __init__(self, *args, **kwargs):
            if kwargs.get("rope_scaling_args") is not None:
                raise RuntimeError("native builder unexpectedly supplied scaled RoPE")
            kwargs["rope_scaling_args"] = RoPEScalingArgs(
                scaling_factor=TRAINING_ROPE["scaling_factor"],
                low_freq_factor=TRAINING_ROPE["low_freq_factor"],
                high_freq_factor=TRAINING_ROPE["high_freq_factor"],
                original_max_position_embeddings=TRAINING_ROPE[
                    "original_max_position_embeddings"
                ],
            )
            super().__init__(*args, **kwargs)

    return TrainingLineageModelArgs, Transformer, Adapter, Attention, commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--tokenizer-assets", type=Path, required=True)
    parser.add_argument("--torchtitan-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.work_root.exists() or args.output.exists():
        raise FileExistsError("work and output paths must be new")

    matrix_payload = args.matrix.read_bytes()
    matrix = json.loads(matrix_payload)
    validate_matrix(matrix)
    if not 0 <= args.task_index < len(matrix["tasks"]):
        raise RuntimeError("r16 task index outside sealed matrix")
    task = matrix["tasks"][args.task_index]
    if task["checkpoint_key"] != args.checkpoint.name:
        raise RuntimeError("checkpoint directory does not match the public matrix key")

    base_path = (
        args.project_root
        / "tools/evaluation/fixed_independent_r15/evaluate_fixed_independent.py"
    )
    base = load_module(base_path, "mfu_fixed_independent_r15")
    validation_payload = args.validation_manifest.read_bytes()
    if sha256_bytes(validation_payload) != matrix["validation_stream"]["manifest_file_sha256"]:
        raise RuntimeError("validation manifest file hash drift")
    validation_manifest = json.loads(validation_payload)
    base.validate_fixed_manifest(validation_manifest)
    validation_data = base.load_validation(args.validation_manifest, validation_manifest)
    metadata_path = args.checkpoint / ".metadata"
    if sha256_file(metadata_path) != task["metadata_sha256"]:
        raise RuntimeError("checkpoint metadata content drift")

    args.work_root.mkdir(parents=True)
    args.output.mkdir(parents=True)
    os.environ["LBT_LIGHT_IMPORT"] = "1"
    sys.path.insert(0, str(args.project_root))
    from low_bits_training.analysis import llama_validation_loss as validation_modules
    from scripts.evaluation import validate_llama8b_conversion_parity as parity_modules

    converted = args.work_root / "converted"
    parity_path = args.output / "canonical-parity.json"
    source_id = "public-local-checkpoint"
    source_digest = sha256_bytes(source_id.encode())
    subprocess.run(
        [
            sys.executable,
            str(args.project_root / "scripts/evaluation/convert_llama8b_dcp_to_hf.py"),
            "--checkpoint", str(args.checkpoint),
            "--output", str(converted),
            "--tokenizer-assets", str(args.tokenizer_assets),
            "--source-job-id", source_id,
            "--source-uri-sha256", source_digest,
            "--route", task["converter_route"],
            "--source-dtype", "float32",
            "--output-dtype", "bfloat16",
            "--expected-shards", str(task["shard_count"]),
            "--expected-ntokens-seen", str(task["expected_ntokens_seen"]),
        ],
        cwd=args.project_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(args.project_root / "scripts/evaluation/validate_llama8b_canonical_parity.py"),
            "--checkpoint", str(args.checkpoint),
            "--converted", str(converted),
            "--receipt", str(parity_path),
            "--torchtitan-root", str(args.torchtitan_root),
            "--expected-route", task["converter_route"],
            "--expected-step", str(task["step"]),
            "--expected-ntokens-seen", str(task["expected_ntokens_seen"]),
            "--expected-source-job-id", source_id,
            "--expected-source-uri-sha256", source_digest,
            "--expected-metadata-sha256", task["metadata_sha256"],
            "--expected-shards", str(task["shard_count"]),
            "--device", args.device,
        ],
        cwd=args.project_root,
        check=True,
    )
    parity = json.loads(parity_path.read_bytes())
    if not parity.get("passed"):
        raise RuntimeError("canonical conversion/state parity did not pass")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    index = json.loads((converted / "model.safetensors.index.json").read_text())
    types = training_lineage_torchtitan_types(args.torchtitan_root, parity_modules)
    native, native_load = parity_modules.build_native_reference(
        args.checkpoint,
        converted,
        task["converter_route"],
        index["weight_map"],
        device,
        8192,
        types,
    )
    metric = base.evaluate_native_full(native, validation_data, device, validation_modules)
    del native
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "schema": RESULT_SCHEMA,
        "matrix_file_sha256": sha256_bytes(matrix_payload),
        "matrix_seal_sha256": matrix["matrix_seal_sha256"],
        "task_index": task["index"],
        "task_identity_sha256": task["task_identity_sha256"],
        "semantic_route_key": task["semantic_route_key"],
        "route_label": task["route_label"],
        "step": task["step"],
        "validation_stream_id": matrix["validation_stream"]["stream_id"],
        "contract": {
            **matrix["evaluation_contract"],
            "training_rope": TRAINING_ROPE,
            "torchtitan_commit": TRAINING_TORCHTITAN_COMMIT,
        },
        "canonical_conversion_parity_sha256": sha256_file(parity_path),
        "torchtitan_native_8192": metric,
        "native_load": native_load,
    }
    result_payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    (args.output / "validation-result.json").write_bytes(result_payload)
    (args.output / "SHA256SUMS").write_text(
        f"{sha256_bytes(result_payload)}  validation-result.json\n"
        f"{sha256_file(parity_path)}  canonical-parity.json\n"
    )
    print(
        "MFU_FIXED_INDEPENDENT_R16_TASK_PASS "
        f"task={task['index']} route={task['semantic_route_key']} "
        f"step={task['step']} nll={metric['nll']:.12f}"
    )


if __name__ == "__main__":
    main()
