#!/usr/bin/env python3
"""Build the public, storage-neutral 44-cell r16 validation matrix."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SCHEMA = "mfu_llama_fixed_independent_validation_matrix_r16_v1"
MATRIX_ID = "fixed-independent-r16-exact-checkpoints-20260902"
STREAM_ID = "llama31-fixed-independent-82dclm18olmo-r15-20260902"
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
EXPECTED_COUNTS = {
    "bf16": 5,
    "te_native": 5,
    "te_fol4": 4,
    "pure_v5": 5,
    "mxfp4": 5,
    "localcta": 5,
    "localcta_mxfp4_hybrid": 1,
    "localcta_h16": 4,
    "mxfp4_h32": 5,
    "operand_h16": 1,
    "operand_h32": 4,
}


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


def validate_matrix(tasks: list[dict]) -> None:
    if len(tasks) != 44:
        raise RuntimeError(f"expected 44 tasks, observed {len(tasks)}")
    if Counter(task["semantic_route_key"] for task in tasks) != Counter(EXPECTED_COUNTS):
        raise RuntimeError("route cell counts drift")
    cells = [(task["semantic_route_key"], task["step"]) for task in tasks]
    if len(set(cells)) != len(cells):
        raise RuntimeError("duplicate route/step cell")
    identities = [task["task_identity_sha256"] for task in tasks]
    if len(set(identities)) != len(identities):
        raise RuntimeError("duplicate task identity")
    for index, task in enumerate(tasks):
        if task["index"] != index or task["task_identity_sha256"] != task_identity(task):
            raise RuntimeError(f"task identity drift: {index}")
        if set(task) != {"index", "task_identity_sha256", *TASK_FIELDS}:
            raise RuntimeError(f"task fields drift: {index}")
        if task["shard_count"] not in {32, 64}:
            raise RuntimeError(f"unsupported DCP shard count: {index}")
        for field in ("checkpoint_key", "metadata_sha256"):
            value = task[field]
            if not isinstance(value, str) or not value:
                raise RuntimeError(f"invalid {field}: {index}")
        if len(task["metadata_sha256"]) != 64:
            raise RuntimeError(f"invalid metadata digest: {index}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    inventory = json.loads(args.inventory.read_bytes())
    if inventory.get("schema") != "mfu_public_checkpoint_inventory_v1":
        raise RuntimeError("checkpoint inventory schema drift")
    tasks = []
    for source in inventory.get("records", []):
        task = {field: source[field] for field in TASK_FIELDS}
        task["task_identity_sha256"] = task_identity(task)
        tasks.append(task)
    tasks.sort(key=lambda item: (item["semantic_route_key"], item["step"]))
    for index, task in enumerate(tasks):
        task["index"] = index
    validate_matrix(tasks)

    validation = json.loads(args.validation_manifest.read_bytes())
    if (
        validation.get("stream_id") != STREAM_ID
        or validation.get("claim") != "fixed-independent-not-proven-held-out"
    ):
        raise RuntimeError("validation manifest identity drift")
    unsigned = dict(validation)
    validation_seal = unsigned.pop("manifest_sha256", None)
    if validation_seal != sha256_bytes(canonical_bytes(unsigned)):
        raise RuntimeError("validation manifest seal drift")

    steps_by_route: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        steps_by_route[task["semantic_route_key"]].append(task["step"])
    matrix = {
        "schema": SCHEMA,
        "matrix_id": MATRIX_ID,
        "claim": "fixed-independent-not-proven-held-out",
        "validation_stream": {
            "stream_id": STREAM_ID,
            "manifest_file_sha256": sha256_file(args.validation_manifest),
            "manifest_seal_sha256": validation_seal,
            "validation_tokens": 6_291_456,
        },
        "evaluation_contract": {
            "sequences": 768,
            "stored_tokens_per_sequence": 8193,
            "scored_tokens_per_sequence": 8192,
            "batch_size": 1,
            "ce_chunk_tokens": 256,
            "model_parameter_dtype": "bfloat16",
            "cross_entropy_dtype": "float32",
            "accumulation_dtype": "float64",
            "attention_backend": "SDPBackend.MATH",
            "inference_semantics": "pinned TorchTitan causal semantics",
        },
        "source_inventory_sha256": sha256_file(args.inventory),
        "scope": {
            "task_count": len(tasks),
            "steps_by_route": dict(sorted(steps_by_route.items())),
        },
        "tasks": tasks,
    }
    matrix["matrix_seal_sha256"] = sha256_bytes(canonical_bytes(matrix))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(f"R16_MATRIX_PASS tasks=44 seal={matrix['matrix_seal_sha256']}")


if __name__ == "__main__":
    main()
