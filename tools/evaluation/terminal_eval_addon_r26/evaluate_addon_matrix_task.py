#!/usr/bin/env python3
"""Envelope adapter for three terminal tasks appended to the r16 evaluator.

The adapter changes only matrix-envelope validation.  It carries no checkpoint
specification, location, scheduler configuration, or submission receipt.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


BASE = (
    Path(__file__).resolve().parents[1]
    / "validation_matrix_r16/evaluate_matrix_task.py"
)
MATRIX_SCHEMA = "mfu_llama_fixed_independent_validation_matrix_r26_v1"
MATRIX_ID = "fixed-independent-r26-terminal-addon-20260902"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def load_base():
    spec = importlib.util.spec_from_file_location("mfu_r16_public_evaluator", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load public r16 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_addon_matrix(base, matrix: dict) -> None:
    unsigned = dict(matrix)
    observed_seal = unsigned.pop("matrix_seal_sha256", None)
    if observed_seal != sha256_bytes(canonical_bytes(unsigned)):
        raise RuntimeError("r26 matrix seal drift")
    tasks = matrix.get("tasks", [])
    if (
        matrix.get("schema") != MATRIX_SCHEMA
        or matrix.get("matrix_id") != MATRIX_ID
        or matrix.get("claim") != "fixed-independent-not-proven-held-out"
        or matrix.get("scope", {}).get("task_count") != 47
        or matrix.get("scope", {}).get("addon_task_count") != 3
        or len(tasks) != 47
    ):
        raise RuntimeError("r26 matrix identity drift")
    parent_digest = sha256_bytes(canonical_bytes(tasks[:44]))
    if matrix.get("source_authorities", {}).get("parent_tasks_sha256") != parent_digest:
        raise RuntimeError("the original 44 tasks are not bound by the parent digest")
    addon_cells = [
        (task.get("semantic_route_key"), task.get("step"), task.get("shard_count"))
        for task in tasks[44:]
    ]
    if addon_cells != [
        ("te_fol4", 38000, 32),
        ("operand_h32", 29000, 64),
        ("operand_h32", 38000, 64),
    ]:
        raise RuntimeError(f"r26 add-on cell drift: {addon_cells}")
    identities: set[str] = set()
    for index, task in enumerate(tasks):
        if task.get("index") != index:
            raise RuntimeError("r26 task index drift")
        identity = task.get("task_identity_sha256")
        if identity != base.task_identity(task):
            raise RuntimeError(f"r26 task identity drift: {index}")
        if identity in identities:
            raise RuntimeError("duplicate r26 task identity")
        identities.add(identity)
        if task.get("shard_count") not in {32, 64}:
            raise RuntimeError(f"r26 task shard geometry drift: {index}")


def main() -> None:
    base = load_base()
    base.validate_matrix = lambda matrix: validate_addon_matrix(base, matrix)
    base.main()


if __name__ == "__main__":
    main()
