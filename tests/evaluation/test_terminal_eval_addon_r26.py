"""CPU-only contract tests for the public r26 envelope adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/evaluation/terminal_eval_addon_r26/evaluate_addon_matrix_task.py"
)
SPEC = importlib.util.spec_from_file_location("mfu_r26_adapter", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)

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


class FakeBase:
    @staticmethod
    def task_identity(task: dict) -> str:
        return adapter.sha256_bytes(
            adapter.canonical_bytes({field: task[field] for field in TASK_FIELDS})
        )


def make_task(index: int, route: str, step: int, shards: int) -> dict:
    value = {
        "semantic_route_key": route,
        "route_label": route,
        "training_recipe": "synthetic public test",
        "lineage_note": "synthetic public test",
        "converter_route": "bf16-unfused-v1",
        "checkpoint_key": f"checkpoint-{index:02d}",
        "step": step,
        "expected_ntokens_seen": step * 128,
        "metadata_sha256": f"{index + 1:064x}",
        "shard_count": shards,
    }
    value["task_identity_sha256"] = FakeBase.task_identity(value)
    value["index"] = index
    return value


def make_matrix() -> dict:
    tasks = [make_task(index, f"parent-{index}", index * 1000, 32) for index in range(44)]
    tasks.extend(
        [
            make_task(44, "te_fol4", 38000, 32),
            make_task(45, "operand_h32", 29000, 64),
            make_task(46, "operand_h32", 38000, 64),
        ]
    )
    value = {
        "schema": adapter.MATRIX_SCHEMA,
        "matrix_id": adapter.MATRIX_ID,
        "claim": "fixed-independent-not-proven-held-out",
        "scope": {"task_count": 47, "addon_task_count": 3},
        "source_authorities": {
            "parent_tasks_sha256": adapter.sha256_bytes(
                adapter.canonical_bytes(tasks[:44])
            )
        },
        "tasks": tasks,
    }
    value["matrix_seal_sha256"] = adapter.sha256_bytes(adapter.canonical_bytes(value))
    return value


def reseal(matrix: dict) -> None:
    matrix.pop("matrix_seal_sha256", None)
    matrix["matrix_seal_sha256"] = adapter.sha256_bytes(adapter.canonical_bytes(matrix))


def test_exact_addon_matrix_passes() -> None:
    matrix = make_matrix()
    adapter.validate_addon_matrix(FakeBase, matrix)
    assert [task["index"] for task in matrix["tasks"][44:]] == [44, 45, 46]


def test_parent_digest_fails_closed() -> None:
    matrix = make_matrix()
    matrix["tasks"][1]["route_label"] = "tampered"
    matrix["tasks"][1]["task_identity_sha256"] = FakeBase.task_identity(matrix["tasks"][1])
    reseal(matrix)
    with pytest.raises(RuntimeError, match="parent digest"):
        adapter.validate_addon_matrix(FakeBase, matrix)


def test_addon_cell_fails_closed() -> None:
    matrix = make_matrix()
    matrix["tasks"][46]["step"] = 39000
    matrix["tasks"][46]["task_identity_sha256"] = FakeBase.task_identity(matrix["tasks"][46])
    reseal(matrix)
    with pytest.raises(RuntimeError, match="add-on cell drift"):
        adapter.validate_addon_matrix(FakeBase, matrix)


def test_adapter_has_no_embedded_checkpoint_spec() -> None:
    source = MODULE_PATH.read_text()
    assert "CHECKPOINT_SPECS" not in source
    assert "checkpoint_uri" not in source
