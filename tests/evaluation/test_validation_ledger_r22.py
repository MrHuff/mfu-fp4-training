"""CPU-only tests for the public r22 validation collector."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/evaluation/validation_ledger_r22/collect_validation.py"
)
SPEC = importlib.util.spec_from_file_location("r22_validation_collector", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def task(index: int) -> dict:
    value = {
        "semantic_route_key": f"route-{index // 5}",
        "route_label": f"Route {index // 5}",
        "training_recipe": "synthetic-test-recipe",
        "lineage_note": "synthetic public test",
        "converter_route": "bf16-unfused-v1",
        "checkpoint_key": f"checkpoint-{index:02d}",
        "step": index * 1000,
        "expected_ntokens_seen": index * 100,
        "metadata_sha256": f"{index:064x}",
        "shard_count": 32,
    }
    value["task_identity_sha256"] = collector.task_identity(value)
    value["index"] = index
    return value


def matrix() -> dict:
    value = {"schema": collector.MATRIX_SCHEMA, "tasks": [task(i) for i in range(44)]}
    value["matrix_seal_sha256"] = collector.sha256_bytes(collector.canonical_bytes(value))
    return value


def test_contract_and_task_identity() -> None:
    value = matrix()
    collector.validate_matrix(value)
    value["tasks"][3]["step"] += 1
    with pytest.raises(RuntimeError, match="matrix seal drift"):
        collector.validate_matrix(value)


def test_pending_partial_and_exact_csv() -> None:
    with tempfile.TemporaryDirectory() as temp:
        result = Path(temp) / "result"
        record = collector.collect_one(task(0), result)
        assert record["status"] == "pending"
        rows = list(csv.DictReader(io.StringIO(collector.render_csv([record]).decode())))
        assert tuple(rows[0]) == collector.CSV_COLUMNS
        result.mkdir()
        (result / "unexpected.json").write_text("{}")
        with pytest.raises(RuntimeError, match="partial result directory"):
            collector.collect_one(task(0), result)


def test_metric_and_strict_ledger() -> None:
    losses = [8192.0] * 768
    metric = {
        "sequences": 768,
        "targets_per_sequence": 8192,
        "token_count": 6_291_456,
        "nll": 1.0,
        "per_sequence_loss_sums": losses,
    }
    assert collector.validate_metric(metric) == (1.0, 0.0, 768, 6_291_456)
    payload = b"".join(
        f"{hashlib.sha256(name.encode()).hexdigest()}  {name}\n".encode()
        for name in collector.HASHED_ARTIFACTS
    )
    assert tuple(collector.parse_ledger(payload)) == collector.HASHED_ARTIFACTS
    with pytest.raises(RuntimeError):
        collector.parse_ledger(payload.rstrip(b"\n"))


def test_parity_seal_uses_newline_contract() -> None:
    unsealed = {"nested": {"passed": True}, "schema_version": 1}
    receipt = dict(unsealed)
    receipt["receipt_sha256"] = collector.sha256_bytes(
        collector.canonical_parity_bytes(unsealed)
    )
    collector.verify_canonical_parity_seal(receipt)
    receipt["nested"] = {"passed": False}
    with pytest.raises(RuntimeError, match="canonical parity receipt seal drift"):
        collector.verify_canonical_parity_seal(receipt)


def test_write_emits_all_44_with_exact_header() -> None:
    value = matrix()
    payload = json.dumps(value, sort_keys=True).encode()
    records = [collector.pending_record(item) for item in value["tasks"]]
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "sealed"
        summary = collector.write_outputs(output, payload, records)
        assert summary["summary"] == {"expected": 44, "complete": 0, "pending": 44}
        with (output / "VALIDATION_LEDGER.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 44
        assert all(row["status"] == "pending" for row in rows)
