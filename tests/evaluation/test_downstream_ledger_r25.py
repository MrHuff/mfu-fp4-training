"""CPU-only tests for the public r25 downstream collector."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/evaluation/downstream_ledger_r25/collect_downstream.py"
)
SPEC = importlib.util.spec_from_file_location("collect_downstream", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_route_contract_is_exact_and_complete() -> None:
    assert len(module.EXPECTED_ROUTE_ORDER) == 9
    assert set(module.EXPECTED_ROUTE_ORDER) == set(module.ROUTE_LABELS)
    assert module.EXPECTED_EVALUATOR_SHA256.startswith("48827b0f")


def test_sha256sum_parser_accepts_canonical_ledger() -> None:
    payload = (
        b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  ./a.json\n"
        b"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  ./x/b.json\n"
    )
    assert module.parse_sha256sums(payload) == {
        "a.json": "a" * 64,
        "x/b.json": "b" * 64,
    }


def test_sha256sum_parser_rejects_traversal_and_duplicates() -> None:
    with pytest.raises(RuntimeError):
        module.parse_sha256sums(b"a" * 64 + b"  ./../a\n")
    duplicate = b"a" * 64 + b"  ./a\n" + b"b" * 64 + b"  ./a\n"
    with pytest.raises(RuntimeError):
        module.parse_sha256sums(duplicate)


def test_compact_schema_is_exact() -> None:
    row = {name: None for name in module.COMPACT_COLUMNS}
    assert b'"semantic_route_key"' in module.compact_json_bytes([row])
    bad = dict(row)
    bad["ambiguous_alias"] = "forbidden"
    with pytest.raises(RuntimeError):
        module.compact_json_bytes([bad])


def test_local_route_validation() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        route = module.EXPECTED_ROUTE_ORDER[0]
        parity = json.dumps({"passed": True}, sort_keys=True).encode() + b"\n"
        metrics = {
            "schema": "mfu_public_downstream_metrics_v1",
            "semantic_route_key": route,
            "exact_step": module.EXPECTED_STEP,
            "evaluator_sha256": module.EXPECTED_EVALUATOR_SHA256,
            "tasks": {
                task: {"shots": shots, "metric": metric, "value": 0.5}
                for task, (shots, metric, _field) in module.TASKS.items()
            },
        }
        metric_payload = json.dumps(metrics, sort_keys=True).encode() + b"\n"
        (root / "canonical-parity.json").write_bytes(parity)
        (root / "metrics.json").write_bytes(metric_payload)
        sums = {
            "canonical-parity.json": hashlib.sha256(parity).hexdigest(),
            "metrics.json": hashlib.sha256(metric_payload).hexdigest(),
        }
        (root / "SHA256SUMS").write_text(
            "".join(f"{digest}  ./{name}\n" for name, digest in sorted(sums.items()))
        )
        row = module.validate_route(root, route)
        assert tuple(row) == module.COMPACT_COLUMNS
        assert row["mmlu_acc"] == 0.5
