#!/usr/bin/env python3
"""Validate local r16 results and emit the compact r22 scientific ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


MATRIX_SCHEMA = "mfu_llama_fixed_independent_validation_matrix_r16_v1"
RESULT_SCHEMA = "mfu_llama_fixed_independent_scaledrope_validation_result_r19_v1"
CSV_COLUMNS = (
    "semantic_route_key",
    "route_label",
    "exact_step",
    "status",
    "nll",
    "sequence_level_standard_error",
    "scored_sequences",
    "target_tokens",
)
HASHED_ARTIFACTS = (
    "validation-result.json",
    "canonical-parity.json",
)
TASK_IDENTITY_FIELDS = (
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


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def canonical_parity_bytes(value: object) -> bytes:
    return canonical_bytes(value) + b"\n"


def verify_canonical_parity_seal(parity: dict) -> None:
    unsigned = dict(parity)
    observed = unsigned.pop("receipt_sha256", None)
    if observed != sha256_bytes(canonical_parity_bytes(unsigned)):
        raise RuntimeError("canonical parity receipt seal drift")


def task_identity(task: dict) -> str:
    return sha256_bytes(
        canonical_bytes({field: task[field] for field in TASK_IDENTITY_FIELDS})
    )


def validate_matrix(matrix: dict) -> None:
    unsigned = dict(matrix)
    seal = unsigned.pop("matrix_seal_sha256", None)
    if seal != sha256_bytes(canonical_bytes(unsigned)):
        raise RuntimeError("matrix seal drift")
    if matrix.get("schema") != MATRIX_SCHEMA or len(matrix.get("tasks", [])) != 44:
        raise RuntimeError("matrix identity drift")
    for index, task in enumerate(matrix["tasks"]):
        if (
            task.get("index") != index
            or task.get("task_identity_sha256") != task_identity(task)
        ):
            raise RuntimeError(f"matrix task identity drift: {index}")


def parse_ledger(payload: bytes) -> dict[str, str]:
    if not payload.endswith(b"\n"):
        raise RuntimeError("checksum ledger must end with newline")
    result: dict[str, str] = {}
    for line in payload.decode().splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or name in result:
            raise RuntimeError("malformed checksum ledger")
        int(digest, 16)
        result[name] = digest
    if tuple(result) != HASHED_ARTIFACTS:
        raise RuntimeError("checksum ledger inventory drift")
    return result


def validate_metric(metric: dict) -> tuple[float, float, int, int]:
    sequences = metric.get("sequences")
    targets = metric.get("targets_per_sequence")
    tokens = metric.get("token_count")
    loss_sums = metric.get("per_sequence_loss_sums")
    if (
        sequences != 768
        or targets != 8192
        or tokens != 6_291_456
        or not isinstance(loss_sums, list)
        or len(loss_sums) != 768
    ):
        raise RuntimeError("validation metric geometry drift")
    values = [float(value) / targets for value in loss_sums]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite validation metric")
    nll = math.fsum(float(value) for value in loss_sums) / tokens
    if not math.isclose(nll, float(metric["nll"]), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("validation NLL arithmetic drift")
    standard_error = statistics.stdev(values) / math.sqrt(sequences)
    return nll, standard_error, sequences, tokens


def pending_record(task: dict) -> dict:
    return {
        "semantic_route_key": task["semantic_route_key"],
        "route_label": task["route_label"],
        "exact_step": task["step"],
        "status": "pending",
        "nll": None,
        "sequence_level_standard_error": None,
        "scored_sequences": None,
        "target_tokens": None,
    }


def collect_one(task: dict, result_dir: Path) -> dict:
    result_path = result_dir / "validation-result.json"
    if not result_path.exists():
        if result_dir.exists() and any(result_dir.iterdir()):
            raise RuntimeError(f"partial result directory: task {task['index']}")
        return pending_record(task)
    paths = {name: result_dir / name for name in HASHED_ARTIFACTS}
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError(f"partial result inventory: task {task['index']}")
    checksums = parse_ledger((result_dir / "SHA256SUMS").read_bytes())
    for name, path in paths.items():
        if sha256_bytes(path.read_bytes()) != checksums[name]:
            raise RuntimeError(f"result checksum drift: task {task['index']} {name}")
    result = json.loads(result_path.read_bytes())
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("task_index") != task["index"]
        or result.get("task_identity_sha256") != task["task_identity_sha256"]
        or result.get("semantic_route_key") != task["semantic_route_key"]
        or result.get("step") != task["step"]
    ):
        raise RuntimeError(f"result identity drift: task {task['index']}")
    parity = json.loads(paths["canonical-parity.json"].read_bytes())
    if not parity.get("passed"):
        raise RuntimeError(f"canonical parity failed: task {task['index']}")
    nll, error, sequences, tokens = validate_metric(result["torchtitan_native_8192"])
    return {
        "semantic_route_key": task["semantic_route_key"],
        "route_label": task["route_label"],
        "exact_step": task["step"],
        "status": "complete",
        "nll": nll,
        "sequence_level_standard_error": error,
        "scored_sequences": sequences,
        "target_tokens": tokens,
    }


def render_csv(records: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode()


def write_outputs(output_dir: Path, matrix_payload: bytes, records: list[dict]) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    csv_payload = render_csv(records)
    (output_dir / "VALIDATION_LEDGER.csv").write_bytes(csv_payload)
    summary = {
        "schema": "mfu_public_validation_ledger_summary_v1",
        "matrix_file_sha256": sha256_bytes(matrix_payload),
        "ledger_sha256": sha256_bytes(csv_payload),
        "summary": {
            "expected": len(records),
            "complete": sum(record["status"] == "complete" for record in records),
            "pending": sum(record["status"] == "pending" for record in records),
        },
    }
    summary_payload = json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n"
    (output_dir / "SUMMARY.json").write_bytes(summary_payload)
    (output_dir / "SHA256SUMS").write_text(
        f"{sha256_bytes(csv_payload)}  VALIDATION_LEDGER.csv\n"
        f"{sha256_bytes(summary_payload)}  SUMMARY.json\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix_payload = args.matrix.read_bytes()
    matrix = json.loads(matrix_payload)
    validate_matrix(matrix)
    records = [
        collect_one(task, args.results / f"task-{task['index']:03d}")
        for task in matrix["tasks"]
    ]
    summary = write_outputs(args.output_dir, matrix_payload, records)
    print(
        "R22_VALIDATION_LEDGER_PASS "
        f"complete={summary['summary']['complete']} "
        f"pending={summary['summary']['pending']}"
    )


if __name__ == "__main__":
    main()
