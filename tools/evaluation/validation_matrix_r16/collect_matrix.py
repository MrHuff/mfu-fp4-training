#!/usr/bin/env python3
"""Collect local r16 validation results into a compact scientific ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


RESULT_SCHEMA = "mfu_llama_fixed_independent_scaledrope_validation_result_r19_v1"
CSV_FIELDS = (
    "semantic_route_key",
    "route_label",
    "exact_step",
    "status",
    "nll",
    "perplexity",
    "sequence_level_standard_error",
    "scored_sequences",
    "target_tokens",
    "bf16_nll_same_exact_step",
    "nll_percent_difference_from_bf16",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_ledger(payload: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in payload.decode().splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not name or name in result:
            raise RuntimeError("malformed result checksum ledger")
        int(digest, 16)
        result[name] = digest
    return result


def collect_one(task: dict, result_dir: Path) -> dict:
    result_path = result_dir / "validation-result.json"
    if not result_path.is_file():
        return {"status": "pending"}
    result_payload = result_path.read_bytes()
    checksums = parse_ledger((result_dir / "SHA256SUMS").read_bytes())
    if checksums.get(result_path.name) != sha256_bytes(result_payload):
        raise RuntimeError(f"result hash drift: task {task['index']}")
    parity_path = result_dir / "canonical-parity.json"
    if checksums.get(parity_path.name) != sha256_bytes(parity_path.read_bytes()):
        raise RuntimeError(f"parity hash drift: task {task['index']}")
    result = json.loads(result_payload)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("task_index") != task["index"]
        or result.get("task_identity_sha256") != task["task_identity_sha256"]
        or result.get("semantic_route_key") != task["semantic_route_key"]
        or result.get("step") != task["step"]
    ):
        raise RuntimeError(f"result identity drift: task {task['index']}")
    metric = result.get("torchtitan_native_8192", {})
    loss_sums = metric.get("per_sequence_loss_sums", [])
    if (
        metric.get("sequences") != 768
        or metric.get("targets_per_sequence") != 8192
        or metric.get("token_count") != 6_291_456
        or len(loss_sums) != 768
    ):
        raise RuntimeError(f"validation geometry drift: task {task['index']}")
    values = [float(value) / 8192 for value in loss_sums]
    nll = float(metric["nll"])
    if not math.isfinite(nll):
        raise RuntimeError(f"non-finite NLL: task {task['index']}")
    return {
        "status": "complete",
        "nll": nll,
        "perplexity": float(metric["perplexity"]),
        "sequence_level_standard_error": statistics.stdev(values)
        / math.sqrt(len(values)),
        "scored_sequences": 768,
        "target_tokens": 6_291_456,
    }


def render_csv(records: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    matrix_payload = args.matrix.read_bytes()
    matrix = json.loads(matrix_payload)
    if len(matrix.get("tasks", [])) != 44:
        raise RuntimeError("r16 collection matrix count drift")

    records = []
    for task in matrix["tasks"]:
        observed = collect_one(task, args.results / f"task-{task['index']:03d}")
        records.append(
            {
                "semantic_route_key": task["semantic_route_key"],
                "route_label": task["route_label"],
                "exact_step": task["step"],
                **observed,
            }
        )
    bf16 = {
        record["exact_step"]: record["nll"]
        for record in records
        if record["semantic_route_key"] == "bf16" and record["status"] == "complete"
    }
    for record in records:
        reference = bf16.get(record["exact_step"])
        if record["status"] == "complete" and reference is not None:
            record["bf16_nll_same_exact_step"] = reference
            record["nll_percent_difference_from_bf16"] = (
                100.0 * (record["nll"] - reference) / reference
            )
        else:
            record["bf16_nll_same_exact_step"] = ""
            record["nll_percent_difference_from_bf16"] = ""

    args.output_dir.mkdir(parents=True)
    csv_payload = render_csv(records)
    (args.output_dir / "VALIDATION_MATRIX.csv").write_bytes(csv_payload)
    summary = {
        "schema": "mfu_public_validation_matrix_summary_v1",
        "matrix_file_sha256": sha256_bytes(matrix_payload),
        "ledger_sha256": sha256_bytes(csv_payload),
        "expected": len(records),
        "complete": sum(record["status"] == "complete" for record in records),
        "pending": sum(record["status"] == "pending" for record in records),
    }
    summary_payload = json.dumps(summary, indent=2, sort_keys=True).encode() + b"\n"
    (args.output_dir / "SUMMARY.json").write_bytes(summary_payload)
    (args.output_dir / "SHA256SUMS").write_text(
        f"{sha256_bytes(csv_payload)}  VALIDATION_MATRIX.csv\n"
        f"{sha256_bytes(summary_payload)}  SUMMARY.json\n"
    )
    print(
        f"R16_COLLECTION_PASS complete={summary['complete']} pending={summary['pending']}"
    )


if __name__ == "__main__":
    main()
