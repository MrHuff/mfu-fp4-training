#!/usr/bin/env python3
"""Collect canonical local downstream results into the compact r25 ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re


EXPECTED_STEP = 38_000
EXPECTED_EVALUATOR_SHA256 = (
    "48827b0f2bb1cb263e6ff5b1d851ce3cd45bd472d87554a86771076b74409466"
)
EXPECTED_ROUTE_ORDER = (
    "bf16",
    "te_native",
    "pure_v5",
    "mxfp4",
    "localcta",
    "localcta_mxfp4_hybrid",
    "operand_h16",
    "localcta_h16",
    "mxfp4_h32",
)
ROUTE_LABELS = {
    "bf16": "BF16",
    "te_native": "TE-native NVFP4",
    "pure_v5": "pure v5",
    "mxfp4": "MXFP4-v4 + row-SR",
    "localcta": "repaired localCTA-v4",
    "localcta_mxfp4_hybrid": "localCTA/MXFP4 hybrid",
    "operand_h16": "operand hybrid (H16)",
    "localcta_h16": "localCTA + fixed signed H16 column RHT",
    "mxfp4_h32": "MXFP4 + fixed signed H32 column RHT",
}
TASKS = {
    "mmlu": (5, "acc", "mmlu_acc"),
    "hellaswag": (10, "acc_norm", "hellaswag_acc_norm"),
    "winogrande": (5, "acc", "winogrande_acc"),
    "arc_challenge": (25, "acc_norm", "arc_challenge_acc_norm"),
}
COMPACT_COLUMNS = (
    "semantic_route_key",
    "route_label",
    "exact_step",
    "status",
    "mmlu_acc",
    "hellaswag_acc_norm",
    "winogrande_acc",
    "arc_challenge_acc_norm",
)
SHA256SUM_RE = re.compile(r"^([0-9a-f]{64})  (.+)$")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_probability(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise RuntimeError(f"{field} is not a finite probability")
    return float(value)


def parse_sha256sums(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("SHA256SUMS is not UTF-8") from error
    if not text or not text.endswith("\n"):
        raise RuntimeError("SHA256SUMS is empty or lacks terminal newline")
    entries: dict[str, str] = {}
    order: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        match = SHA256SUM_RE.fullmatch(line)
        if match is None:
            raise RuntimeError(f"malformed SHA256SUMS line {number}")
        digest, raw_path = match.groups()
        if not raw_path.startswith("./"):
            raise RuntimeError(f"SHA256SUMS path lacks ./ prefix at line {number}")
        relative = raw_path[2:]
        path = PurePosixPath(relative)
        if (
            not relative
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or relative in entries
            or relative == "SHA256SUMS"
        ):
            raise RuntimeError(f"unsafe or duplicate SHA256SUMS path at line {number}")
        entries[relative] = digest
        order.append(relative)
    if order != sorted(order):
        raise RuntimeError("SHA256SUMS path order is not canonical")
    return entries


def validate_checksums(root: Path) -> None:
    sums = parse_sha256sums((root / "SHA256SUMS").read_bytes())
    if set(sums) != {"canonical-parity.json", "metrics.json"}:
        raise RuntimeError("route result inventory drift")
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    if actual != set(sums):
        raise RuntimeError("route files differ from checksum inventory")
    for name, expected in sums.items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"route result checksum drift: {name}")


def validate_route(root: Path, semantic_route_key: str) -> dict[str, object]:
    validate_checksums(root)
    parity = json.loads((root / "canonical-parity.json").read_bytes())
    if parity.get("passed") is not True:
        raise RuntimeError(f"canonical parity failed: {semantic_route_key}")
    document = json.loads((root / "metrics.json").read_bytes())
    if set(document) != {
        "schema",
        "semantic_route_key",
        "exact_step",
        "evaluator_sha256",
        "tasks",
    }:
        raise RuntimeError(f"metrics fields drift: {semantic_route_key}")
    if (
        document["schema"] != "mfu_public_downstream_metrics_v1"
        or document["semantic_route_key"] != semantic_route_key
        or document["exact_step"] != EXPECTED_STEP
        or document["evaluator_sha256"] != EXPECTED_EVALUATOR_SHA256
        or set(document["tasks"]) != set(TASKS)
    ):
        raise RuntimeError(f"metrics identity drift: {semantic_route_key}")
    scores: dict[str, float] = {}
    for task, (shots, metric_key, output_field) in TASKS.items():
        record = document["tasks"][task]
        if set(record) != {"shots", "metric", "value"}:
            raise RuntimeError(f"task fields drift: {semantic_route_key}/{task}")
        if record["shots"] != shots or record["metric"] != metric_key:
            raise RuntimeError(f"task contract drift: {semantic_route_key}/{task}")
        scores[output_field] = require_probability(
            record["value"], f"{semantic_route_key}/{task}"
        )
    return {
        "semantic_route_key": semantic_route_key,
        "route_label": ROUTE_LABELS[semantic_route_key],
        "exact_step": EXPECTED_STEP,
        "status": "complete",
        **scores,
    }


def compact_json_bytes(rows: list[dict[str, object]]) -> bytes:
    for row in rows:
        if tuple(row) != COMPACT_COLUMNS or set(row) != set(COMPACT_COLUMNS):
            raise RuntimeError("compact JSON row schema drift")
    return (json.dumps(rows, indent=2, ensure_ascii=True) + "\n").encode()


def compact_csv_bytes(rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COMPACT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if tuple(row) != COMPACT_COLUMNS:
            raise RuntimeError("compact CSV row schema drift")
        writer.writerow(row)
    return stream.getvalue().encode()


def render_readme(rows: list[dict[str, object]]) -> bytes:
    lines = [
        "# Corrected scaled-RoPE step-38k downstream ledger",
        "",
        "Every row uses the same canonical MMLU 5-shot, HellaSwag 10-shot,",
        "WinoGrande 5-shot, and ARC-Challenge 25-shot panel at seed 42.",
        "The public ledger intentionally excludes checkpoint locations, scheduler",
        "identities, and operational receipts.",
        "",
        "| route | MMLU | HellaSwag | WinoGrande | ARC-Challenge |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['route_label']} | {row['mmlu_acc']:.10f} | "
            f"{row['hellaswag_acc_norm']:.10f} | {row['winogrande_acc']:.10f} | "
            f"{row['arc_challenge_acc_norm']:.10f} |"
        )
    return ("\n".join(lines) + "\n").encode()


def write_output(output_dir: Path, rows: list[dict[str, object]]) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    files = {
        "DOWNSTREAM_EVAL_LEDGER.csv": compact_csv_bytes(rows),
        "README.md": render_readme(rows),
    }
    for name, payload in files.items():
        (output_dir / name).write_bytes(payload)
    ledger = "".join(
        f"{sha256_bytes(payload)}  {name}\n" for name, payload in sorted(files.items())
    ).encode()
    (output_dir / "SHA256SUMS").write_bytes(ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        validate_route(args.results / route, route) for route in EXPECTED_ROUTE_ORDER
    ]
    write_output(args.output_dir, rows)
    print(f"DOWNSTREAM_LEDGER_PASS routes={len(rows)}")


if __name__ == "__main__":
    main()
