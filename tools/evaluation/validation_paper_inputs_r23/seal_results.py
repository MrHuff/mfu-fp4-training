#!/usr/bin/env python3
"""Seal a complete r22 ledger as a storage-neutral public artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


REQUIRED = ("VALIDATION_LEDGER.csv", "SUMMARY.json", "SHA256SUMS")
FORBIDDEN_MARKERS = (
    b"aws_access_key_id",
    b"aws_secret_access_key",
    b"aws_session_token",
    b"accessKeyId",
    b"secretAccessKey",
    b"sessionToken",
    b"AKIA",
    b"s3:" + b"//",
    b"/work" + b"space/",
    b"/opt/mfu/EXTERNAL_PATH",
)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value, separator, name = line.partition("  ")
        if separator != "  " or len(value) != 64 or name in result:
            raise RuntimeError("malformed checksum ledger")
        int(value, 16)
        result[name] = value
    return result


def validate_collector_files(result_dir: Path) -> tuple[dict, list[dict[str, str]]]:
    if any(not (result_dir / name).is_file() for name in REQUIRED):
        raise RuntimeError("canonical collector inventory is incomplete")
    checksums = parse_checksums(result_dir / "SHA256SUMS")
    if set(checksums) != {"VALIDATION_LEDGER.csv", "SUMMARY.json"}:
        raise RuntimeError("collector checksum inventory drift")
    for name, expected in checksums.items():
        if digest((result_dir / name).read_bytes()) != expected:
            raise RuntimeError(f"collector checksum drift: {name}")
    summary = json.loads((result_dir / "SUMMARY.json").read_bytes())
    if (
        summary.get("schema") != "mfu_public_validation_ledger_summary_v1"
        or summary.get("summary") != {"expected": 44, "complete": 44, "pending": 0}
    ):
        raise RuntimeError("collector summary is not 44/44 complete")
    with (result_dir / "VALIDATION_LEDGER.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 44 or any(row.get("status") != "complete" for row in rows):
        raise RuntimeError("validation CSV is not exactly 44/44 complete")
    if len({(row["semantic_route_key"], row["exact_step"]) for row in rows}) != 44:
        raise RuntimeError("validation CSV cells are not unique")
    return summary, rows


def ensure_public_safe(paths: list[Path]) -> None:
    for path in paths:
        payload = path.read_bytes()
        for marker in FORBIDDEN_MARKERS:
            if marker.lower() in payload.lower():
                raise RuntimeError(f"private marker in {path.name}: {marker!r}")


def readme(summary: dict, rows: list[dict[str, str]]) -> bytes:
    return (
        "# Fixed-independent validation ledger\n\n"
        "This directory contains the compact, checksum-verified 44-cell r22 "
        "validation ledger used by the technical report. The stream is fixed "
        "and independent, but is not claimed to be proven held out.\n\n"
        f"Cells: {len(rows)} complete. Ledger SHA-256: "
        f"`{summary['ledger_sha256']}`.\n"
    ).encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    summary, rows = validate_collector_files(args.result_dir)
    args.output_dir.mkdir(parents=True)
    for name in ("VALIDATION_LEDGER.csv", "SUMMARY.json"):
        shutil.copyfile(args.result_dir / name, args.output_dir / name)
    (args.output_dir / "README.md").write_bytes(readme(summary, rows))
    outputs = sorted(args.output_dir.iterdir())
    ensure_public_safe(outputs)
    ledger = "".join(
        f"{digest(path.read_bytes())}  {path.name}\n" for path in outputs
    ).encode()
    (args.output_dir / "SHA256SUMS").write_bytes(ledger)
    ensure_public_safe([args.output_dir / "SHA256SUMS"])
    print(f"R23_PUBLIC_LEDGER_PASS cells={len(rows)} files={len(outputs) + 1}")


if __name__ == "__main__":
    main()
