#!/usr/bin/env python3
"""Export a completed MXFP4+RHT metric history for the report.

The script never reads a credential file and never serializes an API key.  A
caller must provide ``WANDB_API_KEY`` in the process environment. Source
identity values are caller-supplied for verification and are not serialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path

import wandb


EXPECTED_FIRST_STEP = 1
EXPECTED_LAST_STEP = 38_140
EXPECTED_ROWS = 3_815

WAND_B_KEYS = {
    "_step": "step",
    "loss_metrics/global_avg_loss": "loss",
    "loss_metrics/global_max_loss": "global_max_loss",
    "grad_norm": "grad_norm",
    "throughput(tps)": "tps_per_gpu",
    "mfu(%)": "mfu",
    "n_tokens_seen": "n_tokens_seen",
    "lr": "lr",
    "memory/max_active(GiB)": "max_active_gib",
    "memory/max_reserved(GiB)": "max_reserved_gib",
    "memory/num_alloc_retries": "num_alloc_retries",
    "memory/num_ooms": "num_ooms",
}

OUTPUT_FIELDS = [
    "lineage",
    "display_name",
    *WAND_B_KEYS.values(),
    "value_precision",
    "is_interpolated",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_finite(row: dict[str, object], key: str) -> None:
    value = row.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"history row has nonfinite or missing {key}: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-run-name", required=True)
    parser.add_argument("--expected-state", default="finished")
    parser.add_argument("--history-out", type=Path, required=True)
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY must be supplied in the process environment")

    run = wandb.Api(api_key=api_key, timeout=60).run(args.run_path)
    if run.id != args.expected_run_id:
        raise RuntimeError(f"unexpected run id: {run.id}")
    if run.name != args.expected_run_name:
        raise RuntimeError(f"unexpected run name: {run.name}")
    if run.state != args.expected_state:
        raise RuntimeError(f"run is not sealed as finished: {run.state}")

    raw_rows = list(run.scan_history(page_size=10_000))
    if len(raw_rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"unexpected history row count: {len(raw_rows)} != {EXPECTED_ROWS}"
        )

    rows: list[dict[str, object]] = []
    for raw in raw_rows:
        for key in WAND_B_KEYS:
            require_finite(raw, key)
        row: dict[str, object] = {
            "lineage": "mxfp4_rht",
            "display_name": "MXFP4 + fixed-sign column RHT",
            "value_precision": "exported_float_payload",
            "is_interpolated": False,
        }
        row.update({output: raw[source] for source, output in WAND_B_KEYS.items()})
        rows.append(row)

    steps = [int(row["step"]) for row in rows]
    if steps[0] != EXPECTED_FIRST_STEP or steps[-1] != EXPECTED_LAST_STEP:
        raise RuntimeError(f"unexpected history endpoints: {steps[0]}, {steps[-1]}")
    if steps != sorted(steps) or len(set(steps)) != len(steps):
        raise RuntimeError("history steps are not strictly increasing and unique")

    args.history_out.parent.mkdir(parents=True, exist_ok=True)
    with args.history_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"PUBLIC_METRIC_EXPORT_PASS rows={len(rows)} sha256={sha256(args.history_out)}"
    )


if __name__ == "__main__":
    main()
