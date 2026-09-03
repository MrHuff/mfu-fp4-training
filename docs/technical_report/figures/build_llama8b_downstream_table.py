#!/usr/bin/env python3
"""Render the exact-step downstream ledger as a LaTeX table-row fragment."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROUTE_ORDER = [
    "bf16",
    "te_native",
    "te_fol4",
    "pure_v5",
    "mxfp4",
    "mxfp4_h32",
    "localcta",
    "localcta_h16",
    "localcta_mxfp4_hybrid",
    "operand_h16",
    "operand_h32",
]
ROUTE_LABELS = {
    "bf16": "BF16",
    "te_native": r"TE-native \nvfp{}",
    "te_fol4": r"TE \nvfp{}, four final BF16 blocks",
    "pure_v5": r"Global \nvfp{} v5",
    "mxfp4": r"\mxfp{} + row-SR",
    "mxfp4_h32": r"\mxfp{} + row-SR + fixed H32",
    "localcta": r"\localcta{}",
    "localcta_h16": r"\localcta{} + fixed H16",
    "localcta_mxfp4_hybrid": "27/5 depth hybrid",
    "operand_h16": "Operand hybrid + plain H16",
    "operand_h32": "Operand hybrid + fixed H32",
}
METRICS = [
    "mmlu_acc",
    "hellaswag_acc_norm",
    "winogrande_acc",
    "arc_challenge_acc_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    required = {"semantic_route_key", "exact_step", "status", *METRICS}
    if not raw_rows or not required.issubset(raw_rows[0]):
        missing = required - (set(raw_rows[0]) if raw_rows else set())
        raise ValueError(f"downstream ledger is empty or missing columns: {sorted(missing)}")

    rows: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        if row["status"].strip().lower() not in {"complete", "completed", "passed"}:
            continue
        route = row["semantic_route_key"].strip()
        step = int(row["exact_step"])
        if step != 38_000:
            raise ValueError(f"downstream row is not exact step 38,000: {route} at {step}")
        if route in rows:
            raise ValueError(f"duplicate completed downstream row: {route}")
        for field in METRICS:
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {field} for {route}: {row[field]}")
        rows[route] = row
    if "bf16" not in rows:
        raise ValueError("downstream ledger has no completed BF16 reference")
    return rows


def escape_unknown_label(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.ledger)
    expected_routes = set(ROUTE_ORDER)
    if set(rows) != expected_routes:
        missing = sorted(expected_routes - set(rows))
        unexpected = sorted(set(rows) - expected_routes)
        raise ValueError(
            f"downstream route inventory changed: missing={missing}, "
            f"unexpected={unexpected}"
        )

    maxima = {
        field: max(float(rows[route][field]) for route in ROUTE_ORDER)
        for field in METRICS
    }

    lines = ["% Generated from the exact-step downstream ledger; do not edit by hand."]
    for route in ROUTE_ORDER:
        row = rows[route]
        label = ROUTE_LABELS.get(route, escape_unknown_label(route.replace("_", " ")))
        values = []
        for field in METRICS:
            raw_value = float(row[field])
            rendered = f"{100.0 * raw_value:.2f}"
            if raw_value == maxima[field]:
                rendered = rf"\textbf{{{rendered}}}"
            values.append(rendered)
        lines.append(
            f"{label} & {values[0]} & {values[1]} & "
            f"{values[2]} & {values[3]} \\\\"
        )
    # Keep the final booktabs rule inside the included fragment.  LaTeX's
    # input-file hook runs after \input returns and is not alignment material;
    # placing \bottomrule after \input can therefore trigger a misplaced
    # \noalign error on current LaTeX releases.
    lines.append(r"\bottomrule")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
