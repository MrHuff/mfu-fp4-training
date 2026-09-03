#!/usr/bin/env python3
"""Plot exact-step downstream accuracy differences from the BF16 checkpoint."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROUTE_ORDER = [
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
    "te_native": "TE-native NVFP4",
    "te_fol4": "TE NVFP4, four final BF16 blocks",
    "pure_v5": "Global NVFP4 v5",
    "mxfp4": "MXFP4 + row-SR",
    "mxfp4_h32": "MXFP4 + row-SR + fixed H32",
    "localcta": "CTA-local NVFP4",
    "localcta_h16": "CTA-local + fixed H16",
    "localcta_mxfp4_hybrid": "27/5 depth hybrid",
    "operand_h16": "Operand hybrid + plain H16",
    "operand_h32": "Operand hybrid + fixed H32",
}
COLORS = {
    "te_native": "#2878B5",
    "te_fol4": "#64A6D8",
    "pure_v5": "#7656A5",
    "mxfp4": "#63A65F",
    "mxfp4_h32": "#24733F",
    "localcta": "#2A9D8F",
    "localcta_h16": "#61C4B8",
    "localcta_mxfp4_hybrid": "#E58B3D",
    "operand_h16": "#C98B37",
    "operand_h32": "#C94C4C",
}
METRICS = [
    ("mmlu_acc", "MMLU, 5-shot"),
    ("hellaswag_acc_norm", "HellaSwag, 10-shot"),
    ("winogrande_acc", "WinoGrande, 5-shot"),
    ("arc_challenge_acc_norm", "ARC-Challenge, 25-shot"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    required = {
        "semantic_route_key",
        "exact_step",
        "status",
        *(field for field, _ in METRICS),
    }
    if not raw_rows or not required.issubset(raw_rows[0]):
        missing = required - (set(raw_rows[0]) if raw_rows else set())
        raise ValueError(f"downstream ledger is empty or missing columns: {sorted(missing)}")

    rows: dict[str, dict[str, str]] = {}
    for row in raw_rows:
        if row["status"].strip().lower() not in {"complete", "completed", "passed"}:
            continue
        route = row["semantic_route_key"].strip()
        if int(row["exact_step"]) != 38_000:
            raise ValueError(f"downstream row is not exact step 38,000: {route}")
        if route in rows:
            raise ValueError(f"duplicate completed downstream row: {route}")
        for field, _ in METRICS:
            value = float(row[field])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {field} for {route}: {row[field]}")
        rows[route] = row
    if "bf16" not in rows:
        raise ValueError("downstream ledger has no completed BF16 reference")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.ledger)
    expected_routes = {"bf16", *ROUTE_ORDER}
    if set(rows) != expected_routes:
        missing = sorted(expected_routes - set(rows))
        unexpected = sorted(set(rows) - expected_routes)
        raise ValueError(
            f"downstream route inventory changed: missing={missing}, "
            f"unexpected={unexpected}"
        )
    route_order = ROUTE_ORDER

    delta_by_metric = {
        field: [
            100.0 * (float(rows[route][field]) - float(rows["bf16"][field]))
            for route in route_order
        ]
        for field, _ in METRICS
    }
    all_deltas = [value for values in delta_by_metric.values() for value in values]
    common_lower = math.floor(min(0.0, *all_deltas))
    common_upper = math.ceil(max(0.0, *all_deltas))
    common_span = common_upper - common_lower

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.edgecolor": "#7A8088",
            "axes.labelcolor": "#30343B",
            "xtick.color": "#30343B",
            "ytick.color": "#30343B",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 9.0), sharex=False)
    y = np.arange(len(route_order))
    labels = [ROUTE_LABELS.get(route, route.replace("_", " ")) for route in route_order]
    colors = [COLORS.get(route, "#7A8088") for route in route_order]

    for axis, (field, title) in zip(axes.flat, METRICS):
        deltas = delta_by_metric[field]
        axis.barh(y, deltas, color=colors, height=0.62)
        axis.set_xlim(common_lower, common_upper)
        axis.set_xticks(
            np.arange(common_lower, common_upper + 0.1, 2.0, dtype=float)
        )
        axis.axvline(0.0, color="#30343B", linewidth=1.0)
        axis.set_yticks(y, labels)
        axis.tick_params(axis="y", length=0)
        axis.invert_yaxis()
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Accuracy difference from BF16 (points)")
        axis.grid(axis="x", color="#E9EBEE", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        for yi, delta in zip(y, deltas):
            if delta < 0.0 and abs(delta) >= 0.18 * common_span:
                x = delta + 0.025 * common_span
                horizontal_alignment = "left"
                text_color = "white"
            elif delta < 0.0:
                x = delta - 0.018 * common_span
                horizontal_alignment = "right"
                text_color = "#20242A"
            else:
                x = delta + 0.018 * common_span
                horizontal_alignment = "left"
                text_color = "#20242A"
            axis.text(
                x,
                yi,
                f"{delta:+.2f}",
                ha=horizontal_alignment,
                va="center",
                fontsize=8.6,
                color=text_color,
            )

    fig.suptitle(
        "Exact step-38,000 downstream difference from BF16",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.0, w_pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
