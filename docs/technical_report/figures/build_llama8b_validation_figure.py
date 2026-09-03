#!/usr/bin/env python3
"""Plot exact-checkpoint validation NLL and matched BF16 differences."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


TOKENS_PER_STEP = 4_194_304
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
    "te_native": "TE-native NVFP4",
    "te_fol4": "TE NVFP4 + 4 BF16 blocks",
    "pure_v5": "Custom global NVFP4",
    "mxfp4": "MXFP4 + row-SR",
    "mxfp4_h32": "MXFP4 + row-SR + fixed H32",
    "localcta": "CTA-local NVFP4",
    "localcta_h16": "CTA-local + fixed H16",
    "localcta_mxfp4_hybrid": "27/5 depth hybrid",
    "operand_h16": "Operand hybrid + plain H16",
    "operand_h32": "Operand hybrid + fixed H32",
}
EXPECTED_ROUTE_STEPS = {
    "bf16": {2_000, 10_000, 18_000, 29_000, 38_000},
    "te_native": {2_000, 10_000, 18_000, 29_000, 38_000},
    "te_fol4": {2_000, 10_000, 18_000, 29_000, 38_000, 38_147},
    "pure_v5": {2_000, 10_000, 18_000, 29_000, 38_000},
    "mxfp4": {2_000, 10_000, 18_000, 29_000, 38_000},
    "mxfp4_h32": {2_000, 10_000, 18_000, 29_000, 38_000},
    "localcta": {2_000, 10_000, 18_000, 29_000, 38_000},
    "localcta_h16": {10_000, 18_000, 29_000, 38_000},
    "localcta_mxfp4_hybrid": {2_000, 10_000, 18_000, 29_000, 38_000},
    "operand_h16": {2_000, 10_000, 18_000, 29_000, 38_000},
    "operand_h32": {2_000, 3_000, 10_000, 18_000, 19_000, 29_000, 38_000, 38_147},
}
COLORS = {
    "bf16": "#30343B",
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
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "").strip()
    if not value:
        raise ValueError(f"missing {field} for {row.get('semantic_route_key')} step {row.get('exact_step')}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field}: {value}")
    return result


def load_rows(path: Path) -> list[dict[str, float | int | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    required = {
        "semantic_route_key",
        "exact_step",
        "status",
        "nll",
        "sequence_level_standard_error",
        "scored_sequences",
        "target_tokens",
        "global_training_tokens",
    }
    if not raw_rows or not required.issubset(raw_rows[0]):
        missing = required - (set(raw_rows[0]) if raw_rows else set())
        raise ValueError(f"validation ledger is empty or missing columns: {sorted(missing)}")

    rows: list[dict[str, float | int | str]] = []
    seen: set[tuple[str, int]] = set()
    for raw in raw_rows:
        if raw["status"].strip().lower() not in {"complete", "completed", "passed"}:
            raise ValueError(
                f"validation ledger contains a non-complete row: "
                f"{raw.get('semantic_route_key')} step {raw.get('exact_step')}"
            )
        route = raw["semantic_route_key"].strip()
        step = int(raw["exact_step"])
        key = (route, step)
        if key in seen:
            raise ValueError(f"duplicate completed validation row: {key}")
        seen.add(key)
        nll = _number(raw, "nll")
        standard_error = _number(raw, "sequence_level_standard_error")
        if standard_error < 0.0:
            raise ValueError(f"negative standard error for {route} step {step}")
        if int(raw["scored_sequences"]) != 768:
            raise ValueError(f"unexpected sequence count for {route} step {step}")
        if int(raw["target_tokens"]) != 6_291_456:
            raise ValueError(f"unexpected target-token count for {route} step {step}")
        token_text = raw.get("global_training_tokens", "").strip()
        expected_tokens = step * TOKENS_PER_STEP
        tokens = float(token_text)
        if not tokens.is_integer() or not expected_tokens <= tokens < expected_tokens + TOKENS_PER_STEP:
            raise ValueError(f"invalid global_training_tokens for {route} step {step}")
        rows.append(
            {
                "route": route,
                "step": step,
                "tokens": tokens,
                "nll": nll,
                "standard_error": standard_error,
            }
        )
    if not rows:
        raise ValueError("validation ledger contains no completed finite rows")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.ledger)
    by_route: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        by_route[str(row["route"])].append(row)
    if "bf16" not in by_route:
        raise ValueError("validation ledger has no completed BF16 reference rows")

    expected_cell_count = sum(len(steps) for steps in EXPECTED_ROUTE_STEPS.values())
    if len(rows) != expected_cell_count:
        raise ValueError(f"expected {expected_cell_count} exact validation cells, found {len(rows)}")
    if set(by_route) != set(EXPECTED_ROUTE_STEPS):
        raise ValueError(
            f"unexpected validation route inventory: observed={sorted(by_route)} "
            f"expected={sorted(EXPECTED_ROUTE_STEPS)}"
        )
    for route, expected_steps in EXPECTED_ROUTE_STEPS.items():
        observed_steps = {int(row["step"]) for row in by_route[route]}
        if observed_steps != expected_steps:
            raise ValueError(
                f"unexpected exact-step inventory for {route}: "
                f"observed={sorted(observed_steps)} expected={sorted(expected_steps)}"
            )

    bf16 = {int(row["step"]): float(row["nll"]) for row in by_route["bf16"]}
    order = [route for route in ROUTE_ORDER if route in by_route]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": "#7A8088",
            "axes.labelcolor": "#30343B",
            "xtick.color": "#30343B",
            "ytick.color": "#30343B",
        }
    )
    fig, (ax_nll, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(10.2, 9.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0], "hspace": 0.12},
    )

    for index, route in enumerate(order):
        route_rows = sorted(by_route[route], key=lambda row: int(row["step"]))
        x = [float(row["tokens"]) / 1e9 for row in route_rows]
        y = [float(row["nll"]) for row in route_rows]
        yerr = [1.96 * float(row["standard_error"]) for row in route_rows]
        color = COLORS.get(route, "#7A8088")
        marker = MARKERS[index % len(MARKERS)]
        label = ROUTE_LABELS.get(route, route.replace("_", " "))
        ax_nll.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            marker=marker,
            markersize=5.2,
            linewidth=1.6,
            elinewidth=0.8,
            capsize=2.0,
            label=label,
        )

        matched = [row for row in route_rows if int(row["step"]) in bf16]
        delta_x = [float(row["tokens"]) / 1e9 for row in matched]
        delta_y = [
            100.0
            * (bf16[int(row["step"])] - float(row["nll"]))
            / bf16[int(row["step"])]
            for row in matched
        ]
        ax_delta.plot(
            delta_x,
            delta_y,
            color=color,
            marker=marker,
            markersize=5.2,
            linewidth=1.6,
        )

    for axis in (ax_nll, ax_delta):
        axis.grid(color="#E9EBEE", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    ax_nll.set_ylabel("Validation NLL")
    ax_nll.set_title("Fixed external validation stream at exact checkpoints", fontweight="bold")
    ax_delta.axhline(0.0, color="#30343B", linewidth=1.0, linestyle="--")
    ax_delta.set_ylabel("Relative NLL difference vs BF16 (%)")
    ax_delta.set_xlabel("Training tokens (billions)")
    handles, labels = ax_nll.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.2,
        handlelength=2.0,
    )
    fig.text(
        0.995,
        0.125,
        "Markers are evaluated checkpoints; bars show 95% sequence-level intervals; negative relative values are worse.",
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#7A8088",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.23)
    fig.savefig(args.output, dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
