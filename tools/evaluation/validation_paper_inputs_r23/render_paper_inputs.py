#!/usr/bin/env python3
"""Render deterministic paper inputs from the compact 44-cell r22 ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


CLAIM = "fixed-independent-not-proven-held-out"
LEDGER_COLUMNS = (
    "semantic_route_key",
    "route_label",
    "exact_step",
    "status",
    "nll",
    "sequence_level_standard_error",
    "scored_sequences",
    "target_tokens",
)

EXPECTED_STEPS = {
    "bf16": [2000, 10000, 18000, 29000, 38000],
    "localcta": [2000, 10000, 18000, 29000, 38000],
    "localcta_h16": [10000, 18000, 29000, 38000],
    "localcta_mxfp4_hybrid": [38000],
    "mxfp4": [2000, 10000, 18000, 29000, 38000],
    "mxfp4_h32": [2000, 10000, 18000, 29000, 38000],
    "operand_h16": [38000],
    "operand_h32": [3000, 10000, 18000, 19000],
    "pure_v5": [2000, 10000, 18000, 29000, 38000],
    "te_fol4": [2000, 10000, 18000, 29000],
    "te_native": [2000, 10000, 18000, 29000, 38000],
}

ROUTE_ORDER = [
    "bf16",
    "te_native",
    "te_fol4",
    "pure_v5",
    "localcta",
    "mxfp4",
    "localcta_h16",
    "mxfp4_h32",
    "operand_h16",
    "operand_h32",
    "localcta_mxfp4_hybrid",
]

DISPLAY = {
    "bf16": "BF16",
    "te_native": "TE native NVFP4",
    "te_fol4": "TE FOL4",
    "pure_v5": "v5",
    "localcta": "localCTA",
    "mxfp4": "MXFP4",
    "localcta_h16": "localCTA + signed H16",
    "mxfp4_h32": "MXFP4 + signed H32",
    "operand_h16": "operand hybrid H16",
    "operand_h32": "operand hybrid H32",
    "localcta_mxfp4_hybrid": "layer hybrid localCTA/MXFP4",
}

COLORS = {
    "bf16": "#111111",
    "te_native": "#377eb8",
    "te_fol4": "#4daf4a",
    "pure_v5": "#984ea3",
    "localcta": "#ff7f00",
    "mxfp4": "#e41a1c",
    "localcta_h16": "#a65628",
    "mxfp4_h32": "#00a6a6",
    "operand_h16": "#f781bf",
    "operand_h32": "#17becf",
    "localcta_mxfp4_hybrid": "#999999",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def csv_bytes(fieldnames: list[str], rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LEDGER_COLUMNS:
            raise RuntimeError("validation ledger columns drift")
        for raw in reader:
            if raw["status"] != "complete":
                raise RuntimeError(f"incomplete validation cell: {raw}")
            if int(raw["scored_sequences"]) != 768 or int(raw["target_tokens"]) != 6_291_456:
                raise RuntimeError("validation geometry drift")
            rows.append(
                {
                    "semantic_route_key": raw["semantic_route_key"],
                    "route_label": raw["route_label"],
                    "exact_step": int(raw["exact_step"]),
                    "nll": float(raw["nll"]),
                    "sequence_level_standard_error": float(raw["sequence_level_standard_error"]),
                }
            )
    if len(rows) != 44:
        raise RuntimeError(f"expected 44 complete rows, got {len(rows)}")
    actual = {
        route: sorted(int(row["exact_step"]) for row in rows if row["semantic_route_key"] == route)
        for route in EXPECTED_STEPS
    }
    if actual != EXPECTED_STEPS:
        raise RuntimeError(f"route/step matrix drift: {actual!r}")
    return sorted(rows, key=lambda row: (ROUTE_ORDER.index(str(row["semantic_route_key"])), int(row["exact_step"])))


def enriched_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    bf16 = {
        int(row["exact_step"]): float(row["nll"])
        for row in rows
        if row["semantic_route_key"] == "bf16"
    }
    result: list[dict[str, object]] = []
    for row in rows:
        step = int(row["exact_step"])
        nll = float(row["nll"])
        reference = bf16.get(step)
        result.append(
            {
                "semantic_route_key": row["semantic_route_key"],
                "display_name": DISPLAY[str(row["semantic_route_key"])],
                "exact_step": step,
                "nll": f"{nll:.15f}",
                "sequence_level_standard_error": f"{float(row['sequence_level_standard_error']):.15f}",
                "bf16_nll_at_exact_step": "" if reference is None else f"{reference:.15f}",
                "nll_delta_from_bf16_percent": "" if reference is None else f"{100.0 * (nll / reference - 1.0):.9f}",
            }
        )
    return result


def latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


def render_table(rows: list[dict[str, object]]) -> bytes:
    lines = [
        "% Generated from the compact r22 validation ledger; do not edit by hand.",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Step & Validation NLL & $\\Delta$ vs. BF16 \\\\",
        "\\midrule",
    ]
    for row in rows:
        delta = str(row["nll_delta_from_bf16_percent"])
        delta_text = "--" if not delta else f"{float(delta):+.3f}\\%"
        lines.append(
            f"{latex_escape(str(row['display_name']))} & {int(row['exact_step']):,} & "
            f"{float(row['nll']):.4f} & {delta_text} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines).encode()


def render_plots(points: list[dict[str, object]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.2,
            "figure.dpi": 160,
            "savefig.dpi": 240,
        }
    )
    by_route = {
        route: [row for row in points if row["semantic_route_key"] == route]
        for route in ROUTE_ORDER
    }

    def draw(filename: str, minimum_step: int, title: str) -> None:
        fig, ax = plt.subplots(figsize=(7.25, 4.25), constrained_layout=True)
        for route in ROUTE_ORDER:
            records = [row for row in by_route[route] if int(row["exact_step"]) >= minimum_step]
            if not records:
                continue
            x = [int(row["exact_step"]) / 1000.0 for row in records]
            y = [float(row["nll"]) for row in records]
            linestyle = "-" if len(records) > 1 else "None"
            linewidth = 2.1 if route in {"bf16", "mxfp4_h32"} else 1.35
            ax.plot(
                x,
                y,
                color=COLORS[route],
                marker="o",
                markersize=4.2,
                linewidth=linewidth,
                linestyle=linestyle,
                label=DISPLAY[route],
                zorder=4 if route == "bf16" else 3,
            )
        ax.set_title(title)
        ax.set_xlabel("Training step (thousands)")
        ax.set_ylabel("Validation NLL (lower is better)")
        ax.grid(True, linewidth=0.45, alpha=0.28)
        ax.legend(ncol=2, frameon=False, loc="best")
        ax.text(
            0.995,
            0.01,
            "Fixed 6.291M-token stream; training-lineage scaled RoPE; checkpoint points only",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            color="#555555",
        )
        pdf = output / f"{filename}.pdf"
        png = output / f"{filename}.png"
        fig.savefig(pdf, metadata={"Creator": "MFU r23 renderer", "CreationDate": None, "ModDate": None})
        fig.savefig(png, metadata={"Software": "MFU r23 renderer"})
        plt.close(fig)

    draw("validation_curves_all", 0, "Checkpoint-by-checkpoint validation loss")
    draw("validation_curves_late_zoom", 10000, "Validation loss from step 10k")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    ledger_csv_payload = args.ledger_csv.read_bytes()

    rows = load_rows(args.ledger_csv)
    points = enriched_rows(rows)
    step38 = [row for row in points if int(row["exact_step"]) == 38000]
    latest = []
    for route in ROUTE_ORDER:
        candidates = [row for row in points if row["semantic_route_key"] == route]
        latest.append(max(candidates, key=lambda row: int(row["exact_step"])))

    args.output_dir.mkdir(parents=True)
    fields = list(points[0])
    outputs: dict[str, bytes] = {
        "VALIDATION_CURVE_POINTS.csv": csv_bytes(fields, points),
        "VALIDATION_LATEST_BY_ROUTE.csv": csv_bytes(fields, latest),
        "VALIDATION_STEP38000.csv": csv_bytes(fields, step38),
        "validation_table_step38000.tex": render_table(step38),
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_bytes(payload)
    render_plots(points, args.output_dir)

    generated = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_bytes(path.read_bytes())}
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema": "mfu_fixed_independent_validation_paper_inputs_r23_v1",
        "scientific_contract": {
            "claim": CLAIM,
            "checkpoint_points_only_no_smoothing": True,
            "scored_sequences": 768,
            "target_tokens": 6_291_456,
            "rope_semantics": "exact Llama-3.1 8B training-lineage scaled RoPE",
        },
        "source_ledger_sha256": sha256_bytes(ledger_csv_payload),
        "summary": {
            "complete_cells": 44,
            "routes": len(EXPECTED_STEPS),
            "step_38000_routes": len(step38),
        },
        "generated": generated,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_bytes(manifest))
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    (args.output_dir / "MANIFEST.json").write_bytes(manifest_payload)
    ledger = "".join(
        f"{sha256_bytes(path.read_bytes())}  {path.name}\n"
        for path in sorted(args.output_dir.iterdir())
        if path.is_file()
    ).encode()
    (args.output_dir / "SHA256SUMS").write_bytes(ledger)
    print(
        "R23_PAPER_INPUTS_PASS "
        f"cells=44 routes={len(EXPECTED_STEPS)} manifest={manifest['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
