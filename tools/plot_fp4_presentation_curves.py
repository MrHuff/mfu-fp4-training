#!/usr/bin/env python3
"""Generate presentation plots from the FP4 1.2B run logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RUNS = [
    "BF16 + BF16 CCE",
    "TE 2.13 NVFP4 original-style",
    "NVFP4 TK v5 no extras",
    "NVFP4 localCTA v4 no extras",
    "MXFP4 high-water",
]

DISPLAY_NAMES = {
    "BF16 + BF16 CCE": "BF16",
    "TE 2.13 NVFP4 original-style": "TE 2.13 original",
    "NVFP4 TK v5 no extras": "NVFP4 TK v5",
    "NVFP4 localCTA v4 no extras": "NVFP4 localCTA v4",
    "MXFP4 high-water": "MXFP4",
}

COLORS = {
    "BF16 + BF16 CCE": "#4c4c4c",
    "TE 2.13 NVFP4 original-style": "#b05a00",
    "NVFP4 TK v5 no extras": "#4b6cb7",
    "NVFP4 localCTA v4 no extras": "#14866d",
    "MXFP4 high-water": "#8b3fbf",
}

LINESTYLES = {
    "BF16 + BF16 CCE": "-",
    "TE 2.13 NVFP4 original-style": "--",
    "NVFP4 TK v5 no extras": "-",
    "NVFP4 localCTA v4 no extras": "-",
    "MXFP4 high-water": "-",
}

STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[-+0-9.eE]+)\s+"
    r"grad_norm:\s*(?P<grad_norm>[-+0-9.eE]+)\s+"
    r"memory:\s*(?P<memory_gib>[-+0-9.eE]+)GiB"
    r".*?tps:\s*(?P<tps>[0-9,]+)\s+"
    r"tflops:\s*(?P<tflops>[0-9,.]+)\s+"
    r"mfu:\s*(?P<mfu>[-+0-9.eE]+)%"
)
VALIDATION_RE = re.compile(
    r"validate step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[-+0-9.eE]+)\s+"
    r"memory:\s*(?P<memory_gib>[-+0-9.eE]+)GiB"
    r".*?tps:\s*(?P<tps>[0-9,]+)"
)
ANSI_RE = re.compile(r"\x1b\[[0-9;:]*m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default="docs/fp4_nvidia_1p2b_presentation_data_c4packed_loss_refresh_2026_05_29.csv",
    )
    parser.add_argument("--output-dir", default="docs/assets")
    parser.add_argument("--prefix", default="fp4_1p2b_training_curves_2026_06_02")
    parser.add_argument(
        "--runs",
        default=",".join(DEFAULT_RUNS),
        help="Comma-separated run names matching the presentation CSV.",
    )
    return parser.parse_args()


def parse_log(path: Path) -> list[dict[str, float]]:
    rows = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = STEP_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "grad_norm": float(match.group("grad_norm")),
                "memory_gib": float(match.group("memory_gib")),
                "tps": float(match.group("tps").replace(",", "")),
                "tflops": float(match.group("tflops").replace(",", "")),
                "mfu": float(match.group("mfu")),
            }
        )
    if not rows:
        raise ValueError(f"No step metrics found in {path}")
    return rows


def parse_validation_log(path: Path) -> list[dict[str, float]]:
    rows = []
    for raw_line in path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw_line)
        match = VALIDATION_RE.search(line)
        if not match:
            continue
        rows.append(
            {
                "step": int(match.group("step")),
                "loss": float(match.group("loss")),
                "memory_gib": float(match.group("memory_gib")),
                "tps": float(match.group("tps").replace(",", "")),
            }
        )
    return rows


def read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        return {row["run"]: row for row in csv.DictReader(handle)}


def write_curve_csv(path: Path, plotted: list[dict]) -> None:
    fields = [
        "run",
        "display_name",
        "source",
        "step",
        "loss",
        "grad_norm",
        "memory_gib",
        "tps",
        "tflops",
        "mfu",
        "log_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for bundle in plotted:
            for source_name, log_path, rows in (
                ("loss_refresh", bundle["loss_log"], bundle["loss_rows"]),
                ("compute", bundle["compute_log"], bundle["compute_rows"]),
            ):
                for row in rows:
                    writer.writerow(
                        {
                            "run": bundle["run"],
                            "display_name": bundle["display_name"],
                            "source": source_name,
                            "step": int(row["step"]),
                            "loss": row["loss"],
                            "grad_norm": row["grad_norm"],
                            "memory_gib": row["memory_gib"],
                            "tps": row["tps"],
                            "tflops": row["tflops"],
                            "mfu": row["mfu"],
                            "log_path": str(log_path),
                        }
                    )
            if bundle["validation_rows"]:
                for row in bundle["validation_rows"]:
                    writer.writerow(
                        {
                            "run": bundle["run"],
                            "display_name": bundle["display_name"],
                            "source": "validation",
                            "step": int(row["step"]),
                            "loss": row["loss"],
                            "grad_norm": "",
                            "memory_gib": row["memory_gib"],
                            "tps": row["tps"],
                            "tflops": "",
                            "mfu": "",
                            "log_path": str(bundle["validation_log"]),
                        }
                    )


def style_axes(ax, *, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#d8d8d8", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def save_plot(fig, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss(plotted: list[dict], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for bundle in plotted:
        rows = [row for row in bundle["loss_rows"] if row["step"] >= 10]
        ax.plot(
            [row["step"] for row in rows],
            [row["loss"] for row in rows],
            label=bundle["display_name"],
            color=COLORS[bundle["run"]],
            linestyle=LINESTYLES[bundle["run"]],
            linewidth=2.2,
        )
    style_axes(ax, ylabel="Training Loss", ylim=(2.55, 4.4))
    ax.set_title("NVIDIA 1.2B: 500-Step Training Loss", fontsize=14, pad=12)
    ax.text(
        0.0,
        -0.18,
        "Loss curves use packed-C4 refresh logs where available; step-1 warmup is omitted.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, ncol=2, fontsize=10)
    save_plot(fig, out_base)


def plot_mfu(plotted: list[dict], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for bundle in plotted:
        rows = [row for row in bundle["compute_rows"] if row["step"] >= 10]
        ax.plot(
            [row["step"] for row in rows],
            [row["mfu"] for row in rows],
            label=bundle["display_name"],
            color=COLORS[bundle["run"]],
            linestyle=LINESTYLES[bundle["run"]],
            marker="o",
            markersize=3.5,
            linewidth=2.0,
        )
    style_axes(ax, ylabel="MFU (%)", ylim=(40, 100))
    ax.set_title("NVIDIA 1.2B: 500-Step MFU", fontsize=14, pad=12)
    ax.text(
        0.0,
        -0.18,
        "MFU curves use original compute logs used by the slide tables; step-1 warmup is omitted.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, ncol=2, fontsize=10)
    save_plot(fig, out_base)


def has_validation(plotted: list[dict]) -> bool:
    return any(bundle["validation_rows"] for bundle in plotted)


def plot_validation_loss(plotted: list[dict], out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for bundle in plotted:
        rows = [row for row in bundle["validation_rows"] if row["step"] >= 1]
        if not rows:
            continue
        ax.plot(
            [row["step"] for row in rows],
            [row["loss"] for row in rows],
            label=bundle["display_name"],
            color=COLORS[bundle["run"]],
            linestyle=LINESTYLES[bundle["run"]],
            marker="o",
            markersize=3.5,
            linewidth=2.0,
        )
    style_axes(ax, ylabel="Validation Loss")
    ax.set_title("NVIDIA 1.2B: Validation Loss", fontsize=14, pad=12)
    ax.text(
        0.0,
        -0.18,
        "Validation points use Torchtitan validation logs from fresh validation-enabled runs.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    ax.legend(frameon=False, ncol=2, fontsize=10)
    save_plot(fig, out_base)


def plot_combined(plotted: list[dict], out_base: Path) -> None:
    fig, (loss_ax, mfu_ax) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for bundle in plotted:
        loss_rows = [row for row in bundle["loss_rows"] if row["step"] >= 10]
        mfu_rows = [row for row in bundle["compute_rows"] if row["step"] >= 10]
        loss_ax.plot(
            [row["step"] for row in loss_rows],
            [row["loss"] for row in loss_rows],
            label=bundle["display_name"],
            color=COLORS[bundle["run"]],
            linestyle=LINESTYLES[bundle["run"]],
            linewidth=2.0,
        )
        mfu_ax.plot(
            [row["step"] for row in mfu_rows],
            [row["mfu"] for row in mfu_rows],
            label=bundle["display_name"],
            color=COLORS[bundle["run"]],
            linestyle=LINESTYLES[bundle["run"]],
            marker="o",
            markersize=3.2,
            linewidth=2.0,
        )
    style_axes(loss_ax, ylabel="Training Loss", ylim=(2.55, 4.4))
    style_axes(mfu_ax, ylabel="MFU (%)", ylim=(40, 100))
    loss_ax.set_title("Loss", fontsize=13, pad=10)
    mfu_ax.set_title("MFU", fontsize=13, pad=10)
    handles, labels = mfu_ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("NVIDIA 1.2B: 500-Step Training Curves", fontsize=15, y=1.02)
    fig.text(
        0.5,
        -0.095,
        "Loss uses packed-C4 refresh logs; MFU uses original compute logs; step-1 warmup is omitted.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    save_plot(fig, out_base)


def plot_combined_with_validation(plotted: list[dict], out_base: Path) -> None:
    fig, (loss_ax, val_ax, mfu_ax) = plt.subplots(1, 3, figsize=(17.2, 5.0))
    for bundle in plotted:
        loss_rows = [row for row in bundle["loss_rows"] if row["step"] >= 10]
        val_rows = [row for row in bundle["validation_rows"] if row["step"] >= 1]
        mfu_rows = [row for row in bundle["compute_rows"] if row["step"] >= 10]
        loss_ax.plot(
            [row["step"] for row in loss_rows],
            [row["loss"] for row in loss_rows],
            label=bundle["display_name"],
            color=COLORS[bundle["run"]],
            linestyle=LINESTYLES[bundle["run"]],
            linewidth=2.0,
        )
        if val_rows:
            val_ax.plot(
                [row["step"] for row in val_rows],
                [row["loss"] for row in val_rows],
                label=bundle["display_name"],
                color=COLORS[bundle["run"]],
                linestyle=LINESTYLES[bundle["run"]],
                marker="o",
                markersize=3.2,
                linewidth=2.0,
            )
        mfu_ax.plot(
            [row["step"] for row in mfu_rows],
            [row["mfu"] for row in mfu_rows],
            label=bundle["display_name"],
            color=COLORS[bundle["run"]],
            linestyle=LINESTYLES[bundle["run"]],
            marker="o",
            markersize=3.2,
            linewidth=2.0,
        )
    style_axes(loss_ax, ylabel="Training Loss", ylim=(2.55, 4.4))
    style_axes(val_ax, ylabel="Validation Loss")
    style_axes(mfu_ax, ylabel="MFU (%)", ylim=(40, 100))
    loss_ax.set_title("Train Loss", fontsize=13, pad=10)
    val_ax.set_title("Validation Loss", fontsize=13, pad=10)
    mfu_ax.set_title("MFU", fontsize=13, pad=10)
    handles, labels = mfu_ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("NVIDIA 1.2B: 500-Step Curves with Validation", fontsize=15, y=1.02)
    fig.text(
        0.5,
        -0.095,
        "Training loss and validation loss use validation-enabled logs when available; MFU uses the same compute source.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    save_plot(fig, out_base)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = read_csv(input_csv)
    selected_runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    plotted = []
    for run in selected_runs:
        if run not in csv_rows:
            raise KeyError(f"Run {run!r} not found in {input_csv}")
        row = csv_rows[run]
        compute_log = Path(row["compute_log"])
        loss_log = Path(row.get("loss_refresh_log") or row["compute_log"])
        validation_log_text = row.get("validation_log") or ""
        validation_log = Path(validation_log_text) if validation_log_text else loss_log
        if not compute_log.exists():
            raise FileNotFoundError(compute_log)
        if not loss_log.exists():
            raise FileNotFoundError(loss_log)
        if not validation_log.exists():
            raise FileNotFoundError(validation_log)
        plotted.append(
            {
                "run": run,
                "display_name": DISPLAY_NAMES.get(run, run),
                "compute_log": compute_log,
                "loss_log": loss_log,
                "validation_log": validation_log,
                "compute_rows": parse_log(compute_log),
                "loss_rows": parse_log(loss_log),
                "validation_rows": parse_validation_log(validation_log),
            }
        )

    csv_out = output_dir / f"{args.prefix}.csv"
    write_curve_csv(csv_out, plotted)
    plot_loss(plotted, output_dir / f"{args.prefix}_loss")
    plot_mfu(plotted, output_dir / f"{args.prefix}_mfu")
    plot_combined(plotted, output_dir / f"{args.prefix}_combined")
    if has_validation(plotted):
        plot_validation_loss(plotted, output_dir / f"{args.prefix}_validation_loss")
        plot_combined_with_validation(
            plotted,
            output_dir / f"{args.prefix}_combined_with_validation",
        )

    print(f"Wrote curve CSV: {csv_out}")
    print(f"Wrote loss plot: {output_dir / (args.prefix + '_loss.svg')}")
    print(f"Wrote MFU plot: {output_dir / (args.prefix + '_mfu.svg')}")
    print(f"Wrote combined plot: {output_dir / (args.prefix + '_combined.svg')}")
    if has_validation(plotted):
        print(
            f"Wrote validation plot: {output_dir / (args.prefix + '_validation_loss.svg')}"
        )
        print(
            "Wrote validation combined plot: "
            f"{output_dir / (args.prefix + '_combined_with_validation.svg')}"
        )
    else:
        print("No validation rows found; validation plots were not generated.")


if __name__ == "__main__":
    main()
