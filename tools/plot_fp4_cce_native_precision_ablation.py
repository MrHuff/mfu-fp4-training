#!/usr/bin/env python3
"""Plot the native CUDA/TK final-layer CCE precision ablation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = (
    "native-bf16-fwd-bf16-bwd",
    "native-fp4-fwd-bf16-bwd",
    "native-bf16-fwd-fp4-bwd",
    "native-fp4-fwd-fp4-bwd",
)

DISPLAY_NAMES = {
    "native-bf16-fwd-bf16-bwd": "BF16 fwd / BF16 bwd",
    "native-fp4-fwd-bf16-bwd": "FP4 fwd / BF16 bwd",
    "native-bf16-fwd-fp4-bwd": "BF16 fwd / FP4 bwd",
    "native-fp4-fwd-fp4-bwd": "FP4 fwd / FP4 bwd",
}

COLORS = {
    "native-bf16-fwd-bf16-bwd": "#3f3f46",
    "native-fp4-fwd-bf16-bwd": "#2563eb",
    "native-bf16-fwd-fp4-bwd": "#16835f",
    "native-fp4-fwd-fp4-bwd": "#b45309",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing FP4_CCE_TRAIN_MATRIX_train1000-*.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets"),
    )
    parser.add_argument(
        "--prefix",
        default="fp4_cce_native_precision_2026_07_25",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Trailing-window size for the displayed medians.",
    )
    parser.add_argument(
        "--common-eval-every",
        type=int,
        default=50,
        help="Step interval used by the common-BF16 evaluation run.",
    )
    return parser.parse_args()


def load_results(input_dir: Path) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for path in sorted(input_dir.glob("FP4_CCE_TRAIN_MATRIX_train1000-*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"Expected a result list in {path}")
        for result in payload:
            label = result.get("label")
            if label not in VARIANTS:
                continue
            if label in results:
                raise ValueError(f"Duplicate result for {label}: {path}")
            if result.get("status") != "OK":
                raise ValueError(f"{label} did not complete successfully")
            if not result.get("rows"):
                raise ValueError(f"{label} has no step metrics")
            results[label] = result

    missing = [label for label in VARIANTS if label not in results]
    if missing:
        raise ValueError(f"Missing result variants: {', '.join(missing)}")
    return results


def trailing_median(values: list[float], window: int) -> list[float]:
    if window < 1:
        raise ValueError("--smooth-window must be positive")
    return [
        statistics.median(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def style_axes(ax: plt.Axes, *, ylabel: str) -> None:
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def save_plot(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    svg_path = out_base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    svg_lines = svg_path.read_text().splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n")


def plot_training_health(
    results: dict[str, dict],
    out_base: Path,
    *,
    smooth_window: int,
) -> None:
    fig, (loss_ax, grad_ax) = plt.subplots(
        2,
        1,
        figsize=(10.4, 8.4),
        sharex=True,
        gridspec_kw={"height_ratios": (1.05, 1.0)},
    )

    for label in VARIANTS:
        rows = results[label]["rows"]
        steps = [int(row["step"]) for row in rows]
        losses = [float(row["loss"]) for row in rows]
        grad_norms = [float(row["grad_norm"]) for row in rows]
        color = COLORS[label]

        loss_ax.plot(steps, losses, color=color, alpha=0.13, linewidth=0.7)
        loss_ax.plot(
            steps,
            trailing_median(losses, smooth_window),
            color=color,
            linewidth=2.0,
            label=DISPLAY_NAMES[label],
        )
        grad_ax.scatter(
            steps,
            grad_norms,
            color=color,
            alpha=0.12,
            edgecolors="none",
            s=6,
        )
        grad_ax.plot(
            steps,
            trailing_median(grad_norms, smooth_window),
            color=color,
            linewidth=2.0,
        )

    style_axes(loss_ax, ylabel="Training loss")
    style_axes(grad_ax, ylabel="Gradient norm")
    grad_ax.set_yscale("log")
    grad_ax.set_xlabel("Training step")
    loss_ax.legend(frameon=False, ncol=2, fontsize=9)
    loss_ax.set_title(
        "Native CUDA/TK CCE precision matrix: 1,000-step health check",
        fontsize=14,
        pad=12,
    )
    fig.text(
        0.5,
        0.012,
        (
            f"Faint marks are per-step values; solid lines are trailing "
            f"{smooth_window}-step medians. Runs used separate C4 streams and "
            "GPUs, so curves are not a convergence or throughput ranking."
        ),
        ha="center",
        fontsize=8.5,
        color="#52525b",
    )
    fig.subplots_adjust(bottom=0.1, hspace=0.12)
    save_plot(fig, out_base)


def plot_common_bf16_eval(
    result: dict,
    out_base: Path,
    *,
    eval_every: int,
) -> None:
    eval_losses = [float(value) for value in result.get("eval_bf16_losses", ())]
    if not eval_losses:
        raise ValueError("FP4/FP4 result has no common-BF16 evaluation metrics")

    rows_by_step = {int(row["step"]): row for row in result["rows"]}
    steps = [eval_every * index for index in range(1, len(eval_losses) + 1)]
    missing_steps = [step for step in steps if step not in rows_by_step]
    if missing_steps:
        raise ValueError(f"Missing training rows for eval steps: {missing_steps}")
    fp4_losses = [float(rows_by_step[step]["loss"]) for step in steps]
    relative_deltas = [
        abs(fp4_loss - bf16_loss) / bf16_loss * 100.0
        for fp4_loss, bf16_loss in zip(fp4_losses, eval_losses, strict=True)
    ]

    fig, (loss_ax, delta_ax) = plt.subplots(
        2,
        1,
        figsize=(10.4, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": (1.25, 0.75)},
    )
    loss_ax.plot(
        steps,
        fp4_losses,
        color=COLORS["native-fp4-fwd-fp4-bwd"],
        marker="o",
        markersize=4,
        linewidth=2.0,
        label="FP4 forward training loss",
    )
    loss_ax.plot(
        steps,
        eval_losses,
        color="#2563eb",
        marker="s",
        markersize=3.6,
        linewidth=1.8,
        linestyle="--",
        label="Same weights/batch, BF16 forward",
    )
    style_axes(loss_ax, ylabel="Loss")
    loss_ax.legend(frameon=False, fontsize=9)
    loss_ax.set_title(
        "FP4/FP4 cell: common BF16 forward evaluation",
        fontsize=14,
        pad=12,
    )

    delta_ax.plot(
        steps,
        relative_deltas,
        color="#7c3aed",
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    delta_ax.axhline(
        statistics.mean(relative_deltas),
        color="#52525b",
        linewidth=1.2,
        linestyle=":",
        label=f"Mean: {statistics.mean(relative_deltas):.3f}%",
    )
    style_axes(delta_ax, ylabel="Absolute relative delta (%)")
    delta_ax.set_xlabel("Training step")
    delta_ax.legend(frameon=False, fontsize=9)
    delta_ax.text(
        0.99,
        0.93,
        (
            f"max {max(relative_deltas):.3f}% | "
            f"step 1000 {relative_deltas[-1]:.3f}%"
        ),
        transform=delta_ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#52525b",
    )
    fig.subplots_adjust(hspace=0.14)
    save_plot(fig, out_base)


def main() -> int:
    args = parse_args()
    results = load_results(args.input_dir.resolve())
    output_dir = args.output_dir.resolve()
    plot_training_health(
        results,
        output_dir / f"{args.prefix}_training_health",
        smooth_window=args.smooth_window,
    )
    plot_common_bf16_eval(
        results["native-fp4-fwd-fp4-bwd"],
        output_dir / f"{args.prefix}_fp4_common_bf16_eval",
        eval_every=args.common_eval_every,
    )
    print(f"Wrote plots under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
