#!/usr/bin/env python3
"""Plot matched BF16-evaluated trajectories for the FP4 P-cache gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Series:
    name: str
    result: dict


def _named_path(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected NAME=/path/to/result.json")
    return name.strip(), Path(path)


def _load_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"Expected one matrix result in {path}")
    result = payload[0]
    if result.get("status") != "OK":
        raise ValueError(f"Result did not complete successfully: {path}")
    if not result.get("eval_bf16_losses"):
        raise ValueError(f"Result has no common BF16 evaluations: {path}")
    return result


def _style_axis(axis: plt.Axes, ylabel: str) -> None:
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#d4d4d8", linewidth=0.8, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    svg_path = output_base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
        + "\n"
    )


def plot(
    control: Series,
    candidates: list[Series],
    output_base: Path,
    eval_every: int,
) -> None:
    control_losses = [
        float(value) for value in control.result["eval_bf16_losses"]
    ]
    steps = [
        eval_every * index for index in range(1, len(control_losses) + 1)
    ]
    colors = ("#2563eb", "#16835f", "#b45309", "#7c3aed", "#be123c")

    fig, (loss_axis, delta_axis) = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.2, 0.8)},
    )
    loss_axis.plot(
        steps,
        control_losses,
        color="#27272a",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label=control.name,
    )

    for index, series in enumerate(candidates):
        losses = [
            float(value) for value in series.result["eval_bf16_losses"]
        ]
        if len(losses) != len(control_losses):
            raise ValueError(
                f"{series.name} has {len(losses)} evaluations; "
                f"expected {len(control_losses)}"
            )
        color = colors[index % len(colors)]
        deltas = [
            (candidate - reference) / reference * 100.0
            for candidate, reference in zip(
                losses, control_losses, strict=True
            )
        ]
        loss_axis.plot(
            steps,
            losses,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label=series.name,
        )
        delta_axis.plot(
            steps,
            deltas,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label=series.name,
        )

    _style_axis(loss_axis, "Native BF16 evaluation loss")
    loss_axis.set_title(
        "Llama 8B final-layer FP4 cache: matched 1,000-step gate",
        fontsize=14,
        pad=12,
    )
    loss_axis.legend(frameon=False, fontsize=8.5, ncol=2)

    delta_axis.axhspan(-2.0, 2.0, color="#16a34a", alpha=0.08)
    delta_axis.axhline(0.0, color="#52525b", linewidth=1.1)
    delta_axis.axhline(2.0, color="#16a34a", linewidth=0.9, linestyle=":")
    delta_axis.axhline(-2.0, color="#16a34a", linewidth=0.9, linestyle=":")
    _style_axis(delta_axis, "Delta vs BF16 control (%)")
    delta_axis.set_xlabel("Training step")
    delta_axis.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.subplots_adjust(hspace=0.12)
    _save(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=_named_path, required=True)
    parser.add_argument(
        "--candidate",
        type=_named_path,
        action="append",
        default=[],
    )
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.candidate:
        parser.error("at least one --candidate is required")
    if args.eval_every < 1:
        parser.error("--eval-every must be positive")

    control_name, control_path = args.control
    control = Series(control_name, _load_result(control_path))
    candidates = [
        Series(name, _load_result(path)) for name, path in args.candidate
    ]
    plot(control, candidates, args.output, args.eval_every)


if __name__ == "__main__":
    main()
