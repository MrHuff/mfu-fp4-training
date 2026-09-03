#!/usr/bin/env python3
"""Render the QKV/FFN producer--consumer fusion schedule.

The figure is deliberately conceptual: solid boxes show operations that share
one producer--consumer boundary, while dashed arrows show stream overlap.  It
does not imply that every solid box is one monolithic CUDA kernel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "fp4_qkv_ffn_fusion_schedule.pdf"

INK = "#252A34"
MUTED = "#646B76"
SERIAL = "#ECEFF2"
SERIAL_EDGE = "#9AA1AA"
HBM = "#C84C4C"
PRODUCER = "#2A9D8F"
GEMM = "#2878B5"
EPILOGUE = "#E58B3D"
REDUCTION = "#7656A5"
BOUNDARY = "#EEF7F4"
COLLECTIVE = "#FFFFFF"


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    *,
    face: str,
    edge: str | None = None,
    text_color: str = "white",
    fontsize: float = 6.65,
    linestyle: str = "-",
    linewidth: float = 0.8,
    zorder: int = 3,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            facecolor=face,
            edgecolor=edge or face,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        linespacing=1.05,
        zorder=zorder + 1,
    )


def _arrow(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    *,
    dashed: bool = False,
    color: str = INK,
    rad: float = 0.0,
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=7.2,
            linewidth=0.85,
            linestyle=(0, (3, 2)) if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        )
    )


def _boundary(ax: plt.Axes) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.115, 0.095),
            0.865,
            0.315,
            boxstyle="round,pad=0.010,rounding_size=0.018",
            facecolor=BOUNDARY,
            edgecolor="#AED4C8",
            linewidth=0.8,
            zorder=0,
        )
    )


def _lane_labels(ax: plt.Axes, panel: str) -> None:
    ax.text(
        0.002,
        0.965,
        panel,
        ha="left",
        va="top",
        fontsize=8.9,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.101,
        0.666,
        "Serial",
        ha="right",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=MUTED,
    )
    ax.text(
        0.101,
        0.250,
        "Format-\naware",
        ha="right",
        va="center",
        fontsize=6.8,
        fontweight="bold",
        color=PRODUCER,
        linespacing=0.95,
    )


def _serial_pipeline(ax: plt.Axes, labels: list[str]) -> None:
    start, end = 0.12, 0.98
    gap = 0.012
    width = (end - start - gap * (len(labels) - 1)) / len(labels)
    y, h = 0.575, 0.185
    for i, label in enumerate(labels):
        x = start + i * (width + gap)
        is_hbm = label.startswith("HBM")
        _box(
            ax,
            x,
            y,
            width,
            h,
            label,
            face="#FBECEC" if is_hbm else SERIAL,
            edge=HBM if is_hbm else SERIAL_EDGE,
            text_color=HBM if is_hbm else INK,
            fontsize=5.75 if len(labels) >= 7 else 6.15,
            linewidth=0.75,
        )
        if i:
            previous_right = x - gap
            _arrow(ax, previous_right, x, y + h / 2, color=SERIAL_EDGE)


def _fused_pipeline(
    ax: plt.Axes,
    stages: list[tuple[float, str, str]],
    *,
    overlap: str | None = None,
) -> None:
    _boundary(ax)
    start, end = 0.125, 0.970
    gap = 0.014
    available = end - start - gap * (len(stages) - 1)
    total_weight = sum(weight for weight, _, _ in stages)
    x = start
    centers: list[float] = []
    for index, (weight, label, color) in enumerate(stages):
        width = available * weight / total_weight
        is_collective = color == COLLECTIVE
        _box(
            ax,
            x,
            0.155,
            width,
            0.195,
            label,
            face=color,
            edge=REDUCTION if is_collective else None,
            text_color=INK if is_collective else "white",
            fontsize=6.25 if len(stages) >= 5 else 6.55,
            linestyle=(0, (3, 2)) if is_collective else "-",
        )
        centers.append(x + width / 2)
        if index:
            _arrow(ax, x - gap, x, 0.2525)
        x += width + gap

    if overlap:
        left = centers[0]
        right = centers[-1]
        ax.add_patch(
            FancyArrowPatch(
                (left, 0.385),
                (right, 0.385),
                arrowstyle="-|>",
                mutation_scale=7.0,
                linewidth=0.85,
                linestyle=(0, (3, 2)),
                color=REDUCTION,
                connectionstyle="arc3,rad=-0.06",
                zorder=4,
            )
        )
        ax.text(
            (left + right) / 2,
            0.455,
            overlap,
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=REDUCTION,
        )


def _panel_qkv_forward(ax: plt.Axes) -> None:
    _lane_labels(ax, "A   QKV forward")
    _serial_pipeline(
        ax,
        [
            "RMSNorm",
            "HBM\nwrite/read",
            "amax + scales\n+ two packs",
            "Q, K, V\nGEMMs",
            "HBM\nwrite/read",
            "layout +\nQ/K RoPE",
        ],
    )
    _fused_pipeline(
        ax,
        [
            (2.3, "RMSNorm + route scales\n+ row/column FP4 views\n+ saved row statistics", PRODUCER),
            (1.6, "grouped Q/K/V\nFP4 GEMMs", GEMM),
            (1.45, "Q/K RoPE + layout\nepilogue", EPILOGUE),
            (0.85, "attention", GEMM),
        ],
        overlap="route-gated overlap: weight-view production  ||  activation producer",
    )


def _panel_qkv_backward(ax: plt.Axes) -> None:
    _lane_labels(ax, "B   QKV backward")
    _serial_pipeline(
        ax,
        [
            "undo layout\n+ RoPE",
            "HBM\nwrite/read",
            "row-SR\ndY pack",
            "column dY/X\npack (+ H)",
            "Dgrad",
            "Wgrad",
            "RMSNorm bwd\n+ grad reduce",
        ],
    )
    _fused_pipeline(
        ax,
        [
            (2.55, "inverse layout/RoPE + one-read producer\nrow-SR dY for Dgrad; column X,dY\nwith optional paired fixed H for Wgrad", PRODUCER),
            (1.20, "grouped\nDgrad", GEMM),
            (1.20, "grouped\nWgrad", GEMM),
            (1.55, "gradient sum +\nRMSNorm backward", REDUCTION),
            (1.05, "FSDP gradient\nreduce-scatter", COLLECTIVE),
        ],
        overlap="route-gated overlap: Wgrad  ||  Dgrad / RMS reduction",
    )
def _panel_ffn_forward(ax: plt.Axes) -> None:
    _lane_labels(ax, "C   FFN forward")
    _serial_pipeline(
        ax,
        [
            "RMSNorm +\nFP4 pack",
            "W1 GEMM",
            "W3 GEMM",
            "HBM\nwrite/read",
            "SiLU $\\times$ gate",
            "HBM +\nFP4 repack",
            "W2 GEMM",
        ],
    )
    _fused_pipeline(
        ax,
        [
            (2.0, "RMSNorm + route scales\n+ both FP4 views", PRODUCER),
            (1.65, "paired W1/W3\nFP4 GEMMs", GEMM),
            (2.0, "SiLU $\\times$ gate +\nW2 scale/pack producer", PRODUCER),
            (1.25, "W2 FP4\nGEMM", GEMM),
        ],
        overlap="route-gated overlap: weight-view production  ||  activation producer",
    )


def _panel_ffn_backward(ax: plt.Axes) -> None:
    _lane_labels(ax, "D   FFN backward")
    _serial_pipeline(
        ax,
        [
            "dY packs",
            "W2 Dgrad\n+ Wgrad",
            "HBM\nwrite/read",
            "two SiLU\nderivatives",
            "HBM + two\nFP4 packs",
            "W1/W3 Dgrad\n+ Wgrad",
            "sum + RMS\nbwd + reduce",
        ],
    )
    _fused_pipeline(
        ax,
        [
            (1.75, "dY producer\nrow SR + Wgrad view", PRODUCER),
            (1.40, "W2 Dgrad\n+ Wgrad", GEMM),
            (2.1, "SiLU derivative + two branch\nFP4 views in one producer", PRODUCER),
            (1.65, "grouped W1/W3\nDgrad + Wgrad", GEMM),
            (1.35, "sum + RMSNorm\nbackward", REDUCTION),
        ],
        overlap="route-gated overlap: W2 Wgrad  ||  derivative producer; backward weight prefetch starts afterward",
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "text.color": INK,
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        }
    )
    fig, axes = plt.subplots(4, 1, figsize=(7.25, 10.1))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    _panel_qkv_forward(axes[0])
    _panel_qkv_backward(axes[1])
    _panel_ffn_forward(axes[2])
    _panel_ffn_backward(axes[3])

    fig.suptitle(
        "Fusing the repeated producer--consumer boundaries of a Llama block",
        x=0.50,
        y=0.995,
        fontsize=11.2,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.50,
        0.025,
        "Scale contract inside teal producers: MXFP4 completes local amax$_{32}$; global NVFP4 starts tensor amax early; "
        "CTA-local NVFP4 carries tile outer scales to the GEMM, whose epilogue applies $\\alpha_i\\beta_j$.\n"
        "Solid arrows are dependencies. Purple dashed arrows denote overlap, not kernel fusion. HBM boxes mark avoidable materialization.",
        ha="center",
        va="bottom",
        fontsize=6.25,
        color=MUTED,
        linespacing=1.22,
    )
    fig.subplots_adjust(left=0.035, right=0.985, top=0.970, bottom=0.085, hspace=0.20)
    fig.savefig(
        OUTPUT,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Title": "QKV and FFN FP4 fusion schedule",
            "Author": "Robert Hu",
            "Subject": "Format-aware fusion in Llama training",
            "Creator": "Matplotlib",
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
