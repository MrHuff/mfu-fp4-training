#!/usr/bin/env python3
"""Generate the evidence and dataflow figures used by the FP4 report."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
PINK = "#F56F79"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
GREEN = "#63A65F"
ORANGE = "#E58B3D"
RED = "#C94C4C"
DARK = "#30343B"
MID = "#7A8088"
LIGHT = "#E9EBEE"
PURPLE = "#7656A5"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": DARK,
            "axes.edgecolor": MID,
            "xtick.color": DARK,
            "ytick.color": DARK,
            "text.color": DARK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def make_1p2b_mfu() -> None:
    # Committed evidence: docs/fp4_nvidia_1p2b_presentation_2026_05_28.md.
    labels = [
        "BF16",
        "TE original\nFP4 recipe",
        "TE all-block\nNVFP4",
        "TK global\nNVFP4",
        "CTA-local\nNVFP4",
        "Fused\nMXFP4",
    ]
    mfu = [46.67, 46.29, 66.20, 84.37, 87.49, 94.75]
    colors = [MID, BLUE, BLUE, PINK, TEAL, GREEN]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bars = ax.bar(range(len(labels)), mfu, color=colors, width=0.72)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Steady BF16 MFU (%)")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_title("1.2B model: FP4 speed appears only after full-path optimization")
    ax.grid(axis="y", color=LIGHT, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, mfu):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.annotate(
        "The original-style TE recipe\nmeasured the same as BF16",
        xy=(1, mfu[1]),
        xytext=(1.65, 25),
        arrowprops={"arrowstyle": "->", "color": DARK, "lw": 1.2},
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.995,
        -0.22,
        "500 steps; steady window begins at step 50. Different FP4 rows "
        "also use different numerical recipes.",
        transform=ax.transAxes,
        ha="right",
        fontsize=8.5,
        color=MID,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fp4_1p2b_steady_mfu.png", dpi=220)
    plt.close(fig)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    color: str,
    *,
    text_color: str = "white",
    font_size: float = 9.5,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.015",
        facecolor=color,
        edgecolor="none",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        color=text_color,
        fontsize=font_size,
        fontweight="bold",
    )


def _arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.3,
            color=DARK,
        )
    )


def make_fusion_dataflow() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 5.4))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    axes[0].set_title("Unfused boundary: each small operation revisits GPU memory")
    top_boxes = [
        (0.02, "GEMM", BLUE),
        (0.19, "write\nBF16", MID),
        (0.36, "RMSNorm", PINK),
        (0.53, "amax +\nscales", PINK),
        (0.70, "row/column\nquantize", TEAL),
        (0.87, "next\nGEMM", BLUE),
    ]
    for x, text, color in top_boxes:
        _box(axes[0], x, 0.34, 0.11, 0.34, text, color)
    for left, right in zip(top_boxes, top_boxes[1:]):
        _arrow(axes[0], left[0] + 0.11, 0.51, right[0], 0.51)
    axes[0].text(
        0.5,
        0.12,
        "Extra launches, reads, writes, scale reductions, and synchronization",
        ha="center",
        color=MID,
        fontsize=10,
    )

    axes[1].set_title(
        "Format-aware fused boundary: finish useful work while data is on chip"
    )
    _box(axes[1], 0.05, 0.29, 0.18, 0.40, "GEMM\naccumulator", BLUE)
    _box(
        axes[1],
        0.34,
        0.22,
        0.32,
        0.54,
        "CUDA/TK producer\nresidual + RMS + scales\nrow/column FP4 payloads",
        PINK,
    )
    _box(axes[1], 0.77, 0.29, 0.18, 0.40, "next\nFP4 GEMM", GREEN)
    _arrow(axes[1], 0.23, 0.49, 0.34, 0.49)
    _arrow(axes[1], 0.66, 0.49, 0.77, 0.49)
    axes[1].text(
        0.5,
        0.08,
        "The exact contents differ for MXFP4, global NVFP4, and CTA-local NVFP4",
        ha="center",
        color=MID,
        fontsize=10,
    )

    fig.tight_layout(h_pad=1.2)
    fig.savefig(OUTPUT_DIR / "fp4_fusion_dataflow.png", dpi=220)
    plt.close(fig)


def make_format_execution_routes() -> None:
    """Show why the three FP4 formats require different producer schedules."""
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 7.0))
    rows = [
        (
            "MXFP4-v4",
            [
                (0.03, 0.16, "BF16 producer\nRMSNorm / SwiGLU", MID),
                (0.29, 0.28, "fused local amax$_{32}$\nE8M0 + E2M1 pack\nrow and column views", BLUE),
                (0.67, 0.16, "native\nMXFP4 GEMM", GREEN),
            ],
            "Power-of-two scales are local to 32 values: quantization can finish as tiles are produced.",
        ),
        (
            "Global NVFP4 v5",
            [
                (0.03, 0.16, "BF16 producer\nstarts amax early", MID),
                (0.29, 0.18, "tensor amax\ncompletion", RED),
                (0.54, 0.24, "E4M3 scale$_{16}$\n+ E2M1 pack\n+ scale swizzle", PINK),
                (0.85, 0.12, "NVFP4\nGEMM", GREEN),
            ],
            "The finer scale grid adds a tensor-wide completion dependency; overlap matters as much as packing speed.",
        ),
        (
            "CTA-local NVFP4-v4",
            [
                (0.03, 0.18, "BF16 tile\nproducer", MID),
                (0.29, 0.30, "fused tile outer scale\n+ E4M3 scale$_{16}$\n+ row/column payloads", TEAL),
                (0.69, 0.20, "NVFP4 GEMM\nfull K reduction\n$\\times\\,\\alpha_i\\beta_j$ once", GREEN),
            ],
            "No global amax barrier: two K-invariant FP32 outer scales travel with the tile to the GEMM epilogue.",
        ),
    ]
    for ax, (title, boxes, note) in zip(axes, rows):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(0.01, 0.94, title, fontsize=11, fontweight="bold", va="top")
        previous = None
        for x, width, label, color in boxes:
            _box(ax, x, 0.35, width, 0.40, label, color)
            if previous is not None:
                _arrow(ax, previous, 0.55, x, 0.55)
            previous = x + width
        ax.text(0.50, 0.12, note, ha="center", color=MID, fontsize=9.2)
    fig.suptitle(
        "The datatype determines the fusion and synchronization boundary",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(h_pad=0.55)
    fig.savefig(OUTPUT_DIR / "fp4_format_execution_routes.png", dpi=220)
    plt.close(fig)


def make_linear_operand_map() -> None:
    """Map quantized operands to Fprop, Dgrad, and Wgrad."""
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8))
    top, bottom = axes
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    top.set_title("One linear layer creates three different FP4 contractions")
    columns = [
        (0.02, "Fprop", "$\\widehat{Y}=Q_r(X)Q_r(W)^{\\mathsf{T}}$", "row-facing activation and weight"),
        (0.345, "Dgrad", "$\\widehat{dX}=Q_r(dY)Q_c(W)$", "row $dY$; column-facing weight"),
        (0.67, "Wgrad", "$\\widehat{dW}=Q_c(dY)^{\\mathsf{T}}Q_c(X)$", "column-facing gradient and activation"),
    ]
    for x, title, equation, note in columns:
        _box(top, x, 0.48, 0.29, 0.30, title + "\n" + equation, BLUE)
        top.text(x + 0.145, 0.33, note, ha="center", fontsize=9.2, color=MID)
    top.text(
        0.50,
        0.10,
        "The same BF16 tensor needs row- and column-facing scale/layout views for different contractions.",
        ha="center",
        fontsize=9.4,
        color=DARK,
    )

    bottom.set_title("Operand hybrid: assign a format to each contraction")
    hybrid = [
        (0.02, "Fprop", "MXFP4-v4\n2D weight; untransformed", BLUE),
        (0.345, "Dgrad", "CTA-local NVFP4-v4\nrow-$dY$ data SR", TEAL),
        (0.67, "Wgrad", "MXFP4-v4\npaired plain H16", PURPLE),
    ]
    for x, title, body, color in hybrid:
        _box(bottom, x, 0.46, 0.29, 0.32, title + "\n" + body, color)
    bottom.text(
        0.50,
        0.24,
        "A fused backward producer reads each BF16 gradient tile once and emits the MX column-H16 and CTA-local row-SR carriers.",
        ha="center",
        fontsize=9.3,
        color=DARK,
    )
    bottom.text(
        0.50,
        0.09,
        "This mixes operands inside every linear layer and also revises the pure run's fixed-sign H32 Wgrad transform to plain H16.",
        ha="center",
        fontsize=9.3,
        color=MID,
    )
    fig.tight_layout(h_pad=1.0)
    fig.savefig(OUTPUT_DIR / "fp4_linear_operand_hybrid.png", dpi=220)
    plt.close(fig)


def make_2d_weight_contract() -> None:
    """Contrast independent 1D quantizations with an orientation-consistent 2D weight."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.8))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    left, right = axes
    left.set_title("Independent 1D weight quantization")
    _box(left, 0.23, 0.79, 0.54, 0.12, "one BF16 weight $W$", MID, font_size=8.8)
    _box(left, 0.03, 0.48, 0.44, 0.17, "$Q_K(W)=W_{\\rm fwd}$\nforward scales", BLUE, font_size=8.4)
    _box(left, 0.53, 0.48, 0.44, 0.17, "$Q_N(W)=W_{\\rm bwd}$\nDgrad scales", ORANGE, font_size=8.4)
    _arrow(left, 0.42, 0.79, 0.24, 0.65)
    _arrow(left, 0.58, 0.79, 0.76, 0.65)
    left.text(0.50, 0.37, "$W_{\\rm fwd}\\ne W_{\\rm bwd}$", ha="center", fontsize=15, color=RED, fontweight="bold")
    left.text(
        0.50,
        0.15,
        "Independent scales produce two numerical weights.\n"
        "Dgrad then uses a different weight approximation from Fprop.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )

    right.set_title("Orientation-consistent 2D weight quantization")
    _box(right, 0.15, 0.78, 0.70, 0.13, "one $b\\times b$ tile scale + E2M1 payload", PURPLE, font_size=8.6)
    _box(right, 0.02, 0.47, 0.46, 0.17, "forward descriptor\nrow/swizzled view", BLUE, font_size=8.4)
    _box(right, 0.52, 0.47, 0.46, 0.17, "Dgrad descriptor\ncolumn/swizzled view", TEAL, font_size=8.4)
    _arrow(right, 0.42, 0.78, 0.24, 0.64)
    _arrow(right, 0.58, 0.78, 0.76, 0.64)
    right.text(0.50, 0.37, "shared numerical encoding", ha="center", fontsize=12, color=GREEN, fontweight="bold")
    right.text(
        0.50,
        0.14,
        "TE-native, v5, and localCTA use $b=16$; MXFP4 uses $b=32$.\n"
        "Our native v5/localCTA/MX producers emit TK-ready layouts directly.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    fig.tight_layout(w_pad=1.3)
    fig.savefig(OUTPUT_DIR / "fp4_2d_weight_contract.png", dpi=220)
    plt.close(fig)


def make_superseded_downstream_deltas_for_forensics() -> None:
    """Reproduce the no-scaling-RoPE diagnostic; never use it for publication."""
    report_root = OUTPUT_DIR.parent
    ledger_path = (
        report_root
        / "data/canonical_allroute_step38000_eval_r18_20260828/EVAL_LEDGER.csv"
    )
    expected_ledger_sha = "abf350e86ef85c8a07f86961bf54242b371654eaee9a5d256121d2d96b8dfa0a"
    if hashlib.sha256(ledger_path.read_bytes()).hexdigest() != expected_ledger_sha:
        raise RuntimeError("superseded downstream ledger hash changed")
    with ledger_path.open(newline="", encoding="utf-8") as handle:
        rows = {row["label"]: row for row in csv.DictReader(handle)}

    rht_dir = report_root / "data/mxfp4_rht_step38000_eval_r25_20260901"
    rht_path = rht_dir / "COMPLETED.json"
    expected_rht_sha = "9608ac57376b59788b4394c6a761e7106b53240a1597ff1680bf30d9f76c7efa"
    if hashlib.sha256(rht_path.read_bytes()).hexdigest() != expected_rht_sha:
        raise RuntimeError("MXFP4+RHT downstream receipt hash changed")
    rht = json.loads(rht_path.read_text(encoding="utf-8"))
    if rht.get("status") != "pass":
        raise RuntimeError("MXFP4+RHT downstream receipt is not complete")
    if rht.get("evaluation_semantics") != "fully-canonical-torchtitan":
        raise RuntimeError("MXFP4+RHT evaluator semantics are not canonical")

    # Bind the scores to the literal step-38,000 checkpoint. COMPLETED seals
    # task receipts; ROUTE_RECEIPT seals the checkpoint identity and prevents a
    # nearby terminal step from being substituted into the common panel.
    rht_route_path = rht_dir / "ROUTE_RECEIPT.json"
    expected_rht_route_sha = (
        "8ce928a497f2580df1372dd257fcaf92c0f3d3d3727638b7ce40811bd742ebe8"
    )
    if hashlib.sha256(rht_route_path.read_bytes()).hexdigest() != expected_rht_route_sha:
        raise RuntimeError("MXFP4+RHT route receipt hash changed")
    rht_route = json.loads(rht_route_path.read_text(encoding="utf-8"))
    expected_rht_route = {
        "status": "pass",
        "label": "mxfp4_rht",
        "route": "mxfp4-v4-row-sr-fused-v1",
        "step": 38000,
        "ntokens_seen": 2490368000,
        "evaluation_semantics": "fully-canonical-torchtitan",
        "completed_panel_sha256": expected_rht_sha,
    }
    for key, value in expected_rht_route.items():
        if rht_route.get(key) != value:
            raise RuntimeError(f"MXFP4+RHT route identity mismatch for {key}")

    # Exact canonical r22 results recovered from
    # mfu-can38-lcrht-b200-r22-20260830. The report-local receipt is a
    # byte-identical mirror and is mandatory for figure generation.
    expected_localcta_rht_r22 = {
        "mmlu_acc": 0.24426719840478564,
        "hellaswag_acc_norm": 0.6538538139812786,
        "winogrande_acc": 0.6393054459352802,
        "arc_challenge_acc_norm": 0.4402730375426621,
    }
    localcta_rht_path = (
        report_root / "data/localcta_rht_eval_r22_20260830/COMPLETED.json"
    )
    expected_localcta_rht_sha = (
        "e4dbc45f536c7f2391027e7ba139f329a05ac5163bd6b38cbe0ac01d19c01e1e"
    )
    if hashlib.sha256(localcta_rht_path.read_bytes()).hexdigest() != expected_localcta_rht_sha:
        raise RuntimeError("localCTA+RHT r22 downstream receipt hash changed")
    localcta_rht = json.loads(localcta_rht_path.read_text(encoding="utf-8"))
    if localcta_rht.get("status") != "pass":
        raise RuntimeError("localCTA+RHT r22 downstream receipt is not complete")
    if localcta_rht.get("evaluation_semantics") != "fully-canonical-torchtitan":
        raise RuntimeError("localCTA+RHT r22 evaluator semantics are not canonical")
    localcta_rht_values = {
        "mmlu_acc": localcta_rht["tasks"]["mmlu"]["metric_value"],
        "hellaswag_acc_norm": localcta_rht["tasks"]["hellaswag"][
            "metric_value"
        ],
        "winogrande_acc": localcta_rht["tasks"]["winogrande"]["metric_value"],
        "arc_challenge_acc_norm": localcta_rht["tasks"]["arc_challenge"][
            "metric_value"
        ],
    }
    if localcta_rht_values != expected_localcta_rht_r22:
        raise RuntimeError("localCTA+RHT r22 scores differ from the sealed results")

    h16_dir = (
        report_root / "data/h16_operand_hybrid_step38000_eval_r22_20260902"
    )
    h16_path = h16_dir / "COMPLETED.json"
    expected_h16_sha = (
        "132789489c79a0ec3c79a6300b40387d068d43db5f147d6b926b0972777b2ca8"
    )
    if hashlib.sha256(h16_path.read_bytes()).hexdigest() != expected_h16_sha:
        raise RuntimeError("operand-H16 downstream receipt hash changed")
    h16 = json.loads(h16_path.read_text(encoding="utf-8"))
    if not (
        h16.get("status") == "pass"
        and h16.get("evaluation_semantics") == "fully-canonical-torchtitan"
    ):
        raise RuntimeError("operand-H16 downstream panel is not canonical")

    h16_route_path = h16_dir / "ROUTE_RECEIPT.json"
    expected_h16_route_sha = (
        "47f43f9933abcda103e1a216445ce9184b0523f0be78b70ff9dae3d1897f8833"
    )
    if hashlib.sha256(h16_route_path.read_bytes()).hexdigest() != expected_h16_route_sha:
        raise RuntimeError("operand-H16 route receipt hash changed")
    h16_route = json.loads(h16_route_path.read_text(encoding="utf-8"))
    expected_h16_route = {
        "status": "pass",
        "label": "h16_operand_hybrid",
        "route": "mxfp4-v4-row-sr-fused-v1",
        "step": 38000,
        "ntokens_seen": 2490368000,
        "evaluation_semantics": "fully-canonical-torchtitan",
        "completed_panel_sha256": expected_h16_sha,
        "training_code_commit": "2588b447061df4b0b218d000e9cbbe8d23edf05c",
        "training_recipe": "operand-h16-mxfp4-col-rht-localcta-row-sr-dhidden",
    }
    for key, value in expected_h16_route.items():
        if h16_route.get(key) != value:
            raise RuntimeError(f"operand-H16 route identity mismatch for {key}")

    h16_values = {
        "mmlu_acc": h16["tasks"]["mmlu"]["metric_value"],
        "hellaswag_acc_norm": h16["tasks"]["hellaswag"]["metric_value"],
        "winogrande_acc": h16["tasks"]["winogrande"]["metric_value"],
        "arc_challenge_acc_norm": h16["tasks"]["arc_challenge"][
            "metric_value"
        ],
    }
    expected_h16_values = {
        "mmlu_acc": 0.2444808431847315,
        "hellaswag_acc_norm": 0.6642103166699861,
        "winogrande_acc": 0.6314127861089187,
        "arc_challenge_acc_norm": 0.4402730375426621,
    }
    if h16_values != expected_h16_values:
        raise RuntimeError("operand-H16 scores differ from the sealed results")

    metric_keys = [
        ("mmlu_acc", "MMLU 5-shot"),
        ("hellaswag_acc_norm", "HellaSwag 10-shot"),
        ("winogrande_acc", "WinoGrande 5-shot"),
        ("arc_challenge_acc_norm", "ARC-Challenge 25-shot"),
    ]
    route_specs = [
        ("te_native", "TE-native NVFP4", BLUE),
        ("pure_v5", "Global NVFP4 v5", PINK),
        ("mxfp4", "MXFP4 + row-SR", ORANGE),
        ("localcta", "CTA-local NVFP4", TEAL),
        ("localcta_mxfp4_hybrid", "27/5 depth hybrid", MID),
        ("localcta_rht", "CTA-local NVFP4 + fixed H16", RED),
        ("mxfp4_rht", "MXFP4 + row-SR + fixed H32", PURPLE),
        ("operand_h16", "Operand hybrid + plain H16", GREEN),
    ]
    rht_values = {
        "mmlu_acc": rht["tasks"]["mmlu"]["metric_value"],
        "hellaswag_acc_norm": rht["tasks"]["hellaswag"]["metric_value"],
        "winogrande_acc": rht["tasks"]["winogrande"]["metric_value"],
        "arc_challenge_acc_norm": rht["tasks"]["arc_challenge"]["metric_value"],
    }
    bf16 = rows["bf16"]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True)
    for ax, (metric, title) in zip(axes.flat, metric_keys):
        values = []
        labels = []
        colors = []
        baseline = float(bf16[metric])
        for key, label, color in route_specs:
            if key == "mxfp4_rht":
                value = rht_values[metric]
            elif key == "localcta_rht":
                value = localcta_rht_values[metric]
            elif key == "operand_h16":
                value = h16_values[metric]
            else:
                value = float(rows[key][metric])
            values.append(100.0 * (value - baseline))
            labels.append(label)
            colors.append(color)
        y = np.arange(len(labels))
        ax.barh(y, values, color=colors, height=0.62)
        ax.axvline(0.0, color=DARK, linewidth=1.0)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", color=LIGHT, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for yi, value in zip(y, values):
            ha = "left"
            pad = 0.12
            ax.text(value + pad, yi, f"{value:+.2f}", va="center", ha=ha, fontsize=8.3)
        ax.set_xlim(-8.25, 2.75)
    for ax in axes[1]:
        ax.set_xlabel("Accuracy difference from BF16 (percentage points)")
    fig.suptitle(
        "Superseded no-scaling-RoPE downstream diagnostic",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.005,
        "Forensic reproduction only: both evaluator arms omitted the long-context RoPE scaling used in training.",
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), h_pad=1.0, w_pad=1.2)
    fig.savefig(OUTPUT_DIR / "superseded_fp4_downstream_delta_bf16.png", dpi=220)
    plt.close(fig)


def make_recipe_cost_ledger() -> None:
    # These are observed component costs, not a stack. RHT/SR come from the
    # 1B MXFP4 screens; scaling and BF16-island values come from 1.2B runs.
    labels = [
        "Gradient stochastic rounding\n1B MXFP4, 20-step screen",
        "Activation RHT\n1B MXFP4, 50 steps",
        "Global tensor scaling\n1.2B NVFP4, no extras, 500 steps",
        "Global scaling under RHT + SR\n1.2B NVFP4 bridge, 500 steps",
        "BF16 output-layer fallback\n1.2B MXFP4 body, 50 steps",
        "Late BF16 carve-out (~15% policy)\n1.2B TE: final 4/20 blocks, 500 steps",
    ]
    costs = [2.94, 3.10, 3.12, 8.07, 10.98, 19.91]
    colors = [ORANGE, BLUE, PINK, TEAL, RED, DARK]

    fig, ax = plt.subplots(figsize=(10.4, 6.5))
    bars = ax.barh(range(len(labels)), costs, color=colors, height=0.58)
    bars[0].set_hatch("//")
    bars[0].set_edgecolor("white")
    bars[0].set_linewidth(0.8)
    ax.set_xlim(0, 22.0)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Observed BF16-MFU cost (percentage points)")
    ax.set_title("Model-level BF16-MFU cost of FP4 overheads")
    ax.grid(axis="x", color=LIGHT, linewidth=0.9)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, costs):
        ax.text(
            value + 0.22,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            ha="left",
            va="center",
            color=DARK,
            fontsize=9.5,
            fontweight="bold",
        )
    ax.text(
        0.5,
        -0.25,
        "Positive values are BF16-MFU points lost; bars are not additive.\n"
        "Hatched SR is a short screen; a matched 500-step factorial is still "
        "owed.\nLate BF16 is the measured final-4-block proxy for the paper's "
        "~15% sensitive-linear policy.",
        transform=ax.transAxes,
        ha="center",
        fontsize=8.5,
        color=MID,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fp4_recipe_cost_ledger.png", dpi=220)
    plt.close(fig)


def make_localcta_scaling() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.7))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    left, right = axes
    left.set_title("Global NVFP4 tensor scaling", pad=13)
    _box(left, 0.04, 0.73, 0.20, 0.14, "BF16\nA or B", MID)
    _box(left, 0.35, 0.73, 0.25, 0.14, "global amax\nreduction", RED)
    _box(left, 0.71, 0.73, 0.24, 0.14, "one tensor\nscale $s_g$", ORANGE)
    _arrow(left, 0.24, 0.80, 0.35, 0.80)
    _arrow(left, 0.60, 0.80, 0.71, 0.80)

    left.plot([0.08, 0.92], [0.63, 0.63], color=RED, linewidth=2.2)
    left.text(
        0.50,
        0.655,
        "inter-CTA completion barrier",
        color=RED,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )
    _box(
        left,
        0.16,
        0.38,
        0.30,
        0.15,
        "block amax\nE4M3 scales",
        PINK,
    )
    _box(
        left,
        0.56,
        0.38,
        0.30,
        0.15,
        "FP4 payloads\n+ scale swizzle",
        TEAL,
    )
    _arrow(left, 0.31, 0.63, 0.31, 0.53)
    _arrow(left, 0.46, 0.455, 0.56, 0.455)
    _box(left, 0.33, 0.11, 0.34, 0.16, "Tensor Core GEMM\nthen decode", BLUE)
    _arrow(left, 0.71, 0.38, 0.60, 0.27)
    left.text(
        0.50,
        0.025,
        "All blocks depend on the tensor-wide result.",
        ha="center",
        fontsize=9.2,
        color=MID,
    )

    right.set_title("CTA-local hierarchical scaling", pad=13)
    _box(right, 0.03, 0.73, 0.25, 0.14, "A row tile $i$\n256 rows", MID)
    _box(right, 0.38, 0.73, 0.25, 0.14, "B column tile $j$\n256 columns", MID)
    _box(right, 0.05, 0.45, 0.21, 0.16, "$q_A$, E4M3\n$\\alpha_i$ in FP32", TEAL)
    _box(right, 0.40, 0.45, 0.21, 0.16, "$q_B$, E4M3\n$\\beta_j$ in FP32", TEAL)
    _arrow(right, 0.155, 0.73, 0.155, 0.61)
    _arrow(right, 0.505, 0.73, 0.505, 0.61)

    grid_x, grid_y = 0.71, 0.48
    cell_w, cell_h = 0.105, 0.10
    for row in range(2):
        for col in range(2):
            patch = FancyBboxPatch(
                (grid_x + col * (cell_w + 0.012), grid_y + row * (cell_h + 0.012)),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.006,rounding_size=0.008",
                facecolor=BLUE if (row, col) == (0, 0) else LIGHT,
                edgecolor=BLUE,
                linewidth=1.0,
            )
            right.add_patch(patch)
            right.text(
                grid_x + col * (cell_w + 0.012) + cell_w / 2,
                grid_y + row * (cell_h + 0.012) + cell_h / 2,
                "$C_{ij}$" if (row, col) == (0, 0) else "$C$ tile",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if (row, col) == (0, 0) else DARK,
                fontweight="bold",
            )
    _arrow(right, 0.26, 0.53, 0.70, 0.56)
    _arrow(right, 0.61, 0.53, 0.70, 0.56)
    right.text(
        0.815,
        0.71,
        "output-tile CTAs",
        ha="center",
        fontsize=9.2,
        color=DARK,
        fontweight="bold",
    )

    _box(
        right,
        0.15,
        0.15,
        0.70,
        0.17,
        "full K accumulation in TMEM\n"
        "epilogue multiply once by $\\alpha_i\\beta_j$",
        GREEN,
    )
    _arrow(right, 0.815, 0.48, 0.69, 0.32)
    right.text(
        0.50,
        0.025,
        "No tensor-wide amax barrier; outer scales stay constant along K.",
        ha="center",
        fontsize=9.2,
        color=MID,
    )

    fig.tight_layout(w_pad=1.3)
    fig.savefig(OUTPUT_DIR / "fp4_localcta_scaling.png", dpi=220)
    plt.close(fig)


def make_mxfp4_scale_rounding() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    axes[0].set_title("Encode-safe scale: round exponent upward")
    _box(axes[0], 0.06, 0.67, 0.34, 0.16, "ideal $s^*=\\max|x|/6$", MID)
    _box(axes[0], 0.60, 0.67, 0.34, 0.16, "$2^{\\lceil\\log_2 s^*\\rceil}$", BLUE)
    _arrow(axes[0], 0.40, 0.75, 0.60, 0.75)
    _box(axes[0], 0.19, 0.34, 0.62, 0.18, "block maximum fits\nno E2M1 saturation", GREEN)
    _arrow(axes[0], 0.77, 0.67, 0.66, 0.52)
    axes[0].text(
        0.50,
        0.16,
        "Safer encoding, but the quantization step can be up to 2x coarser.",
        ha="center",
        fontsize=9.5,
        color=MID,
        wrap=True,
    )

    axes[1].set_title("Decode-dense scale: round exponent downward")
    _box(axes[1], 0.06, 0.67, 0.34, 0.16, "ideal $s^*=\\max|x|/6$", MID)
    _box(axes[1], 0.60, 0.67, 0.34, 0.16, "$2^{\\lfloor\\log_2 s^*\\rfloor}$", TEAL)
    _arrow(axes[1], 0.40, 0.75, 0.60, 0.75)
    _box(
        axes[1],
        0.19,
        0.34,
        0.62,
        0.18,
        "denser grid for most values\npossible outlier saturation",
        ORANGE,
    )
    _arrow(axes[1], 0.77, 0.67, 0.66, 0.52)
    axes[1].text(
        0.50,
        0.16,
        "Adjacent E8M0 scales differ by exactly 2; this is the mechanism "
        "behind one-step scale halving.",
        ha="center",
        fontsize=9.5,
        color=MID,
        wrap=True,
    )

    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUTPUT_DIR / "fp4_mxfp4_scale_rounding.png", dpi=220)
    plt.close(fig)


def main() -> None:
    _style()
    make_1p2b_mfu()
    make_fusion_dataflow()
    make_format_execution_routes()
    make_linear_operand_map()
    make_2d_weight_contract()
    # The former downstream plot used an evaluator that omitted the training
    # model's long-context RoPE scaling. Keep its sealed inputs as provenance,
    # but do not regenerate a publication figure from superseded scores.
    make_recipe_cost_ledger()
    make_localcta_scaling()
    make_mxfp4_scale_rounding()


if __name__ == "__main__":
    main()
