import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def average(values: list[float]) -> float:
    return float(sum(values) / len(values))


def heatmap(ax, data: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, cmap: str) -> None:
    im = ax.imshow(data, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def build_error_ratio_matrices(cases: list[dict]) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metrics = ["mean_abs", "p99_abs", "max_abs", "mean_rel", "p99_rel", "max_rel", "rms"]
    labels = [case["name"] for case in cases]
    reg_recon = []
    grp_recon = []
    reg_out = []
    grp_out = []

    for case in cases:
        tr = case["results"]["tensor_reconstruction"]
        reg = case["results"]["regular_matmul_output"]["metrics"]
        grp = case["results"]["grouped_qkv_like_matmul_output"]["metrics"]

        reg_recon.append([
            average([
                tr["regular_activation"]["metrics"]["localcta_vs_bf16"][metric] / tr["regular_activation"]["metrics"]["baseline_v5_vs_bf16"][metric],
                tr["regular_weight"]["metrics"]["localcta_vs_bf16"][metric] / tr["regular_weight"]["metrics"]["baseline_v5_vs_bf16"][metric],
            ])
            for metric in metrics
        ])
        grp_recon.append([
            average([
                tr["grouped_activation"]["metrics"]["localcta_vs_bf16"][metric] / tr["grouped_activation"]["metrics"]["baseline_v5_vs_bf16"][metric],
                tr["grouped_weight"]["metrics"]["localcta_vs_bf16"][metric] / tr["grouped_weight"]["metrics"]["baseline_v5_vs_bf16"][metric],
            ])
            for metric in metrics
        ])
        reg_out.append([
            reg["localcta_prepared_vs_bf16"][metric] / reg["baseline_v5_vs_bf16"][metric]
            for metric in metrics
        ])
        grp_out.append([
            grp["localcta_prepared_vs_bf16"][metric] / grp["baseline_v5_vs_bf16"][metric]
            for metric in metrics
        ])

    return labels, metrics, np.array(reg_recon), np.array(grp_recon), np.array(reg_out), np.array(grp_out)


def build_diagnostic_matrix(cases: list[dict]) -> tuple[list[str], list[str], np.ndarray]:
    labels = [case["name"] for case in cases]
    metrics = [
        "reg_act_max/p999",
        "reg_act_scale_cv",
        "reg_w_scale_cv",
        "grp_act_max/p999",
        "grp_act_scale_cv",
        "grp_w_scale_cv",
    ]
    rows = []
    for case in cases:
        tr = case["results"]["tensor_reconstruction"]
        rows.append([
            tr["regular_activation"]["input_summary"]["max_over_p999"],
            tr["regular_activation"]["localcta_chunk_scale_summary"]["coeff_var"],
            tr["regular_weight"]["localcta_chunk_scale_summary"]["coeff_var"],
            tr["grouped_activation"]["input_summary"]["max_over_p999"],
            tr["grouped_activation"]["localcta_chunk_scale_summary"]["coeff_var"],
            tr["grouped_weight"]["localcta_chunk_scale_summary"]["coeff_var"],
        ])
    return labels, metrics, np.array(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot localCTA numerics sweep heatmaps")
    parser.add_argument("--input-json", type=Path, default=Path("/tmp/localcta_numerics_sweep.json"))
    parser.add_argument("--output-error", type=Path, default=Path(__file__).resolve().parent / "LOCALCTA_NUMERICS_ERROR_RATIOS.png")
    parser.add_argument("--output-diagnostics", type=Path, default=Path(__file__).resolve().parent / "LOCALCTA_NUMERICS_OUTLIER_DIAGNOSTICS.png")
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text())
    cases = payload["cases"]

    labels, metric_labels, reg_recon, grp_recon, reg_out, grp_out = build_error_ratio_matrices(cases)
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), constrained_layout=True)
    heatmap(axes[0, 0], reg_recon, labels, metric_labels, "Regular Reconstruction Ratio: localCTA / baseline_v5", "coolwarm")
    heatmap(axes[0, 1], grp_recon, labels, metric_labels, "Grouped Reconstruction Ratio: localCTA / baseline_v5", "coolwarm")
    heatmap(axes[1, 0], reg_out, labels, metric_labels, "Regular Matmul Ratio: localCTA / baseline_v5", "coolwarm")
    heatmap(axes[1, 1], grp_out, labels, metric_labels, "Grouped Matmul Ratio: localCTA / baseline_v5", "coolwarm")
    fig.suptitle("LocalCTA Numerics Error Ratios", fontsize=16)
    args.output_error.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_error, dpi=200)
    plt.close(fig)

    diag_labels, diag_metric_labels, diagnostics = build_diagnostic_matrix(cases)
    fig, ax = plt.subplots(1, 1, figsize=(14, 8), constrained_layout=True)
    heatmap(ax, diagnostics, diag_labels, diag_metric_labels, "Input Outlier Severity and LocalCTA Chunk-Scale Variation", "viridis")
    args.output_diagnostics.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_diagnostics, dpi=200)
    plt.close(fig)

    print(f"error_ratio_plot={args.output_error}")
    print(f"diagnostic_plot={args.output_diagnostics}")


if __name__ == "__main__":
    main()
