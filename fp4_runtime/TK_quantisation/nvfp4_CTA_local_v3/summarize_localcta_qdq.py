import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SECTIONS = [
    "regular_activation",
    "regular_weight",
    "grouped_activation",
    "grouped_weight",
]

SECTION_LABELS = {
    "regular_activation": "Regular Activation",
    "regular_weight": "Regular Weight",
    "grouped_activation": "Grouped Activation",
    "grouped_weight": "Grouped Weight",
}

ABS_METRICS = ["rms", "p99_abs", "max_abs"]
REL_METRICS = ["mean_rel", "p99_rel", "max_rel"]


def ratio(numer: float, denom: float) -> float:
    return numer / denom if denom != 0 else float("inf")


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["cases"]


def section_ratio_row(case: dict, section: str, metrics: list[str]) -> list[float]:
    metrics_blob = case["results"]["tensor_reconstruction"][section]["metrics"]
    local = metrics_blob["localcta_vs_bf16"]
    base = metrics_blob["baseline_v5_vs_bf16"]
    return [ratio(local[metric], base[metric]) for metric in metrics]


def build_matrix(cases: list[dict], section: str, metrics: list[str]) -> np.ndarray:
    return np.array([section_ratio_row(case, section, metrics) for case in cases], dtype=float)


def heatmap(ax, data: np.ndarray, row_labels: list[str], col_labels: list[str], title: str) -> None:
    im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=min(0.95, np.nanmin(data)), vmax=max(1.15, np.nanmax(data)))
    ax.set_title(title)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def render_plots(cases: list[dict], output_abs: Path, output_rel: Path) -> None:
    case_labels = [case["name"] for case in cases]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    for ax, section in zip(axes.flat, SECTIONS):
        heatmap(
            ax,
            build_matrix(cases, section, ABS_METRICS),
            case_labels,
            ABS_METRICS,
            f"{SECTION_LABELS[section]} Abs Error Ratio: localCTA / baseline_v5",
        )
    fig.suptitle("LocalCTA Quantize-Dequantize Absolute Error Ratios", fontsize=16)
    output_abs.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_abs, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    for ax, section in zip(axes.flat, SECTIONS):
        heatmap(
            ax,
            build_matrix(cases, section, REL_METRICS),
            case_labels,
            REL_METRICS,
            f"{SECTION_LABELS[section]} Rel Error Ratio: localCTA / baseline_v5",
        )
    fig.suptitle("LocalCTA Quantize-Dequantize Relative Error Ratios", fontsize=16)
    output_rel.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_rel, dpi=200)
    plt.close(fig)


def best_case(cases: list[dict], section: str, metric: str) -> tuple[str, float]:
    scored = [
        (case["name"], ratio(
            case["results"]["tensor_reconstruction"][section]["metrics"]["localcta_vs_bf16"][metric],
            case["results"]["tensor_reconstruction"][section]["metrics"]["baseline_v5_vs_bf16"][metric],
        ))
        for case in cases
    ]
    return min(scored, key=lambda item: item[1])


def worst_case(cases: list[dict], section: str, metric: str) -> tuple[str, float]:
    scored = [
        (case["name"], ratio(
            case["results"]["tensor_reconstruction"][section]["metrics"]["localcta_vs_bf16"][metric],
            case["results"]["tensor_reconstruction"][section]["metrics"]["baseline_v5_vs_bf16"][metric],
        ))
        for case in cases
    ]
    return max(scored, key=lambda item: item[1])


def build_report(cases: list[dict], output_abs: Path, output_rel: Path) -> str:
    lines: list[str] = []
    lines.append("# LocalCTA Quantize-Dequantize Report")
    lines.append("")
    lines.append("This report isolates tensor reconstruction only: BF16 -> quantize -> dequantize.")
    lines.append("Ratios are `localCTA / baseline_v5`, so values below `1.0x` mean localCTA is better.")
    lines.append("")
    lines.append("## Headline Findings")
    lines.append("")

    best_sparse = best_case(cases, "regular_activation", "rms")
    worst_row = worst_case(cases, "regular_activation", "rms")
    best_max_abs = best_case(cases, "regular_activation", "max_abs")
    worst_p99_abs = worst_case(cases, "regular_activation", "p99_abs")

    lines.append(f"- Best regular-activation RMS case for localCTA: `{best_sparse[0]}` at `{best_sparse[1]:.3f}x`.")
    lines.append(f"- Worst regular-activation RMS case for localCTA: `{worst_row[0]}` at `{worst_row[1]:.3f}x`.")
    lines.append(f"- Best regular-activation max-abs case for localCTA: `{best_max_abs[0]}` at `{best_max_abs[1]:.3f}x`.")
    lines.append(f"- Worst regular-activation p99-abs case for localCTA: `{worst_p99_abs[0]}` at `{worst_p99_abs[1]:.3f}x`.")
    lines.append("- Relative-error ratios stay close to `1.0x` almost everywhere; the larger differences show up in bulk absolute-error metrics like RMS.")
    lines.append("- In other words: localCTA often reduces the single worst absolute miss, but it usually does not improve the overall reconstruction distribution.")
    lines.append("")
    lines.append("## Per-Case Summary")
    lines.append("")
    lines.append("| Case | Reg Act RMS | Reg Act p99 abs | Reg Act max abs | Reg W RMS | Reg W p99 abs | Reg W max abs |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        ra = case["results"]["tensor_reconstruction"]["regular_activation"]["metrics"]
        rw = case["results"]["tensor_reconstruction"]["regular_weight"]["metrics"]
        lines.append(
            f"| `{case['name']}` | "
            f"{ratio(ra['localcta_vs_bf16']['rms'], ra['baseline_v5_vs_bf16']['rms']):.3f}x | "
            f"{ratio(ra['localcta_vs_bf16']['p99_abs'], ra['baseline_v5_vs_bf16']['p99_abs']):.3f}x | "
            f"{ratio(ra['localcta_vs_bf16']['max_abs'], ra['baseline_v5_vs_bf16']['max_abs']):.3f}x | "
            f"{ratio(rw['localcta_vs_bf16']['rms'], rw['baseline_v5_vs_bf16']['rms']):.3f}x | "
            f"{ratio(rw['localcta_vs_bf16']['p99_abs'], rw['baseline_v5_vs_bf16']['p99_abs']):.3f}x | "
            f"{ratio(rw['localcta_vs_bf16']['max_abs'], rw['baseline_v5_vs_bf16']['max_abs']):.3f}x |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Normal large shapes: localCTA is consistently worse on reconstruction RMS, typically around `1.05x-1.07x`.")
    lines.append("- Sparse activation spikes: this is the one encouraging case. Activation RMS gets slightly better, but only marginally, and p99 abs still gets a bit worse.")
    lines.append("- Row-correlated spikes: this is the clear failure mode. LocalCTA gets substantially worse on reconstruction RMS even though max abs is still better.")
    lines.append("- Heavy tails: Laplace inputs increase the gap, especially in p99 abs.")
    lines.append("- Max absolute error alone is misleading here. LocalCTA often improves max abs while still degrading RMS and p99 abs.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"- Absolute error ratios: `{output_abs.name}`")
    lines.append(f"- Relative error ratios: `{output_rel.name}`")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("The current CTA-local scheme is not yet winning on pure quantize-dequantize for most tested distributions. ")
    lines.append("Its main visible benefit is usually smaller worst-case absolute misses, not lower bulk reconstruction error.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize localCTA vs baseline_v5 quantize-dequantize numerics")
    parser.add_argument("--input-json", type=Path, default=Path("/tmp/localcta_numerics_sweep.json"))
    root = Path(__file__).resolve().parent
    parser.add_argument("--output-report", type=Path, default=root / "LOCALCTA_QDQ_REPORT.md")
    parser.add_argument("--output-abs-plot", type=Path, default=root / "LOCALCTA_QDQ_ABS_RATIOS.png")
    parser.add_argument("--output-rel-plot", type=Path, default=root / "LOCALCTA_QDQ_REL_RATIOS.png")
    args = parser.parse_args()

    cases = load_cases(args.input_json)
    render_plots(cases, args.output_abs_plot, args.output_rel_plot)
    report = build_report(cases, args.output_abs_plot, args.output_rel_plot)
    args.output_report.write_text(report)
    print(f"report={args.output_report}")
    print(f"abs_plot={args.output_abs_plot}")
    print(f"rel_plot={args.output_rel_plot}")


if __name__ == "__main__":
    main()
