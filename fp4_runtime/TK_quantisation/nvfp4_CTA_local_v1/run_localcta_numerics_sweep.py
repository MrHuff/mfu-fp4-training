import argparse
import copy
import json
from pathlib import Path

import torch

import test_localcta_numerics as numerics

ROOT = Path(__file__).resolve().parent


def case_definitions() -> list[dict]:
    return [
        {
            "name": "ffn_qkv_normal_m16384",
            "description": "Representative large FFN/QKV-like normal inputs",
            "regular_m": 16384,
            "regular_n": 5632,
            "regular_k": 2048,
            "grouped_m": 16384,
            "grouped_k": 2048,
            "grouped_splits": [2048, 2048, 2048],
            "distribution": "normal",
        },
        {
            "name": "ffn_qkv_normal_m65536",
            "description": "Very large-M FFN/QKV-like normal inputs",
            "regular_m": 65536,
            "regular_n": 5632,
            "regular_k": 2048,
            "grouped_m": 65536,
            "grouped_k": 2048,
            "grouped_splits": [2048, 2048, 2048],
            "distribution": "normal",
            "max_quantile_samples": 1_000_000,
        },
        {
            "name": "square_normal_4096",
            "description": "Square-ish high-K case",
            "regular_m": 4096,
            "regular_n": 4096,
            "regular_k": 4096,
            "grouped_m": 4096,
            "grouped_k": 4096,
            "grouped_splits": [4096, 4096, 4096],
            "distribution": "normal",
        },
        {
            "name": "laplace_both_m4096",
            "description": "Heavy-tailed Laplace activations and weights",
            "regular_m": 4096,
            "regular_n": 5632,
            "regular_k": 2048,
            "grouped_m": 4096,
            "grouped_k": 2048,
            "grouped_splits": [2048, 2048, 2048],
            "distribution": "laplace",
        },
        {
            "name": "sparse_spikes_activation_m4096",
            "description": "Sparse activation spikes with otherwise normal weights",
            "regular_m": 4096,
            "regular_n": 5632,
            "regular_k": 2048,
            "grouped_m": 4096,
            "grouped_k": 2048,
            "grouped_splits": [2048, 2048, 2048],
            "distribution": "normal",
            "regular_activation_dist": "sparse_spikes",
            "grouped_activation_dist": "sparse_spikes",
            "sparse_spike_prob": 2e-4,
            "sparse_spike_scale": 32.0,
        },
        {
            "name": "row_spikes_activation_m4096",
            "description": "Row-correlated activation spikes with normal weights",
            "regular_m": 4096,
            "regular_n": 5632,
            "regular_k": 2048,
            "grouped_m": 4096,
            "grouped_k": 2048,
            "grouped_splits": [2048, 2048, 2048],
            "distribution": "normal",
            "regular_activation_dist": "row_spikes",
            "grouped_activation_dist": "row_spikes",
            "row_spike_prob": 0.02,
            "row_spike_scale": 12.0,
        },
        {
            "name": "chunk_spikes_weight_m4096",
            "description": "Chunk-local weight spikes with normal activations",
            "regular_m": 4096,
            "regular_n": 5632,
            "regular_k": 2048,
            "grouped_m": 4096,
            "grouped_k": 2048,
            "grouped_splits": [2048, 2048, 2048],
            "distribution": "normal",
            "regular_weight_dist": "chunk_spikes",
            "grouped_weight_dist": "chunk_spikes",
            "chunk_spike_prob": 0.10,
            "chunk_spike_scale": 8.0,
        },
    ]


def apply_case_overrides(base_args: argparse.Namespace, case: dict) -> argparse.Namespace:
    args = copy.deepcopy(base_args)
    for key, value in case.items():
        if key in {"name", "description"}:
            continue
        setattr(args, key.replace("-", "_"), value)
    return args


def safe_ratio(numer: float, denom: float) -> float:
    return numer / denom if denom != 0 else float("inf")


def summarize_case(case_name: str, case_result: dict) -> dict:
    results = case_result["results"]
    tr = results["tensor_reconstruction"]
    regular = results["regular_matmul_output"]
    grouped = results["grouped_qkv_like_matmul_output"]

    reg_act_local = tr["regular_activation"]["metrics"]["localcta_vs_bf16"]
    reg_act_base = tr["regular_activation"]["metrics"]["baseline_v5_vs_bf16"]
    reg_w_local = tr["regular_weight"]["metrics"]["localcta_vs_bf16"]
    reg_w_base = tr["regular_weight"]["metrics"]["baseline_v5_vs_bf16"]
    grp_act_local = tr["grouped_activation"]["metrics"]["localcta_vs_bf16"]
    grp_act_base = tr["grouped_activation"]["metrics"]["baseline_v5_vs_bf16"]
    grp_w_local = tr["grouped_weight"]["metrics"]["localcta_vs_bf16"]
    grp_w_base = tr["grouped_weight"]["metrics"]["baseline_v5_vs_bf16"]
    reg_local = regular["metrics"]["localcta_prepared_vs_bf16"]
    reg_base = regular["metrics"]["baseline_v5_vs_bf16"]
    grp_local = grouped["metrics"]["localcta_prepared_vs_bf16"]
    grp_base = grouped["metrics"]["baseline_v5_vs_bf16"]

    return {
        "name": case_name,
        "description": case_result["description"],
        "regular_recon_rms_ratio_mean": 0.5 * (
            safe_ratio(reg_act_local["rms"], reg_act_base["rms"]) +
            safe_ratio(reg_w_local["rms"], reg_w_base["rms"])
        ),
        "grouped_recon_rms_ratio_mean": 0.5 * (
            safe_ratio(grp_act_local["rms"], grp_act_base["rms"]) +
            safe_ratio(grp_w_local["rms"], grp_w_base["rms"])
        ),
        "regular_matmul_rms_ratio": safe_ratio(reg_local["rms"], reg_base["rms"]),
        "grouped_matmul_rms_ratio": safe_ratio(grp_local["rms"], grp_base["rms"]),
        "regular_matmul_mean_abs_ratio": safe_ratio(reg_local["mean_abs"], reg_base["mean_abs"]),
        "grouped_matmul_mean_abs_ratio": safe_ratio(grp_local["mean_abs"], grp_base["mean_abs"]),
        "regular_act_chunk_cv": tr["regular_activation"]["localcta_chunk_scale_summary"]["coeff_var"],
        "regular_w_chunk_cv": tr["regular_weight"]["localcta_chunk_scale_summary"]["coeff_var"],
        "grouped_act_chunk_cv": tr["grouped_activation"]["localcta_chunk_scale_summary"]["coeff_var"],
        "grouped_w_chunk_cv": tr["grouped_weight"]["localcta_chunk_scale_summary"]["coeff_var"],
    }


def interpret_case(summary: dict) -> str:
    recon_better = summary["regular_recon_rms_ratio_mean"] < 1.0 and summary["grouped_recon_rms_ratio_mean"] < 1.0
    regular_worse = summary["regular_matmul_rms_ratio"] > 1.0
    grouped_worse = summary["grouped_matmul_rms_ratio"] > 1.0
    if recon_better and regular_worse and grouped_worse:
        return "better tensor reconstruction, worse regular and grouped matmul"
    if recon_better and regular_worse:
        return "better reconstruction, worse regular matmul"
    if recon_better and grouped_worse:
        return "better reconstruction, worse grouped matmul"
    if regular_worse or grouped_worse:
        return "matmul degradation dominates"
    return "localcta tracks or beats baseline"


def build_markdown_report(case_results: list[dict], summaries: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# LocalCTA Prepared Numerics Report")
    lines.append("")
    lines.append("This report compares `localcta_prepared` against `baseline_v5` for tensor reconstruction and BF16 matmul output.")
    lines.append("")
    lines.append("## Headline Findings")
    lines.append("")

    worst_regular = max(summaries, key=lambda item: item["regular_matmul_rms_ratio"])
    worst_grouped = max(summaries, key=lambda item: item["grouped_matmul_rms_ratio"])
    highest_cv = max(summaries, key=lambda item: max(item["regular_act_chunk_cv"], item["regular_w_chunk_cv"], item["grouped_act_chunk_cv"], item["grouped_w_chunk_cv"]))
    recon_better_matmul_worse = [
        item["name"] for item in summaries
        if item["regular_recon_rms_ratio_mean"] < 1.0 and item["regular_matmul_rms_ratio"] > 1.0
    ]

    lines.append(f"- Worst regular matmul RMS gap: `{worst_regular['name']}` at `{worst_regular['regular_matmul_rms_ratio']:.3f}x` baseline.")
    lines.append(f"- Worst grouped matmul RMS gap: `{worst_grouped['name']}` at `{worst_grouped['grouped_matmul_rms_ratio']:.3f}x` baseline.")
    lines.append(f"- Highest localCTA chunk-scale variation: `{highest_cv['name']}` with max coeff-var `{max(highest_cv['regular_act_chunk_cv'], highest_cv['regular_w_chunk_cv'], highest_cv['grouped_act_chunk_cv'], highest_cv['grouped_w_chunk_cv']):.3f}`.")
    if recon_better_matmul_worse:
        lines.append(f"- Cases where localCTA reconstructs better on average but still loses in regular matmul RMS: `{', '.join(recon_better_matmul_worse)}`.")
    lines.append("")
    lines.append("## Case Summary")
    lines.append("")
    lines.append("| Case | Regular Recon RMS Ratio | Grouped Recon RMS Ratio | Regular Matmul RMS Ratio | Grouped Matmul RMS Ratio | Max Chunk-Scale CV | Interpretation |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for summary in summaries:
        max_cv = max(summary["regular_act_chunk_cv"], summary["regular_w_chunk_cv"], summary["grouped_act_chunk_cv"], summary["grouped_w_chunk_cv"])
        lines.append(
            f"| `{summary['name']}` | "
            f"{summary['regular_recon_rms_ratio_mean']:.3f}x | "
            f"{summary['grouped_recon_rms_ratio_mean']:.3f}x | "
            f"{summary['regular_matmul_rms_ratio']:.3f}x | "
            f"{summary['grouped_matmul_rms_ratio']:.3f}x | "
            f"{max_cv:.3f} | "
            f"{interpret_case(summary)} |"
        )
    lines.append("")
    lines.append("## Detailed Notes")
    lines.append("")

    for case_result, summary in zip(case_results, summaries):
        cfg = case_result["results"]["config"]
        tr = case_result["results"]["tensor_reconstruction"]
        reg = case_result["results"]["regular_matmul_output"]
        grp = case_result["results"]["grouped_qkv_like_matmul_output"]
        lines.append(f"### {case_result['name']}")
        lines.append("")
        lines.append(case_result["description"])
        lines.append("")
        lines.append(f"- Regular shape: `{tuple(cfg['regular_shape'])}`")
        lines.append(f"- Grouped shape: `{tuple(cfg['grouped_shape'])}`, splits=`{cfg['grouped_splits']}`")
        lines.append(f"- Distributions: `{cfg['distributions']}`")
        lines.append(
            f"- Regular reconstruction RMS ratios: activation `{safe_ratio(tr['regular_activation']['metrics']['localcta_vs_bf16']['rms'], tr['regular_activation']['metrics']['baseline_v5_vs_bf16']['rms']):.3f}x`, "
            f"weight `{safe_ratio(tr['regular_weight']['metrics']['localcta_vs_bf16']['rms'], tr['regular_weight']['metrics']['baseline_v5_vs_bf16']['rms']):.3f}x`"
        )
        lines.append(
            f"- Regular matmul RMS ratio: `{summary['regular_matmul_rms_ratio']:.3f}x`; mean-abs ratio: `{summary['regular_matmul_mean_abs_ratio']:.3f}x`"
        )
        lines.append(
            f"- Grouped matmul RMS ratio: `{summary['grouped_matmul_rms_ratio']:.3f}x`; mean-abs ratio: `{summary['grouped_matmul_mean_abs_ratio']:.3f}x`"
        )
        lines.append(
            f"- Regular activation input abs p99/p999/max: "
            f"`{tr['regular_activation']['input_summary']['abs_p99']:.4e} / {tr['regular_activation']['input_summary']['abs_p999']:.4e} / {tr['regular_activation']['input_summary']['abs_max']:.4e}`"
        )
        lines.append(
            f"- LocalCTA regular chunk-scale coeff-var: activation `{tr['regular_activation']['localcta_chunk_scale_summary']['coeff_var']:.3f}`, "
            f"weight `{tr['regular_weight']['localcta_chunk_scale_summary']['coeff_var']:.3f}`"
        )
        lines.append(
            f"- Output RMS amplification: regular localCTA `{reg['diagnostics']['localcta_output_rms_amplification']:.3f}` vs baseline `{reg['diagnostics']['baseline_output_rms_amplification']:.3f}`; "
            f"grouped localCTA `{grp['diagnostics']['localcta_output_rms_amplification']:.3f}` vs baseline `{grp['diagnostics']['baseline_output_rms_amplification']:.3f}`"
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a curated numerics sweep and write a markdown report")
    parser.add_argument("--output-json", type=Path, default=Path("/tmp/localcta_numerics_sweep.json"))
    parser.add_argument("--output-md", type=Path, default=ROOT / "LOCALCTA_NUMERICS_REPORT.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case-filter", type=str, default=None, help="substring filter for case names")
    args = parser.parse_args()

    base_parser = numerics.build_parser()
    base_args = base_parser.parse_args([])
    base_args.seed = args.seed

    selected_cases = case_definitions()
    if args.case_filter:
        selected_cases = [case for case in selected_cases if args.case_filter in case["name"]]
    if not selected_cases:
        raise SystemExit("No cases matched --case-filter")

    results_bundle: list[dict] = []
    summaries: list[dict] = []

    for case in selected_cases:
        print(f"\n=== Running {case['name']} ===")
        case_args = apply_case_overrides(base_args, case)
        case_results = numerics.run_experiment(case_args)
        result_entry = {
            "name": case["name"],
            "description": case["description"],
            "results": case_results,
        }
        results_bundle.append(result_entry)
        summaries.append(summarize_case(case["name"], result_entry))
        torch.cuda.empty_cache()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"cases": results_bundle, "summaries": summaries}, indent=2))
    report = build_markdown_report(results_bundle, summaries)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(report)

    print(f"\njson_report={args.output_json}")
    print(f"markdown_report={args.output_md}")


if __name__ == "__main__":
    main()
