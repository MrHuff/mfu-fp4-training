import argparse
import copy
import json
from pathlib import Path

import torch

import test_localcta_numerics as numerics


SCENARIOS = {
    "normal_both": {},
    "laplace_both": {
        "distribution": "laplace",
    },
    "student_t_both": {
        "distribution": "student_t",
    },
    "sparse_spikes_activation": {
        "distribution": "normal",
        "regular_activation_dist": "sparse_spikes",
        "grouped_activation_dist": "sparse_spikes",
    },
    "row_spikes_activation": {
        "distribution": "normal",
        "regular_activation_dist": "row_spikes",
        "grouped_activation_dist": "row_spikes",
    },
    "chunk_spikes_weight": {
        "distribution": "normal",
        "regular_weight_dist": "chunk_spikes",
        "grouped_weight_dist": "chunk_spikes",
    },
}


def parse_float_list(value: str) -> list[float]:
    parts = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not parts:
        raise ValueError("expected at least one float value")
    return [float(piece) for piece in parts]


def apply_scenario(args: argparse.Namespace, scenario: str) -> argparse.Namespace:
    cfg = SCENARIOS[scenario]
    out = copy.deepcopy(args)
    for key, value in cfg.items():
        setattr(out, key, value)
    return out


def summarize_result(results: dict) -> dict:
    recon = results["tensor_reconstruction"]["regular_activation"]["metrics"]
    regular = results["regular_matmul_output"]["metrics"]
    grouped = results["grouped_qkv_like_matmul_output"]["metrics"]
    return {
        "qdq_rms_localcta": recon["localcta_vs_bf16"]["rms"],
        "qdq_rms_baseline_v5": recon["baseline_v5_vs_bf16"]["rms"],
        "qdq_rms_ratio": (
            recon["localcta_vs_bf16"]["rms"] /
            recon["baseline_v5_vs_bf16"]["rms"]
        ),
        "regular_rms_localcta": regular["localcta_prepared_vs_bf16"]["rms"],
        "regular_rms_baseline_v5": regular["baseline_v5_vs_bf16"]["rms"],
        "regular_rms_ratio": (
            regular["localcta_prepared_vs_bf16"]["rms"] /
            regular["baseline_v5_vs_bf16"]["rms"]
        ),
        "grouped_rms_localcta": grouped["localcta_prepared_vs_bf16"]["rms"],
        "grouped_rms_baseline_v5": grouped["baseline_v5_vs_bf16"]["rms"],
        "grouped_rms_ratio": (
            grouped["localcta_prepared_vs_bf16"]["rms"] /
            grouped["baseline_v5_vs_bf16"]["rms"]
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = numerics.build_parser()
    parser.description = "Sweep localCTA global scale numerator across distributions"
    parser.add_argument(
        "--scale-nums",
        type=parse_float_list,
        default=parse_float_list("1344,1680,2016,2352,2688,3024,3360"),
    )
    parser.add_argument(
        "--scenarios",
        type=lambda s: [piece.strip() for piece in s.split(",") if piece.strip()],
        default=list(SCENARIOS.keys()),
    )
    parser.add_argument("--summary-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario}")

    records: list[dict] = []
    best_by_scenario: dict[str, dict] = {}

    for scenario in args.scenarios:
        for scale_num in args.scale_nums:
            run_args = apply_scenario(args, scenario)
            run_args.localcta_global_scale_num = scale_num
            results = numerics.run_experiment(run_args)
            summary = summarize_result(results)
            record = {
                "scenario": scenario,
                "scale_num": scale_num,
                "summary": summary,
                "config": results["config"],
            }
            records.append(record)
            print(
                f"scenario={scenario:24s} scale_num={scale_num:8.1f} "
                f"qdq_rms_ratio={summary['qdq_rms_ratio']:.4f} "
                f"regular_rms_ratio={summary['regular_rms_ratio']:.4f} "
                f"grouped_rms_ratio={summary['grouped_rms_ratio']:.4f}"
            )

        scenario_records = [record for record in records if record["scenario"] == scenario]
        scenario_best = min(
            scenario_records,
            key=lambda record: (
                record["summary"]["regular_rms_ratio"] +
                record["summary"]["grouped_rms_ratio"] +
                record["summary"]["qdq_rms_ratio"]
            ),
        )
        best_by_scenario[scenario] = scenario_best

    print("\nBest by scenario:")
    for scenario in args.scenarios:
        best = best_by_scenario[scenario]
        summary = best["summary"]
        print(
            f"{scenario:24s} scale_num={best['scale_num']:8.1f} "
            f"qdq={summary['qdq_rms_ratio']:.4f} "
            f"regular={summary['regular_rms_ratio']:.4f} "
            f"grouped={summary['grouped_rms_ratio']:.4f}"
        )

    if args.summary_json:
        out = {
            "records": records,
            "best_by_scenario": best_by_scenario,
        }
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"\nsummary_json={path}")

    torch.cuda.synchronize()
    numerics.local_q.tk_localcta_reset_global_scale_num()


if __name__ == "__main__":
    main()
