import argparse
import copy
import json
from pathlib import Path

import torch

import test_localcta_numerics as numerics


KEY_SCENARIOS = {
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


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.to(torch.float32) - b.to(torch.float32)
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def apply_scenario(args: argparse.Namespace, scenario: str) -> argparse.Namespace:
    cfg = copy.deepcopy(args)
    for key, value in KEY_SCENARIOS[scenario].items():
        setattr(cfg, key, value)
    return cfg


def resolve_distributions(args: argparse.Namespace) -> tuple[str, str, str, str]:
    return (
        numerics.resolve_distribution(args, args.regular_activation_dist),
        numerics.resolve_distribution(args, args.regular_weight_dist),
        numerics.resolve_distribution(args, args.grouped_activation_dist),
        numerics.resolve_distribution(args, args.grouped_weight_dist),
    )


def prepare_baselines(args: argparse.Namespace) -> dict:
    (
        regular_activation_dist,
        regular_weight_dist,
        grouped_activation_dist,
        grouped_weight_dist,
    ) = resolve_distributions(args)

    reg_A = numerics.make_tensor(
        args.regular_m, args.regular_k,
        k_dim=args.regular_k, device="cuda", dist=regular_activation_dist,
        tensor_scale=args.tensor_scale,
        sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
        student_t_df=args.student_t_df,
        row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
        chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
    )
    reg_B = numerics.make_tensor(
        args.regular_n, args.regular_k,
        k_dim=args.regular_k, device="cuda", dist=regular_weight_dist,
        tensor_scale=args.tensor_scale,
        sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
        student_t_df=args.student_t_df,
        row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
        chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
    )
    grp_A = numerics.make_tensor(
        args.grouped_m, args.grouped_k,
        k_dim=args.grouped_k, device="cuda", dist=grouped_activation_dist,
        tensor_scale=args.tensor_scale,
        sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
        student_t_df=args.student_t_df,
        row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
        chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
    )
    grp_W = numerics.make_tensor(
        sum(args.grouped_splits), args.grouped_k,
        k_dim=args.grouped_k, device="cuda", dist=grouped_weight_dist,
        tensor_scale=args.tensor_scale,
        sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
        student_t_df=args.student_t_df,
        row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
        chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
    )

    reg_ref = torch.matmul(reg_A, reg_B.t()).to(torch.bfloat16)
    grp_ref = torch.matmul(grp_A, grp_W.t()).to(torch.bfloat16)

    reg_A_v5 = numerics.q_v5.tk_quantize_for_gemm(reg_A, False, True)
    reg_B_v5 = numerics.q_v5.tk_quantize_for_gemm(reg_B, False, True)
    grp_A_v5 = numerics.q_v5.tk_quantize_for_gemm(grp_A, False, True)
    grp_W_v5 = numerics.q_v5.tk_group_quantize_for_gemm(grp_W, args.grouped_splits)

    reg_A_v5_recon = numerics.local_q.tk_localcta_reconstruct_row(
        reg_A_v5[0], reg_A_v5[1],
        numerics.constant_chunk_grid(reg_A.size(0), reg_A.size(1), reg_A_v5[4]),
    )
    reg_B_v5_recon = numerics.local_q.tk_localcta_reconstruct_row(
        reg_B_v5[0], reg_B_v5[1],
        numerics.constant_chunk_grid(reg_B.size(0), reg_B.size(1), reg_B_v5[4]),
    )

    reg_out_v5 = torch.empty(reg_A.size(0), reg_B.size(0), dtype=torch.bfloat16, device="cuda")
    grp_out_v5 = torch.empty(grp_A.size(0), grp_W.size(0), dtype=torch.bfloat16, device="cuda")
    numerics.legacy_gemm.nvfp4_gemm(
        reg_A_v5[0], reg_A_v5[1], reg_A_v5[4],
        reg_B_v5[0], reg_B_v5[1], reg_B_v5[4],
        reg_out_v5,
    )
    numerics.legacy_gemm.nvfp4_grouped_gemm(
        grp_A_v5[0], grp_A_v5[1], grp_A_v5[4],
        grp_W_v5[0], grp_W_v5[1], grp_W_v5[2],
        grp_out_v5,
    )
    torch.cuda.synchronize()

    baselines = {
        "reg_A": reg_A,
        "reg_B": reg_B,
        "grp_A": grp_A,
        "grp_W": grp_W,
        "reg_ref": reg_ref,
        "grp_ref": grp_ref,
        "baseline_qdq_reg_A_rms": rms(reg_A_v5_recon, reg_A),
        "baseline_qdq_reg_B_rms": rms(reg_B_v5_recon, reg_B),
        "baseline_reg_rms": rms(reg_out_v5, reg_ref),
        "baseline_grp_rms": rms(grp_out_v5, grp_ref),
    }

    del reg_A_v5, reg_B_v5, grp_A_v5, grp_W_v5, reg_A_v5_recon, reg_B_v5_recon, reg_out_v5, grp_out_v5
    torch.cuda.empty_cache()
    return baselines


def evaluate_constant(scale_num: int, baselines: dict, splits: list[int]) -> dict:
    numerics.local_q.tk_localcta_set_global_scale_num(float(scale_num))

    reg_A_fast = numerics.local_q.tk_localcta_quantize_for_gemm_fast(baselines["reg_A"], False, True)
    reg_B_fast = numerics.local_q.tk_localcta_quantize_for_gemm_fast(baselines["reg_B"], False, True)
    reg_A_recon = numerics.local_q.tk_localcta_reconstruct_row(reg_A_fast[0], reg_A_fast[1], reg_A_fast[4])
    reg_B_recon = numerics.local_q.tk_localcta_reconstruct_row(reg_B_fast[0], reg_B_fast[1], reg_B_fast[4])

    reg_A_prepared = numerics.local_q.tk_localcta_quantize_for_gemm_prepared(baselines["reg_A"], False, True)
    reg_B_prepared = numerics.local_q.tk_localcta_quantize_for_gemm_prepared(baselines["reg_B"], False, True)
    grp_A_prepared = numerics.local_q.tk_localcta_quantize_for_gemm_prepared(baselines["grp_A"], False, True)
    grp_W_prepared = numerics.local_q.tk_localcta_group_quantize_for_gemm_prepared(baselines["grp_W"], splits)

    reg_out = torch.empty_like(baselines["reg_ref"])
    grp_out = torch.empty_like(baselines["grp_ref"])
    numerics.local_gemm.nvfp4_localcta_fast_gemm(
        reg_A_prepared[0], reg_A_prepared[1],
        reg_B_prepared[0], reg_B_prepared[1],
        reg_out,
    )
    numerics.local_gemm.nvfp4_localcta_fast_grouped_gemm(
        grp_A_prepared[0], grp_A_prepared[1],
        grp_W_prepared[0], grp_W_prepared[1],
        grp_out,
    )
    torch.cuda.synchronize()

    qdq_reg_A_rms = rms(reg_A_recon, baselines["reg_A"])
    qdq_reg_B_rms = rms(reg_B_recon, baselines["reg_B"])
    reg_rms = rms(reg_out, baselines["reg_ref"])
    grp_rms = rms(grp_out, baselines["grp_ref"])

    result = {
        "scale_num": scale_num,
        "qdq_reg_A_rms_ratio": qdq_reg_A_rms / baselines["baseline_qdq_reg_A_rms"],
        "qdq_reg_B_rms_ratio": qdq_reg_B_rms / baselines["baseline_qdq_reg_B_rms"],
        "regular_rms_ratio": reg_rms / baselines["baseline_reg_rms"],
        "grouped_rms_ratio": grp_rms / baselines["baseline_grp_rms"],
    }
    result["score"] = (
        result["qdq_reg_A_rms_ratio"] +
        result["qdq_reg_B_rms_ratio"] +
        result["regular_rms_ratio"] +
        result["grouped_rms_ratio"]
    ) / 4.0

    del reg_A_fast, reg_B_fast, reg_A_recon, reg_B_recon
    del reg_A_prepared, reg_B_prepared, grp_A_prepared, grp_W_prepared, reg_out, grp_out
    torch.cuda.empty_cache()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = numerics.build_parser()
    parser.description = "Dense localCTA scale-number search with cached baselines"
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument(
        "--scenarios",
        type=lambda s: [piece.strip() for piece in s.split(",") if piece.strip()],
        default=list(KEY_SCENARIOS.keys()),
    )
    parser.add_argument("--summary-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for scenario in args.scenarios:
        if scenario not in KEY_SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")

    all_results: dict[str, list[dict]] = {}
    best_by_scenario: dict[str, dict] = {}
    for scenario in args.scenarios:
        scenario_args = apply_scenario(args, scenario)
        baselines = prepare_baselines(scenario_args)
        records: list[dict] = []
        for scale_num in range(args.start, args.end + 1):
            rec = evaluate_constant(scale_num, baselines, scenario_args.grouped_splits)
            records.append(rec)
            if scale_num == args.start or scale_num == args.end or (scale_num - args.start) % 16 == 0:
                print(
                    f"scenario={scenario:24s} scale_num={scale_num:5d} "
                    f"qdqA={rec['qdq_reg_A_rms_ratio']:.4f} "
                    f"qdqB={rec['qdq_reg_B_rms_ratio']:.4f} "
                    f"reg={rec['regular_rms_ratio']:.4f} "
                    f"grp={rec['grouped_rms_ratio']:.4f} "
                    f"score={rec['score']:.4f}"
                )
        best = min(records, key=lambda rec: rec["score"])
        all_results[scenario] = records
        best_by_scenario[scenario] = best
        print(
            f"BEST scenario={scenario:24s} scale_num={best['scale_num']:5d} "
            f"qdqA={best['qdq_reg_A_rms_ratio']:.4f} qdqB={best['qdq_reg_B_rms_ratio']:.4f} "
            f"reg={best['regular_rms_ratio']:.4f} grp={best['grouped_rms_ratio']:.4f} "
            f"score={best['score']:.4f}"
        )
        del baselines
        torch.cuda.empty_cache()

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"results": all_results, "best_by_scenario": best_by_scenario}, indent=2))
        print(f"summary_json={path}")

    numerics.local_q.tk_localcta_reset_global_scale_num()


if __name__ == "__main__":
    main()
