import argparse
import copy
import json
from pathlib import Path

import torch

import test_localcta_numerics as numerics


SCENARIOS = {
    "normal_both": {},
    "laplace_both": {"distribution": "laplace"},
    "student_t_both": {"distribution": "student_t"},
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


FAMILIES = {
    "regular_ffn_small": {"kind": "regular", "M": 4096, "N": 5632, "K": 2048},
    "regular_ffn_large": {"kind": "regular", "M": 16384, "N": 5632, "K": 2048},
    "grouped_qkv_small": {"kind": "grouped", "M": 4096, "K": 2048, "splits": [2048, 1024, 1024]},
    "grouped_qkv_large": {"kind": "grouped", "M": 16384, "K": 2048, "splits": [2048, 1024, 1024]},
    "batched_square": {"kind": "batched", "M": 1024, "N": 1024, "K": 1024, "batches": 2},
    "batched_tall": {"kind": "batched", "M": 1024, "N": 8192, "K": 8192, "batches": 2},
}


def parse_int_list(value: str) -> list[int]:
    parts = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not parts:
        raise ValueError("expected at least one integer")
    return [int(piece) for piece in parts]


def rms(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.to(torch.float32) - b.to(torch.float32)
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def apply_scenario(args: argparse.Namespace, scenario: str) -> argparse.Namespace:
    out = copy.deepcopy(args)
    for key, value in SCENARIOS[scenario].items():
        setattr(out, key, value)
    return out


def resolve_distributions(args: argparse.Namespace) -> tuple[str, str]:
    return (
        numerics.resolve_distribution(args, args.regular_activation_dist),
        numerics.resolve_distribution(args, args.regular_weight_dist),
    )


def make_tensor(rows: int, cols: int, *, k_dim: int, dist: str, args: argparse.Namespace) -> torch.Tensor:
    return numerics.make_tensor(
        rows, cols,
        k_dim=k_dim, device="cuda", dist=dist,
        tensor_scale=args.tensor_scale,
        sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
        student_t_df=args.student_t_df,
        row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
        chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
    )


def prepare_regular(args: argparse.Namespace, cfg: dict) -> dict:
    act_dist, wt_dist = resolve_distributions(args)
    A = make_tensor(cfg["M"], cfg["K"], k_dim=cfg["K"], dist=act_dist, args=args)
    B = make_tensor(cfg["N"], cfg["K"], k_dim=cfg["K"], dist=wt_dist, args=args)
    ref = torch.matmul(A, B.t()).to(torch.bfloat16)

    A_v5 = numerics.q_v5.tk_quantize_for_gemm(A, False, True)
    B_v5 = numerics.q_v5.tk_quantize_for_gemm(B, False, True)
    A_v5_recon = numerics.local_q.tk_localcta_reconstruct_row(
        A_v5[0], A_v5[1], numerics.constant_chunk_grid(A.size(0), A.size(1), A_v5[4]),
    )
    B_v5_recon = numerics.local_q.tk_localcta_reconstruct_row(
        B_v5[0], B_v5[1], numerics.constant_chunk_grid(B.size(0), B.size(1), B_v5[4]),
    )
    out_v5 = torch.empty_like(ref)
    numerics.legacy_gemm.nvfp4_gemm(
        A_v5[0], A_v5[1], A_v5[4],
        B_v5[0], B_v5[1], B_v5[4],
        out_v5,
    )
    torch.cuda.synchronize()

    return {
        "A": A,
        "B": B,
        "ref": ref,
        "baseline_qdq_A_rms": rms(A_v5_recon, A),
        "baseline_qdq_B_rms": rms(B_v5_recon, B),
        "baseline_out_rms": rms(out_v5, ref),
    }


def eval_regular(scale_num: int, payload: dict) -> dict:
    numerics.local_q.tk_localcta_set_global_scale_num(float(scale_num))
    A_fast = numerics.local_q.tk_localcta_quantize_for_gemm_fast(payload["A"], False, True)
    B_fast = numerics.local_q.tk_localcta_quantize_for_gemm_fast(payload["B"], False, True)
    A_recon = numerics.local_q.tk_localcta_reconstruct_row(A_fast[0], A_fast[1], A_fast[4])
    B_recon = numerics.local_q.tk_localcta_reconstruct_row(B_fast[0], B_fast[1], B_fast[4])

    A_prepared = numerics.local_q.tk_localcta_quantize_for_gemm_prepared(payload["A"], False, True)
    B_prepared = numerics.local_q.tk_localcta_quantize_for_gemm_prepared(payload["B"], False, True)
    out = torch.empty_like(payload["ref"])
    numerics.local_gemm.nvfp4_localcta_fast_gemm(
        A_prepared[0], A_prepared[1],
        B_prepared[0], B_prepared[1],
        out,
    )
    torch.cuda.synchronize()
    result = {
        "qdq_A_ratio": rms(A_recon, payload["A"]) / payload["baseline_qdq_A_rms"],
        "qdq_B_ratio": rms(B_recon, payload["B"]) / payload["baseline_qdq_B_rms"],
        "out_ratio": rms(out, payload["ref"]) / payload["baseline_out_rms"],
    }
    result["score"] = (result["qdq_A_ratio"] + result["qdq_B_ratio"] + result["out_ratio"]) / 3.0
    return result


def prepare_grouped(args: argparse.Namespace, cfg: dict) -> dict:
    act_dist, wt_dist = resolve_distributions(args)
    A = make_tensor(cfg["M"], cfg["K"], k_dim=cfg["K"], dist=act_dist, args=args)
    W = make_tensor(sum(cfg["splits"]), cfg["K"], k_dim=cfg["K"], dist=wt_dist, args=args)
    ref = torch.matmul(A, W.t()).to(torch.bfloat16)

    A_v5 = numerics.q_v5.tk_quantize_for_gemm(A, False, True)
    W_v5 = numerics.q_v5.tk_group_quantize_for_gemm(W, cfg["splits"])
    out_v5 = torch.empty_like(ref)
    numerics.legacy_gemm.nvfp4_grouped_gemm(
        A_v5[0], A_v5[1], A_v5[4],
        W_v5[0], W_v5[1], W_v5[2],
        out_v5,
    )
    torch.cuda.synchronize()
    return {
        "A": A,
        "W": W,
        "splits": cfg["splits"],
        "ref": ref,
        "baseline_out_rms": rms(out_v5, ref),
    }


def eval_grouped(scale_num: int, payload: dict) -> dict:
    numerics.local_q.tk_localcta_set_global_scale_num(float(scale_num))
    A_prepared = numerics.local_q.tk_localcta_quantize_for_gemm_prepared(payload["A"], False, True)
    W_prepared = numerics.local_q.tk_localcta_group_quantize_for_gemm_prepared(payload["W"], payload["splits"])
    out = torch.empty_like(payload["ref"])
    numerics.local_gemm.nvfp4_localcta_fast_grouped_gemm(
        A_prepared[0], A_prepared[1],
        W_prepared[0], W_prepared[1],
        out,
    )
    torch.cuda.synchronize()
    out_ratio = rms(out, payload["ref"]) / payload["baseline_out_rms"]
    return {"out_ratio": out_ratio, "score": out_ratio}


def prepare_batched(args: argparse.Namespace, cfg: dict) -> dict:
    act_dist, wt_dist = resolve_distributions(args)
    A_list = [make_tensor(cfg["M"], cfg["K"], k_dim=cfg["K"], dist=act_dist, args=args) for _ in range(cfg["batches"])]
    B_list = [make_tensor(cfg["N"], cfg["K"], k_dim=cfg["K"], dist=wt_dist, args=args) for _ in range(cfg["batches"])]
    refs = [torch.matmul(A, B.t()).to(torch.bfloat16) for A, B in zip(A_list, B_list)]
    ref_accum = torch.zeros_like(refs[0])
    for ref in refs:
        ref_accum.add_(ref)

    A_v5 = [numerics.q_v5.tk_quantize_for_gemm(A, False, True) for A in A_list]
    B_v5 = [numerics.q_v5.tk_quantize_for_gemm(B, False, True) for B in B_list]
    outs_v5 = [torch.empty_like(ref) for ref in refs]
    numerics.legacy_gemm.nvfp4_batched_gemm(
        [x[0] for x in A_v5], [x[1] for x in A_v5], [x[4] for x in A_v5],
        [x[0] for x in B_v5], [x[1] for x in B_v5], [x[4] for x in B_v5],
        outs_v5,
    )
    out_accum_v5 = torch.empty_like(ref_accum)
    numerics.legacy_gemm.nvfp4_batched_accum_gemm(
        [x[0] for x in A_v5], [x[1] for x in A_v5], [x[4] for x in A_v5],
        [x[0] for x in B_v5], [x[1] for x in B_v5], [x[4] for x in B_v5],
        out_accum_v5,
    )
    torch.cuda.synchronize()

    baseline_batched = sum(rms(out, ref) for out, ref in zip(outs_v5, refs)) / len(refs)
    baseline_accum = rms(out_accum_v5, ref_accum)
    return {
        "A_list": A_list,
        "B_list": B_list,
        "refs": refs,
        "ref_accum": ref_accum,
        "baseline_batched_rms": baseline_batched,
        "baseline_accum_rms": baseline_accum,
    }


def eval_batched(scale_num: int, payload: dict) -> dict:
    numerics.local_q.tk_localcta_set_global_scale_num(float(scale_num))
    A_prepared = [numerics.local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True) for A in payload["A_list"]]
    B_prepared = [numerics.local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True) for B in payload["B_list"]]
    outs = [torch.empty_like(ref) for ref in payload["refs"]]
    numerics.local_gemm.nvfp4_localcta_fast_batched_gemm(
        [x[0] for x in A_prepared], [x[1] for x in A_prepared],
        [x[0] for x in B_prepared], [x[1] for x in B_prepared],
        outs,
    )
    out_accum = torch.empty_like(payload["ref_accum"])
    numerics.local_gemm.nvfp4_localcta_fast_batched_accum_gemm(
        [x[0] for x in A_prepared], [x[1] for x in A_prepared],
        [x[0] for x in B_prepared], [x[1] for x in B_prepared],
        out_accum,
    )
    torch.cuda.synchronize()

    batched_ratio = (
        sum(rms(out, ref) for out, ref in zip(outs, payload["refs"])) / len(payload["refs"])
    ) / payload["baseline_batched_rms"]
    accum_ratio = rms(out_accum, payload["ref_accum"]) / payload["baseline_accum_rms"]
    return {
        "batched_ratio": batched_ratio,
        "accum_ratio": accum_ratio,
        "score": 0.5 * (batched_ratio + accum_ratio),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = numerics.build_parser()
    parser.description = "Global candidate search for localCTA scale number"
    parser.add_argument("--scale-nums", type=parse_int_list, default=parse_int_list("1450,1457,1493,1536,1559"))
    parser.add_argument(
        "--scenarios",
        type=lambda s: [piece.strip() for piece in s.split(",") if piece.strip()],
        default=list(SCENARIOS.keys()),
    )
    parser.add_argument(
        "--families",
        type=lambda s: [piece.strip() for piece in s.split(",") if piece.strip()],
        default=list(FAMILIES.keys()),
    )
    parser.add_argument("--summary-json", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario}")
    for family in args.families:
        if family not in FAMILIES:
            raise ValueError(f"unknown family {family}")

    results: list[dict] = []
    aggregate_by_scale: dict[int, list[float]] = {scale: [] for scale in args.scale_nums}

    for scenario in args.scenarios:
        scenario_args = apply_scenario(args, scenario)
        for family_name in args.families:
            family = FAMILIES[family_name]
            if family["kind"] == "regular":
                baseline = prepare_regular(scenario_args, family)
                for scale in args.scale_nums:
                    rec = eval_regular(scale, baseline)
                    results.append({"scenario": scenario, "family": family_name, "scale_num": scale, **rec})
                    aggregate_by_scale[scale].append(rec["score"])
                    print(
                        f"scenario={scenario:24s} family={family_name:18s} scale={scale:5d} "
                        f"qdqA={rec['qdq_A_ratio']:.4f} qdqB={rec['qdq_B_ratio']:.4f} "
                        f"out={rec['out_ratio']:.4f} score={rec['score']:.4f}"
                    )
            elif family["kind"] == "grouped":
                baseline = prepare_grouped(scenario_args, family)
                for scale in args.scale_nums:
                    rec = eval_grouped(scale, baseline)
                    results.append({"scenario": scenario, "family": family_name, "scale_num": scale, **rec})
                    aggregate_by_scale[scale].append(rec["score"])
                    print(
                        f"scenario={scenario:24s} family={family_name:18s} scale={scale:5d} "
                        f"out={rec['out_ratio']:.4f} score={rec['score']:.4f}"
                    )
            else:
                baseline = prepare_batched(scenario_args, family)
                for scale in args.scale_nums:
                    rec = eval_batched(scale, baseline)
                    results.append({"scenario": scenario, "family": family_name, "scale_num": scale, **rec})
                    aggregate_by_scale[scale].append(rec["score"])
                    print(
                        f"scenario={scenario:24s} family={family_name:18s} scale={scale:5d} "
                        f"batched={rec['batched_ratio']:.4f} accum={rec['accum_ratio']:.4f} score={rec['score']:.4f}"
                    )
            torch.cuda.empty_cache()

    ranking = sorted(
        ((sum(scores) / len(scores), scale) for scale, scores in aggregate_by_scale.items()),
        key=lambda pair: pair[0],
    )

    print("\nGlobal ranking:")
    for avg_score, scale in ranking:
        print(f"scale={scale:5d} avg_score={avg_score:.6f}")

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"results": results, "ranking": ranking}, indent=2))
        print(f"\nsummary_json={path}")

    numerics.local_q.tk_localcta_reset_global_scale_num()


if __name__ == "__main__":
    main()
