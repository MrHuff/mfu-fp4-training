import argparse
import json
from pathlib import Path

import torch

import test_localcta_numerics as base

ROOT = Path(__file__).resolve().parent
V5_ROOT = ROOT.parent / "nvfp4_v5"

import _tk_quant_localcta as local_q  # type: ignore
import _tk_quant_v5 as q_v5  # type: ignore


def quantize_local_raw(tensor: torch.Tensor) -> dict:
    q = local_q.tk_localcta_quantize_for_gemm_fast(tensor, False, True)
    return {
        "fp4": q[0],
        "sc": q[1],
        "sg_grid": q[4],
    }


def quantize_v5_raw(tensor: torch.Tensor) -> dict:
    q = q_v5.tk_quantize_for_gemm(tensor, False, True)
    sg_grid = base.constant_chunk_grid(tensor.size(0), tensor.size(1), q[4])
    return {
        "fp4": q[0],
        "sc": q[1],
        "sg_grid": sg_grid,
        "sg_scalar": q[4],
    }


def reconstruct_variant(fp4: torch.Tensor, sc: torch.Tensor, sg_grid: torch.Tensor) -> torch.Tensor:
    return local_q.tk_localcta_reconstruct_row(fp4, sc, sg_grid)


def compare_variants(
    variants: dict[str, torch.Tensor],
    ref: torch.Tensor,
    *,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
) -> dict:
    metrics = {
        name: base.chunked_error_metrics(
            tensor,
            ref,
            epsilon=epsilon,
            top_k=top_k,
            chunk_elems=chunk_elems,
            max_quantile_samples=max_quantile_samples,
        )
        for name, tensor in variants.items()
    }
    summary = {}
    if "localcta_native" in metrics and "baseline_v5_native" in metrics:
        summary["localcta_native_over_baseline_rms"] = (
            metrics["localcta_native"]["rms"] / max(metrics["baseline_v5_native"]["rms"], 1e-12)
        )
        summary["localcta_native_over_baseline_p99_abs"] = (
            metrics["localcta_native"]["p99_abs"] / max(metrics["baseline_v5_native"]["p99_abs"], 1e-12)
        )
    if "localcta_with_baseline_sg" in metrics:
        summary["localcta_sg_swap_rms_delta"] = (
            metrics["localcta_with_baseline_sg"]["rms"] - metrics["localcta_native"]["rms"]
        )
        summary["localcta_sg_swap_p99_abs_delta"] = (
            metrics["localcta_with_baseline_sg"]["p99_abs"] - metrics["localcta_native"]["p99_abs"]
        )
    if "localcta_scaled_sg" in metrics:
        summary["localcta_scaled_sg_rms_delta"] = (
            metrics["localcta_scaled_sg"]["rms"] - metrics["localcta_native"]["rms"]
        )
        summary["localcta_scaled_sg_p99_abs_delta"] = (
            metrics["localcta_scaled_sg"]["p99_abs"] - metrics["localcta_native"]["p99_abs"]
        )
    if "baseline_v5_with_localcta_sg" in metrics:
        summary["baseline_sg_swap_rms_delta"] = (
            metrics["baseline_v5_with_localcta_sg"]["rms"] - metrics["baseline_v5_native"]["rms"]
        )
        summary["baseline_sg_swap_p99_abs_delta"] = (
            metrics["baseline_v5_with_localcta_sg"]["p99_abs"] - metrics["baseline_v5_native"]["p99_abs"]
        )
    return {"metrics": metrics, "summary": summary}


def tensor_ablation_section(
    name: str,
    tensor: torch.Tensor,
    *,
    distribution: str,
    localcta_sg_rescale: float,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
) -> dict:
    local_raw = quantize_local_raw(tensor)
    v5_raw = quantize_v5_raw(tensor)
    scaled_local_sg = local_raw["sg_grid"] * localcta_sg_rescale

    variants = {
        "localcta_native": reconstruct_variant(local_raw["fp4"], local_raw["sc"], local_raw["sg_grid"]),
        "localcta_scaled_sg": reconstruct_variant(local_raw["fp4"], local_raw["sc"], scaled_local_sg),
        "localcta_with_baseline_sg": reconstruct_variant(local_raw["fp4"], local_raw["sc"], v5_raw["sg_grid"]),
        "baseline_v5_native": reconstruct_variant(v5_raw["fp4"], v5_raw["sc"], v5_raw["sg_grid"]),
        "baseline_v5_with_localcta_sg": reconstruct_variant(v5_raw["fp4"], v5_raw["sc"], local_raw["sg_grid"]),
    }
    torch.cuda.synchronize()

    result = compare_variants(
        variants,
        tensor,
        epsilon=epsilon,
        top_k=top_k,
        chunk_elems=chunk_elems,
        max_quantile_samples=max_quantile_samples,
    )
    result.update(
        {
            "name": name,
            "shape": list(tensor.shape),
            "distribution": distribution,
            "input_summary": base.tensor_input_summary(tensor, max_quantile_samples=max_quantile_samples),
            "localcta_chunk_scale_summary": base.scale_summary_localcta(local_raw["sg_grid"]),
            "localcta_scaled_chunk_scale_summary": base.scale_summary_localcta(scaled_local_sg),
            "baseline_v5_scale_summary": base.scale_summary_baseline(v5_raw["sg_scalar"]),
        }
    )

    del local_raw, v5_raw
    for tensor_variant in variants.values():
        del tensor_variant
    torch.cuda.empty_cache()
    return result


def build_regular_matmul_ablation(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    localcta_sg_rescale: float,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
) -> dict:
    A_local = quantize_local_raw(A)
    B_local = quantize_local_raw(B)
    A_v5 = quantize_v5_raw(A)
    B_v5 = quantize_v5_raw(B)
    A_scaled_sg = A_local["sg_grid"] * localcta_sg_rescale
    B_scaled_sg = B_local["sg_grid"] * localcta_sg_rescale

    A_variants = {
        "localcta_native": reconstruct_variant(A_local["fp4"], A_local["sc"], A_local["sg_grid"]),
        "localcta_scaled_sg": reconstruct_variant(A_local["fp4"], A_local["sc"], A_scaled_sg),
        "localcta_with_baseline_sg": reconstruct_variant(A_local["fp4"], A_local["sc"], A_v5["sg_grid"]),
        "baseline_v5_native": reconstruct_variant(A_v5["fp4"], A_v5["sc"], A_v5["sg_grid"]),
        "baseline_v5_with_localcta_sg": reconstruct_variant(A_v5["fp4"], A_v5["sc"], A_local["sg_grid"]),
    }
    B_variants = {
        "localcta_native": reconstruct_variant(B_local["fp4"], B_local["sc"], B_local["sg_grid"]),
        "localcta_scaled_sg": reconstruct_variant(B_local["fp4"], B_local["sc"], B_scaled_sg),
        "localcta_with_baseline_sg": reconstruct_variant(B_local["fp4"], B_local["sc"], B_v5["sg_grid"]),
        "baseline_v5_native": reconstruct_variant(B_v5["fp4"], B_v5["sc"], B_v5["sg_grid"]),
        "baseline_v5_with_localcta_sg": reconstruct_variant(B_v5["fp4"], B_v5["sc"], B_local["sg_grid"]),
    }

    combos = {
        "localcta_native": ("localcta_native", "localcta_native"),
        "localcta_scaled_sg": ("localcta_scaled_sg", "localcta_scaled_sg"),
        "localcta_with_baseline_sg": ("localcta_with_baseline_sg", "localcta_with_baseline_sg"),
        "baseline_v5_native": ("baseline_v5_native", "baseline_v5_native"),
        "baseline_v5_with_localcta_sg": ("baseline_v5_with_localcta_sg", "baseline_v5_with_localcta_sg"),
        "localcta_scaled_A_only": ("localcta_scaled_sg", "localcta_native"),
        "localcta_scaled_B_only": ("localcta_native", "localcta_scaled_sg"),
        "localcta_A_swap_only": ("localcta_with_baseline_sg", "localcta_native"),
        "localcta_B_swap_only": ("localcta_native", "localcta_with_baseline_sg"),
        "baseline_A_swap_only": ("baseline_v5_with_localcta_sg", "baseline_v5_native"),
        "baseline_B_swap_only": ("baseline_v5_native", "baseline_v5_with_localcta_sg"),
    }

    ref = torch.matmul(A, B.t()).to(torch.bfloat16)
    outputs = {}
    for name, (a_key, b_key) in combos.items():
        outputs[name] = torch.matmul(A_variants[a_key], B_variants[b_key].t()).to(torch.bfloat16)
    torch.cuda.synchronize()

    result = compare_variants(
        outputs,
        ref,
        epsilon=epsilon,
        top_k=top_k,
        chunk_elems=chunk_elems,
        max_quantile_samples=max_quantile_samples,
    )
    result["shape"] = [A.size(0), B.size(0)]
    result["inputs"] = {"A": list(A.shape), "B": list(B.shape)}
    result["combo_definitions"] = {k: list(v) for k, v in combos.items()}

    del A_local, B_local, A_v5, B_v5, ref
    for tensor_variant in A_variants.values():
        del tensor_variant
    for tensor_variant in B_variants.values():
        del tensor_variant
    for out in outputs.values():
        del out
    torch.cuda.empty_cache()
    return result


def build_grouped_matmul_ablation(
    A: torch.Tensor,
    W: torch.Tensor,
    splits: list[int],
    *,
    localcta_sg_rescale: float,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
) -> dict:
    A_local = quantize_local_raw(A)
    W_local = quantize_local_raw(W)
    A_v5 = quantize_v5_raw(A)
    W_v5 = quantize_v5_raw(W)
    A_scaled_sg = A_local["sg_grid"] * localcta_sg_rescale
    W_scaled_sg = W_local["sg_grid"] * localcta_sg_rescale

    A_variants = {
        "localcta_native": reconstruct_variant(A_local["fp4"], A_local["sc"], A_local["sg_grid"]),
        "localcta_scaled_sg": reconstruct_variant(A_local["fp4"], A_local["sc"], A_scaled_sg),
        "localcta_with_baseline_sg": reconstruct_variant(A_local["fp4"], A_local["sc"], A_v5["sg_grid"]),
        "baseline_v5_native": reconstruct_variant(A_v5["fp4"], A_v5["sc"], A_v5["sg_grid"]),
        "baseline_v5_with_localcta_sg": reconstruct_variant(A_v5["fp4"], A_v5["sc"], A_local["sg_grid"]),
    }
    W_variants = {
        "localcta_native": reconstruct_variant(W_local["fp4"], W_local["sc"], W_local["sg_grid"]),
        "localcta_scaled_sg": reconstruct_variant(W_local["fp4"], W_local["sc"], W_scaled_sg),
        "localcta_with_baseline_sg": reconstruct_variant(W_local["fp4"], W_local["sc"], W_v5["sg_grid"]),
        "baseline_v5_native": reconstruct_variant(W_v5["fp4"], W_v5["sc"], W_v5["sg_grid"]),
        "baseline_v5_with_localcta_sg": reconstruct_variant(W_v5["fp4"], W_v5["sc"], W_local["sg_grid"]),
    }

    combos = {
        "localcta_native": ("localcta_native", "localcta_native"),
        "localcta_scaled_sg": ("localcta_scaled_sg", "localcta_scaled_sg"),
        "localcta_with_baseline_sg": ("localcta_with_baseline_sg", "localcta_with_baseline_sg"),
        "baseline_v5_native": ("baseline_v5_native", "baseline_v5_native"),
        "baseline_v5_with_localcta_sg": ("baseline_v5_with_localcta_sg", "baseline_v5_with_localcta_sg"),
        "localcta_scaled_A_only": ("localcta_scaled_sg", "localcta_native"),
        "localcta_scaled_B_only": ("localcta_native", "localcta_scaled_sg"),
        "localcta_A_swap_only": ("localcta_with_baseline_sg", "localcta_native"),
        "localcta_B_swap_only": ("localcta_native", "localcta_with_baseline_sg"),
        "baseline_A_swap_only": ("baseline_v5_with_localcta_sg", "baseline_v5_native"),
        "baseline_B_swap_only": ("baseline_v5_native", "baseline_v5_with_localcta_sg"),
    }

    ref = torch.matmul(A, W.t()).to(torch.bfloat16)
    outputs = {}
    for name, (a_key, w_key) in combos.items():
        outputs[name] = torch.matmul(A_variants[a_key], W_variants[w_key].t()).to(torch.bfloat16)
    torch.cuda.synchronize()

    result = compare_variants(
        outputs,
        ref,
        epsilon=epsilon,
        top_k=top_k,
        chunk_elems=chunk_elems,
        max_quantile_samples=max_quantile_samples,
    )
    result["shape"] = [A.size(0), W.size(0)]
    result["inputs"] = {"A": list(A.shape), "W": list(W.shape), "splits": splits}
    result["combo_definitions"] = {k: list(v) for k, v in combos.items()}

    del A_local, W_local, A_v5, W_v5, ref
    for tensor_variant in A_variants.values():
        del tensor_variant
    for tensor_variant in W_variants.values():
        del tensor_variant
    for out in outputs.values():
        del out
    torch.cuda.empty_cache()
    return result


def print_section(title: str, payload: dict) -> None:
    print(f"\n[{title}]")
    for name, metrics in payload["metrics"].items():
        print(
            f"  {name}: rms={metrics['rms']:.6e} "
            f"p99_abs={metrics['p99_abs']:.6e} "
            f"max_abs={metrics['max_abs']:.6e}"
        )
    for key, value in payload["summary"].items():
        print(f"  {key}={value:.6e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Numerics ablation harness for localCTA vs baseline_v5 sg semantics")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regular-m", type=int, default=16384)
    parser.add_argument("--regular-n", type=int, default=5632)
    parser.add_argument("--regular-k", type=int, default=2048)
    parser.add_argument("--grouped-m", type=int, default=16384)
    parser.add_argument("--grouped-k", type=int, default=2048)
    parser.add_argument("--grouped-splits", type=base.parse_splits, default=base.parse_splits("2048,2048,2048"))
    parser.add_argument("--distribution", choices=base.DISTRIBUTIONS, default="normal")
    parser.add_argument("--regular-activation-dist", choices=base.DISTRIBUTIONS, default=None)
    parser.add_argument("--regular-weight-dist", choices=base.DISTRIBUTIONS, default=None)
    parser.add_argument("--grouped-activation-dist", choices=base.DISTRIBUTIONS, default=None)
    parser.add_argument("--grouped-weight-dist", choices=base.DISTRIBUTIONS, default=None)
    parser.add_argument("--tensor-scale", type=float, default=1.0)
    parser.add_argument("--sparse-spike-prob", type=float, default=2e-4)
    parser.add_argument("--sparse-spike-scale", type=float, default=32.0)
    parser.add_argument("--row-spike-prob", type=float, default=0.02)
    parser.add_argument("--row-spike-scale", type=float, default=12.0)
    parser.add_argument("--chunk-spike-prob", type=float, default=0.10)
    parser.add_argument("--chunk-spike-scale", type=float, default=8.0)
    parser.add_argument("--localcta-sg-rescale", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-elems", type=int, default=8_388_608)
    parser.add_argument("--max-quantile-samples", type=int, default=1_000_000)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def run_experiment(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    regular_activation_dist = base.resolve_distribution(args, args.regular_activation_dist)
    regular_weight_dist = base.resolve_distribution(args, args.regular_weight_dist)
    grouped_activation_dist = base.resolve_distribution(args, args.grouped_activation_dist)
    grouped_weight_dist = base.resolve_distribution(args, args.grouped_weight_dist)

    with torch.inference_mode():
        reg_A = base.make_tensor(
            args.regular_m, args.regular_k,
            k_dim=args.regular_k, device="cuda", dist=regular_activation_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )
        reg_B = base.make_tensor(
            args.regular_n, args.regular_k,
            k_dim=args.regular_k, device="cuda", dist=regular_weight_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )
        grp_A = base.make_tensor(
            args.grouped_m, args.grouped_k,
            k_dim=args.grouped_k, device="cuda", dist=grouped_activation_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )
        grp_W = base.make_tensor(
            sum(args.grouped_splits), args.grouped_k,
            k_dim=args.grouped_k, device="cuda", dist=grouped_weight_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )

        results = {
            "config": {
                "regular_shape": [args.regular_m, args.regular_n, args.regular_k],
                "grouped_shape": [args.grouped_m, sum(args.grouped_splits), args.grouped_k],
                "grouped_splits": args.grouped_splits,
                "distributions": {
                    "regular_activation": regular_activation_dist,
                    "regular_weight": regular_weight_dist,
                    "grouped_activation": grouped_activation_dist,
                    "grouped_weight": grouped_weight_dist,
                },
            },
            "tensor_reconstruction": {
                "regular_activation": tensor_ablation_section(
                    "regular_activation",
                    reg_A,
                    distribution=regular_activation_dist,
                    localcta_sg_rescale=args.localcta_sg_rescale,
                    epsilon=args.epsilon,
                    top_k=args.top_k,
                    chunk_elems=args.chunk_elems,
                    max_quantile_samples=args.max_quantile_samples,
                ),
                "regular_weight": tensor_ablation_section(
                    "regular_weight",
                    reg_B,
                    distribution=regular_weight_dist,
                    localcta_sg_rescale=args.localcta_sg_rescale,
                    epsilon=args.epsilon,
                    top_k=args.top_k,
                    chunk_elems=args.chunk_elems,
                    max_quantile_samples=args.max_quantile_samples,
                ),
                "grouped_activation": tensor_ablation_section(
                    "grouped_activation",
                    grp_A,
                    distribution=grouped_activation_dist,
                    localcta_sg_rescale=args.localcta_sg_rescale,
                    epsilon=args.epsilon,
                    top_k=args.top_k,
                    chunk_elems=args.chunk_elems,
                    max_quantile_samples=args.max_quantile_samples,
                ),
                "grouped_weight": tensor_ablation_section(
                    "grouped_weight",
                    grp_W,
                    distribution=grouped_weight_dist,
                    localcta_sg_rescale=args.localcta_sg_rescale,
                    epsilon=args.epsilon,
                    top_k=args.top_k,
                    chunk_elems=args.chunk_elems,
                    max_quantile_samples=args.max_quantile_samples,
                ),
            },
            "regular_matmul_ablation": build_regular_matmul_ablation(
                reg_A,
                reg_B,
                localcta_sg_rescale=args.localcta_sg_rescale,
                epsilon=args.epsilon,
                top_k=args.top_k,
                chunk_elems=args.chunk_elems,
                max_quantile_samples=args.max_quantile_samples,
            ),
            "grouped_matmul_ablation": build_grouped_matmul_ablation(
                grp_A,
                grp_W,
                args.grouped_splits,
                localcta_sg_rescale=args.localcta_sg_rescale,
                epsilon=args.epsilon,
                top_k=args.top_k,
                chunk_elems=args.chunk_elems,
                max_quantile_samples=args.max_quantile_samples,
            ),
        }

    return results


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_experiment(args)

    for name, payload in results["tensor_reconstruction"].items():
        print_section(f"tensor_reconstruction:{name}", payload)
    print_section("regular_matmul_ablation", results["regular_matmul_ablation"])
    print_section("grouped_matmul_ablation", results["grouped_matmul_ablation"])

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2))
        print(f"\nfull_results_json={json_path}")


if __name__ == "__main__":
    main()
