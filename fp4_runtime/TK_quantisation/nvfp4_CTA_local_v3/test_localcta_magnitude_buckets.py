import argparse
import json
from pathlib import Path

import torch

import test_localcta_numerics as base
import _tk_quant_localcta as local_q  # type: ignore
import _tk_quant_v5 as q_v5  # type: ignore


def quantize_local_raw(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = local_q.tk_localcta_quantize_for_gemm_fast(tensor, False, True)
    return q[0], q[1], q[4]


def quantize_v5_raw(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q_v5.tk_quantize_for_gemm(tensor, False, True)
    sg_grid = base.constant_chunk_grid(tensor.size(0), tensor.size(1), q[4])
    return q[0], q[1], sg_grid, q[4]


def fp8_scale_saturation_summary(scales: torch.Tensor) -> dict:
    values = scales.to(torch.float32).abs().reshape(-1)
    return {
        "mean_abs": float(values.mean().item()),
        "p99_abs": float(torch.quantile(values, 0.99).item()),
        "max_abs": float(values.max().item()),
        "frac_abs_ge_440": float((values >= 440.0).to(torch.float32).mean().item()),
        "frac_abs_ge_447": float((values >= 447.0).to(torch.float32).mean().item()),
        "frac_abs_eq_448": float((values == 448.0).to(torch.float32).mean().item()),
    }


def magnitude_bucket_metrics(
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    num_buckets: int,
    sample_cap: int,
) -> list[dict]:
    flat_ref = ref.reshape(-1).to(torch.float32).abs()
    flat_err = (out.reshape(-1).to(torch.float32) - ref.reshape(-1).to(torch.float32)).abs()
    total = flat_ref.numel()
    stride = max(1, (total + sample_cap - 1) // sample_cap)
    ref_sample = flat_ref[::stride].cpu()

    nonzero = ref_sample[ref_sample > 0]
    if nonzero.numel() == 0:
        edges = torch.linspace(0.0, 1.0, num_buckets + 1)
    else:
        quantiles = torch.linspace(0.0, 1.0, num_buckets + 1)
        edges = torch.quantile(nonzero, quantiles).cpu()
        edges[0] = 0.0
        for i in range(1, edges.numel()):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-12

    buckets: list[dict] = []
    for i in range(num_buckets):
        lo = float(edges[i].item())
        hi = float(edges[i + 1].item())
        if i == num_buckets - 1:
            mask = (flat_ref >= lo) & (flat_ref <= hi)
        else:
            mask = (flat_ref >= lo) & (flat_ref < hi)
        count = int(mask.sum().item())
        if count == 0:
            buckets.append({"bucket": i, "lo": lo, "hi": hi, "count": 0})
            continue
        bucket_err = flat_err[mask]
        buckets.append(
            {
                "bucket": i,
                "lo": lo,
                "hi": hi,
                "count": count,
                "mean_abs": float(bucket_err.mean().item()),
                "rms": float(torch.sqrt((bucket_err * bucket_err).mean()).item()),
                "p99_abs": float(torch.quantile(bucket_err, 0.99).item()),
            }
        )
    return buckets


def bucket_ratio_summary(local_buckets: list[dict], baseline_buckets: list[dict]) -> list[dict]:
    ratios = []
    for l_bucket, b_bucket in zip(local_buckets, baseline_buckets):
        if l_bucket.get("count", 0) == 0 or b_bucket.get("count", 0) == 0:
            continue
        ratios.append(
            {
                "bucket": l_bucket["bucket"],
                "lo": l_bucket["lo"],
                "hi": l_bucket["hi"],
                "count": l_bucket["count"],
                "mean_abs_ratio": l_bucket["mean_abs"] / max(b_bucket["mean_abs"], 1e-12),
                "rms_ratio": l_bucket["rms"] / max(b_bucket["rms"], 1e-12),
                "p99_abs_ratio": l_bucket["p99_abs"] / max(b_bucket["p99_abs"], 1e-12),
            }
        )
    return ratios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Magnitude-bucket diagnostics for localCTA vs baseline_v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=5632)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--distribution", choices=base.DISTRIBUTIONS, default="normal")
    parser.add_argument("--activation-dist", choices=base.DISTRIBUTIONS, default=None)
    parser.add_argument("--weight-dist", choices=base.DISTRIBUTIONS, default=None)
    parser.add_argument("--tensor-scale", type=float, default=1.0)
    parser.add_argument("--sparse-spike-prob", type=float, default=2e-4)
    parser.add_argument("--sparse-spike-scale", type=float, default=32.0)
    parser.add_argument("--row-spike-prob", type=float, default=0.02)
    parser.add_argument("--row-spike-scale", type=float, default=12.0)
    parser.add_argument("--chunk-spike-prob", type=float, default=0.10)
    parser.add_argument("--chunk-spike-scale", type=float, default=8.0)
    parser.add_argument("--num-buckets", type=int, default=10)
    parser.add_argument("--sample-cap", type=int, default=1_000_000)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    activation_dist = base.resolve_distribution(args, args.activation_dist)
    weight_dist = base.resolve_distribution(args, args.weight_dist)

    with torch.inference_mode():
        A = base.make_tensor(
            args.m, args.k,
            k_dim=args.k, device="cuda", dist=activation_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )
        B = base.make_tensor(
            args.n, args.k,
            k_dim=args.k, device="cuda", dist=weight_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )

        A_local_fp4, A_local_sc, A_local_sg = quantize_local_raw(A)
        B_local_fp4, B_local_sc, B_local_sg = quantize_local_raw(B)
        A_v5_fp4, A_v5_sc, A_v5_sg_grid, A_v5_sg = quantize_v5_raw(A)
        B_v5_fp4, B_v5_sc, B_v5_sg_grid, B_v5_sg = quantize_v5_raw(B)

        A_local = local_q.tk_localcta_reconstruct_row(A_local_fp4, A_local_sc, A_local_sg)
        A_v5 = local_q.tk_localcta_reconstruct_row(A_v5_fp4, A_v5_sc, A_v5_sg_grid)
        B_local = local_q.tk_localcta_reconstruct_row(B_local_fp4, B_local_sc, B_local_sg)
        B_v5 = local_q.tk_localcta_reconstruct_row(B_v5_fp4, B_v5_sc, B_v5_sg_grid)

        ref_out = torch.matmul(A, B.t()).to(torch.bfloat16)
        local_out = torch.matmul(A_local, B_local.t()).to(torch.bfloat16)
        baseline_out = torch.matmul(A_v5, B_v5.t()).to(torch.bfloat16)
        torch.cuda.synchronize()

    recon_local_buckets = magnitude_bucket_metrics(A_local, A, num_buckets=args.num_buckets, sample_cap=args.sample_cap)
    recon_base_buckets = magnitude_bucket_metrics(A_v5, A, num_buckets=args.num_buckets, sample_cap=args.sample_cap)
    matmul_local_buckets = magnitude_bucket_metrics(local_out, ref_out, num_buckets=args.num_buckets, sample_cap=args.sample_cap)
    matmul_base_buckets = magnitude_bucket_metrics(baseline_out, ref_out, num_buckets=args.num_buckets, sample_cap=args.sample_cap)

    results = {
        "config": {
            "shape": [args.m, args.n, args.k],
            "distribution": {"activation": activation_dist, "weight": weight_dist},
        },
        "reconstruction": {
            "localcta": recon_local_buckets,
            "baseline_v5": recon_base_buckets,
            "ratio": bucket_ratio_summary(recon_local_buckets, recon_base_buckets),
        },
        "matmul_output": {
            "localcta": matmul_local_buckets,
            "baseline_v5": matmul_base_buckets,
            "ratio": bucket_ratio_summary(matmul_local_buckets, matmul_base_buckets),
        },
        "scale_saturation": {
            "activation_localcta_sc": fp8_scale_saturation_summary(A_local_sc),
            "activation_baseline_v5_sc": fp8_scale_saturation_summary(A_v5_sc),
            "weight_localcta_sc": fp8_scale_saturation_summary(B_local_sc),
            "weight_baseline_v5_sc": fp8_scale_saturation_summary(B_v5_sc),
            "activation_localcta_sg": base.scale_summary_localcta(A_local_sg),
            "activation_baseline_v5_sg": base.scale_summary_baseline(A_v5_sg),
            "weight_localcta_sg": base.scale_summary_localcta(B_local_sg),
            "weight_baseline_v5_sg": base.scale_summary_baseline(B_v5_sg),
        },
    }

    print("[reconstruction bucket ratios: localcta / baseline_v5]")
    for row in results["reconstruction"]["ratio"]:
        print(
            f"  bucket={row['bucket']} lo={row['lo']:.3e} hi={row['hi']:.3e} "
            f"rms_ratio={row['rms_ratio']:.3f} mean_abs_ratio={row['mean_abs_ratio']:.3f} "
            f"p99_abs_ratio={row['p99_abs_ratio']:.3f}"
        )
    print("\n[matmul bucket ratios: localcta / baseline_v5]")
    for row in results["matmul_output"]["ratio"]:
        print(
            f"  bucket={row['bucket']} lo={row['lo']:.3e} hi={row['hi']:.3e} "
            f"rms_ratio={row['rms_ratio']:.3f} mean_abs_ratio={row['mean_abs_ratio']:.3f} "
            f"p99_abs_ratio={row['p99_abs_ratio']:.3f}"
        )
    print("\n[fp8 scale saturation]")
    for name, summary in results["scale_saturation"].items():
        if "frac_abs_ge_440" in summary:
            print(
                f"  {name}: ge440={summary['frac_abs_ge_440']:.6e} "
                f"ge447={summary['frac_abs_ge_447']:.6e} eq448={summary['frac_abs_eq_448']:.6e} "
                f"p99={summary['p99_abs']:.3e} max={summary['max_abs']:.3e}"
            )

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2))
        print(f"\nfull_results_json={json_path}")


if __name__ == "__main__":
    main()
