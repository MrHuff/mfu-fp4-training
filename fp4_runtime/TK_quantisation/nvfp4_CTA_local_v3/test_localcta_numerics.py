import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
FP4_ROOT = ROOT.parents[1]
LOCALCTA_GEMM_ROOT = FP4_ROOT / "ThunderKittens" / "kernels" / "gemm" / "nvfp4_b200" / "localCTA_epilogue"
LEGACY_GEMM_ROOT = FP4_ROOT / "ThunderKittens" / "kernels" / "gemm" / "nvfp4_b200"
V5_ROOT = ROOT.parent / "nvfp4_v5"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V5_ROOT))
sys.path.insert(0, str(LOCALCTA_GEMM_ROOT))
sys.path.insert(0, str(LEGACY_GEMM_ROOT))

import _C as legacy_gemm  # type: ignore
import _C_nv_localcta_gemm as local_gemm  # type: ignore
import _tk_quant_localcta as local_q  # type: ignore
import _tk_quant_v5 as q_v5  # type: ignore

DISTRIBUTIONS = (
    "normal",
    "laplace",
    "student_t",
    "sparse_spikes",
    "row_spikes",
    "chunk_spikes",
)


def parse_splits(value: str) -> list[int]:
    parts = [int(piece) for piece in value.split(",") if piece]
    if not parts:
        raise ValueError("grouped-splits must contain at least one split")
    return parts


def sample_distribution(
    rows: int,
    cols: int,
    *,
    device: str,
    dist: str,
    sparse_spike_prob: float,
    sparse_spike_scale: float,
    student_t_df: float,
    row_spike_prob: float,
    row_spike_scale: float,
    chunk_spike_prob: float,
    chunk_spike_scale: float,
) -> torch.Tensor:
    shape = (rows, cols)
    x = torch.randn(shape, dtype=torch.float32, device=device)

    if dist == "normal":
        return x

    if dist == "laplace":
        u = torch.rand(shape, dtype=torch.float32, device=device).clamp_(1e-6, 1.0 - 1e-6) - 0.5
        return -torch.sign(u) * torch.log1p(-2.0 * torch.abs(u)) / math.sqrt(2.0)

    if dist == "student_t":
        z = torch.randn(shape, dtype=torch.float32, device=device)
        gamma = torch.distributions.Gamma(student_t_df * 0.5, 2.0).sample(shape).to(device=device, dtype=torch.float32)
        return z / torch.sqrt(gamma / student_t_df)

    if dist == "sparse_spikes":
        spike_mask = torch.rand(shape, dtype=torch.float32, device=device) < sparse_spike_prob
        return torch.where(spike_mask, x * sparse_spike_scale, x)

    if dist == "row_spikes":
        row_mask = torch.rand((rows, 1), dtype=torch.float32, device=device) < row_spike_prob
        row_scale = torch.where(
            row_mask,
            torch.full((rows, 1), row_spike_scale, dtype=torch.float32, device=device),
            torch.ones((rows, 1), dtype=torch.float32, device=device),
        )
        return x * row_scale

    if dist == "chunk_spikes":
        chunk_rows = math.ceil(rows / 128)
        chunk_cols = math.ceil(cols / 128)
        chunk_mask = torch.rand((chunk_rows, chunk_cols), dtype=torch.float32, device=device) < chunk_spike_prob
        chunk_scale = torch.where(
            chunk_mask,
            torch.full((chunk_rows, chunk_cols), chunk_spike_scale, dtype=torch.float32, device=device),
            torch.ones((chunk_rows, chunk_cols), dtype=torch.float32, device=device),
        )
        expanded = chunk_scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
        return x * expanded

    raise ValueError(f"Unsupported distribution: {dist}")


def make_tensor(
    rows: int,
    cols: int,
    *,
    k_dim: int,
    device: str,
    dist: str,
    tensor_scale: float,
    sparse_spike_prob: float,
    sparse_spike_scale: float,
    student_t_df: float,
    row_spike_prob: float,
    row_spike_scale: float,
    chunk_spike_prob: float,
    chunk_spike_scale: float,
) -> torch.Tensor:
    base = sample_distribution(
        rows,
        cols,
        device=device,
        dist=dist,
        sparse_spike_prob=sparse_spike_prob,
        sparse_spike_scale=sparse_spike_scale,
        student_t_df=student_t_df,
        row_spike_prob=row_spike_prob,
        row_spike_scale=row_spike_scale,
        chunk_spike_prob=chunk_spike_prob,
        chunk_spike_scale=chunk_spike_scale,
    )
    base.mul_(tensor_scale / (k_dim ** 0.25))
    return base.to(torch.bfloat16)


def tensor_input_summary(tensor: torch.Tensor, *, max_quantile_samples: int) -> dict:
    flat = tensor.reshape(-1).to(torch.float32)
    total = flat.numel()
    stride = max(1, math.ceil(total / max_quantile_samples))
    sample = flat[::stride].abs().cpu()
    mean = float(flat.mean().item())
    std = float(flat.std(unbiased=False).item())
    abs_flat = flat.abs()

    def q(quantile: float) -> float:
        return float(torch.quantile(sample, quantile).item())

    p95 = q(0.95)
    p99 = q(0.99)
    p999 = q(0.999)
    max_abs = float(abs_flat.max().item())

    return {
        "numel": total,
        "mean": mean,
        "std": std,
        "abs_mean": float(abs_flat.mean().item()),
        "abs_p95": p95,
        "abs_p99": p99,
        "abs_p999": p999,
        "abs_max": max_abs,
        "quantile_sample_size": int(sample.numel()),
        "quantile_sampling_stride": stride,
        "max_over_p999": max_abs / max(p999, 1e-12),
        "p999_over_p99": p999 / max(p99, 1e-12),
        "frac_abs_gt_6std": float((abs_flat > (6.0 * max(std, 1e-12))).to(torch.float32).mean().item()),
        "frac_abs_gt_8std": float((abs_flat > (8.0 * max(std, 1e-12))).to(torch.float32).mean().item()),
    }


def resolve_distribution(args: argparse.Namespace, override_name: str | None) -> str:
    return override_name if override_name is not None else args.distribution


def constant_chunk_grid(rows: int, cols: int, sg: torch.Tensor) -> torch.Tensor:
    return torch.full((rows // 128, cols // 128), sg.item(), device=sg.device, dtype=torch.float32)


def unravel_index(flat_index: int, shape: tuple[int, ...]) -> list[int]:
    coords: list[int] = []
    remaining = flat_index
    for dim in reversed(shape):
        coords.append(remaining % dim)
        remaining //= dim
    return list(reversed(coords))


def topk_push(heap: list[tuple[float, int, dict]], value: float, record: dict, k: int) -> None:
    import heapq

    item = (value, int(record["flat_index"]), record)
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif value > heap[0][0]:
        heapq.heapreplace(heap, item)


def finalize_heap(heap: list[tuple[float, int, dict]]) -> list[dict]:
    return [record for _, _, record in sorted(heap, key=lambda pair: (pair[0], pair[1]), reverse=True)]


def chunked_error_metrics(
    out: torch.Tensor,
    ref: torch.Tensor,
    *,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
) -> dict:
    assert out.shape == ref.shape
    flat_out = out.reshape(-1)
    flat_ref = ref.reshape(-1)
    total = flat_out.numel()
    sample_step = max(1, math.ceil(total / max_quantile_samples))

    sum_rel = 0.0
    sum_abs = 0.0
    sum_rel_sq = 0.0
    sum_abs_sq = 0.0
    max_rel = 0.0
    max_abs = 0.0

    rel_heap: list[tuple[float, int, dict]] = []
    abs_heap: list[tuple[float, int, dict]] = []
    rel_samples: list[torch.Tensor] = []
    abs_samples: list[torch.Tensor] = []

    for start in range(0, total, chunk_elems):
        end = min(total, start + chunk_elems)
        out_chunk = flat_out[start:end].to(torch.float32)
        ref_chunk = flat_ref[start:end].to(torch.float32)
        abs_err = (out_chunk - ref_chunk).abs()
        rel_err = abs_err / (ref_chunk.abs() + epsilon)

        sum_rel += rel_err.sum().item()
        sum_abs += abs_err.sum().item()
        sum_rel_sq += (rel_err * rel_err).sum().item()
        sum_abs_sq += (abs_err * abs_err).sum().item()
        max_rel = max(max_rel, rel_err.max().item())
        max_abs = max(max_abs, abs_err.max().item())

        offset = (sample_step - (start % sample_step)) % sample_step
        rel_samples.append(rel_err[offset::sample_step].cpu())
        abs_samples.append(abs_err[offset::sample_step].cpu())

        local_k = min(top_k, end - start)
        rel_vals, rel_idx = torch.topk(rel_err, k=local_k)
        abs_vals, abs_idx = torch.topk(abs_err, k=local_k)

        for value_tensor, idx_tensor in zip(rel_vals, rel_idx):
            local_idx = int(idx_tensor.item())
            global_idx = start + local_idx
            record = {
                "flat_index": global_idx,
                "index": unravel_index(global_idx, tuple(out.shape)),
                "out": float(out_chunk[local_idx].item()),
                "ref": float(ref_chunk[local_idx].item()),
                "abs_error": float(abs_err[local_idx].item()),
                "rel_error": float(value_tensor.item()),
            }
            topk_push(rel_heap, float(value_tensor.item()), record, top_k)

        for value_tensor, idx_tensor in zip(abs_vals, abs_idx):
            local_idx = int(idx_tensor.item())
            global_idx = start + local_idx
            record = {
                "flat_index": global_idx,
                "index": unravel_index(global_idx, tuple(out.shape)),
                "out": float(out_chunk[local_idx].item()),
                "ref": float(ref_chunk[local_idx].item()),
                "abs_error": float(value_tensor.item()),
                "rel_error": float(rel_err[local_idx].item()),
            }
            topk_push(abs_heap, float(value_tensor.item()), record, top_k)

    rel_quantiles = torch.cat(rel_samples) if rel_samples else torch.empty(0, dtype=torch.float32)
    abs_quantiles = torch.cat(abs_samples) if abs_samples else torch.empty(0, dtype=torch.float32)
    count = max(total, 1)
    mean_rel = sum_rel / count
    mean_abs = sum_abs / count
    std_rel = math.sqrt(max((sum_rel_sq / count) - (mean_rel * mean_rel), 0.0))
    std_abs = math.sqrt(max((sum_abs_sq / count) - (mean_abs * mean_abs), 0.0))

    def q(values: torch.Tensor, quantile: float) -> float:
        if values.numel() == 0:
            return float("nan")
        return float(torch.quantile(values, quantile).item())

    return {
        "numel": total,
        "epsilon": epsilon,
        "quantiles_sampled": sample_step > 1,
        "quantile_sample_size": int(rel_quantiles.numel()),
        "quantile_sampling_stride": sample_step,
        "mean_rel": mean_rel,
        "median_rel": q(rel_quantiles, 0.5),
        "max_rel": max_rel,
        "std_rel": std_rel,
        "p99_rel": q(rel_quantiles, 0.99),
        "p999_rel": q(rel_quantiles, 0.999),
        "iqr_rel": q(rel_quantiles, 0.75) - q(rel_quantiles, 0.25),
        "mean_abs": mean_abs,
        "median_abs": q(abs_quantiles, 0.5),
        "max_abs": max_abs,
        "std_abs": std_abs,
        "p99_abs": q(abs_quantiles, 0.99),
        "p999_abs": q(abs_quantiles, 0.999),
        "iqr_abs": q(abs_quantiles, 0.75) - q(abs_quantiles, 0.25),
        "rms": math.sqrt(sum_abs_sq / count),
        "worst_offenders_rel": finalize_heap(rel_heap),
        "worst_offenders_abs": finalize_heap(abs_heap),
    }


def compare_summary(local_metrics: dict, baseline_metrics: dict) -> dict:
    keys = ["mean_abs", "p99_abs", "p999_abs", "max_abs", "rms", "mean_rel", "p99_rel", "p999_rel", "max_rel"]
    summary: dict[str, float] = {}
    for key in keys:
        summary[f"localcta_minus_baseline_{key}"] = local_metrics[key] - baseline_metrics[key]
        denom = baseline_metrics[key]
        summary[f"localcta_over_baseline_{key}"] = (
            local_metrics[key] / denom if denom != 0 else float("inf")
        )
    return summary


def scale_summary_localcta(sg_chunks: torch.Tensor) -> dict:
    values = sg_chunks.to(torch.float32).reshape(-1).cpu()
    return {
        "num_chunks": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p01": float(torch.quantile(values, 0.01).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "coeff_var": float(values.std(unbiased=False).item() / max(values.mean().item(), 1e-12)),
    }


def scale_summary_baseline(sg: torch.Tensor) -> dict:
    value = float(sg.to(torch.float32).item())
    return {
        "num_chunks": 1,
        "mean": value,
        "std": 0.0,
        "min": value,
        "max": value,
        "p01": value,
        "p99": value,
        "coeff_var": 0.0,
    }


def build_tensor_reconstruction_section(
    name: str,
    tensor: torch.Tensor,
    *,
    distribution: str,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
) -> dict:
    local_quant = local_q.tk_localcta_quantize_for_gemm_fast(tensor, False, True)
    v5_quant = q_v5.tk_quantize_for_gemm(tensor, False, True)
    local_recon = local_q.tk_localcta_reconstruct_row(local_quant[0], local_quant[1], local_quant[4])
    v5_sg_grid = constant_chunk_grid(tensor.size(0), tensor.size(1), v5_quant[4])
    v5_recon = local_q.tk_localcta_reconstruct_row(v5_quant[0], v5_quant[1], v5_sg_grid)
    torch.cuda.synchronize()

    local_vs_ref = chunked_error_metrics(
        local_recon, tensor,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )
    baseline_vs_ref = chunked_error_metrics(
        v5_recon, tensor,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )
    local_vs_baseline = chunked_error_metrics(
        local_recon, v5_recon,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )

    result = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "distribution": distribution,
        "input_summary": tensor_input_summary(tensor, max_quantile_samples=max_quantile_samples),
        "localcta_chunk_scale_summary": scale_summary_localcta(local_quant[4]),
        "baseline_v5_scale_summary": scale_summary_baseline(v5_quant[4]),
        "metrics": {
            "localcta_vs_bf16": local_vs_ref,
            "baseline_v5_vs_bf16": baseline_vs_ref,
            "localcta_vs_baseline_v5": local_vs_baseline,
        },
        "comparison_summary": compare_summary(local_vs_ref, baseline_vs_ref),
    }

    del local_quant, v5_quant, local_recon, v5_recon, v5_sg_grid
    torch.cuda.empty_cache()
    return result


def regular_matmul_section(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
    activation_recon: dict,
    weight_recon: dict,
) -> dict:
    A_local = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
    B_local = local_q.tk_localcta_quantize_for_gemm_prepared(B, False, True)
    A_v5 = q_v5.tk_quantize_for_gemm(A, False, True)
    B_v5 = q_v5.tk_quantize_for_gemm(B, False, True)

    local_out = torch.empty(A.size(0), B.size(0), dtype=torch.bfloat16, device="cuda")
    baseline_out = torch.empty_like(local_out)
    ref = torch.matmul(A, B.t()).to(torch.bfloat16)

    local_gemm.nvfp4_localcta_fast_gemm(
        A_local[0], A_local[1],
        B_local[0], B_local[1],
        local_out,
    )
    legacy_gemm.nvfp4_gemm(
        A_v5[0], A_v5[1], A_v5[4],
        B_v5[0], B_v5[1], B_v5[4],
        baseline_out,
    )
    torch.cuda.synchronize()

    local_vs_ref = chunked_error_metrics(
        local_out, ref,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )
    baseline_vs_ref = chunked_error_metrics(
        baseline_out, ref,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )
    local_vs_baseline = chunked_error_metrics(
        local_out, baseline_out,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )

    local_operand_recon_rms = 0.5 * (
        activation_recon["metrics"]["localcta_vs_bf16"]["rms"] +
        weight_recon["metrics"]["localcta_vs_bf16"]["rms"]
    )
    baseline_operand_recon_rms = 0.5 * (
        activation_recon["metrics"]["baseline_v5_vs_bf16"]["rms"] +
        weight_recon["metrics"]["baseline_v5_vs_bf16"]["rms"]
    )

    result = {
        "shape": [A.size(0), B.size(0)],
        "inputs": {"A": list(A.shape), "B": list(B.shape)},
        "metrics": {
            "localcta_prepared_vs_bf16": local_vs_ref,
            "baseline_v5_vs_bf16": baseline_vs_ref,
            "localcta_prepared_vs_baseline_v5": local_vs_baseline,
        },
        "comparison_summary": compare_summary(local_vs_ref, baseline_vs_ref),
        "diagnostics": {
            "localcta_operand_recon_rms_mean": local_operand_recon_rms,
            "baseline_operand_recon_rms_mean": baseline_operand_recon_rms,
            "localcta_output_rms_amplification": local_vs_ref["rms"] / max(local_operand_recon_rms, 1e-12),
            "baseline_output_rms_amplification": baseline_vs_ref["rms"] / max(baseline_operand_recon_rms, 1e-12),
        },
    }

    del A_local, B_local, A_v5, B_v5, local_out, baseline_out, ref
    torch.cuda.empty_cache()
    return result


def grouped_matmul_section(
    A: torch.Tensor,
    W: torch.Tensor,
    splits: list[int],
    *,
    epsilon: float,
    top_k: int,
    chunk_elems: int,
    max_quantile_samples: int,
    activation_recon: dict,
    weight_recon: dict,
) -> dict:
    A_local = local_q.tk_localcta_quantize_for_gemm_prepared(A, False, True)
    W_local = local_q.tk_localcta_group_quantize_for_gemm_prepared(W, splits)
    A_v5 = q_v5.tk_quantize_for_gemm(A, False, True)
    W_v5 = q_v5.tk_group_quantize_for_gemm(W, splits)

    local_out = torch.empty(A.size(0), W.size(0), dtype=torch.bfloat16, device="cuda")
    baseline_out = torch.empty_like(local_out)
    ref = torch.matmul(A, W.t()).to(torch.bfloat16)

    local_gemm.nvfp4_localcta_fast_grouped_gemm(
        A_local[0], A_local[1],
        W_local[0], W_local[1],
        local_out,
    )
    legacy_gemm.nvfp4_grouped_gemm(
        A_v5[0], A_v5[1], A_v5[4],
        W_v5[0], W_v5[1], W_v5[2],
        baseline_out,
    )
    torch.cuda.synchronize()

    local_vs_ref = chunked_error_metrics(
        local_out, ref,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )
    baseline_vs_ref = chunked_error_metrics(
        baseline_out, ref,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )
    local_vs_baseline = chunked_error_metrics(
        local_out, baseline_out,
        epsilon=epsilon, top_k=top_k, chunk_elems=chunk_elems, max_quantile_samples=max_quantile_samples,
    )

    local_operand_recon_rms = 0.5 * (
        activation_recon["metrics"]["localcta_vs_bf16"]["rms"] +
        weight_recon["metrics"]["localcta_vs_bf16"]["rms"]
    )
    baseline_operand_recon_rms = 0.5 * (
        activation_recon["metrics"]["baseline_v5_vs_bf16"]["rms"] +
        weight_recon["metrics"]["baseline_v5_vs_bf16"]["rms"]
    )

    result = {
        "shape": [A.size(0), W.size(0)],
        "inputs": {"A": list(A.shape), "W": list(W.shape), "splits": splits},
        "metrics": {
            "localcta_prepared_vs_bf16": local_vs_ref,
            "baseline_v5_vs_bf16": baseline_vs_ref,
            "localcta_prepared_vs_baseline_v5": local_vs_baseline,
        },
        "comparison_summary": compare_summary(local_vs_ref, baseline_vs_ref),
        "diagnostics": {
            "localcta_operand_recon_rms_mean": local_operand_recon_rms,
            "baseline_operand_recon_rms_mean": baseline_operand_recon_rms,
            "localcta_output_rms_amplification": local_vs_ref["rms"] / max(local_operand_recon_rms, 1e-12),
            "baseline_output_rms_amplification": baseline_vs_ref["rms"] / max(baseline_operand_recon_rms, 1e-12),
        },
    }

    del A_local, W_local, A_v5, W_v5, local_out, baseline_out, ref
    torch.cuda.empty_cache()
    return result


def print_metric_line(label: str, metrics: dict) -> None:
    print(
        f"{label}: mean_abs={metrics['mean_abs']:.6e} "
        f"p99_abs={metrics['p99_abs']:.6e} "
        f"p999_abs={metrics['p999_abs']:.6e} "
        f"max_abs={metrics['max_abs']:.6e} "
        f"rms={metrics['rms']:.6e}"
    )


def print_recon_summary(name: str, payload: dict) -> None:
    print(f"\n[tensor_reconstruction] {name} shape={tuple(payload['shape'])} distribution={payload['distribution']}")
    print_metric_line("  localcta_vs_bf16", payload["metrics"]["localcta_vs_bf16"])
    print_metric_line("  baseline_v5_vs_bf16", payload["metrics"]["baseline_v5_vs_bf16"])
    print_metric_line("  localcta_vs_baseline_v5", payload["metrics"]["localcta_vs_baseline_v5"])
    input_summary = payload["input_summary"]
    print(
        f"  input_abs_stats: p99={input_summary['abs_p99']:.6e} "
        f"p999={input_summary['abs_p999']:.6e} max={input_summary['abs_max']:.6e} "
        f"max/p999={input_summary['max_over_p999']:.6e}"
    )
    scale = payload["localcta_chunk_scale_summary"]
    print(
        f"  localcta_chunk_scales: mean={scale['mean']:.6e} std={scale['std']:.6e} "
        f"min={scale['min']:.6e} max={scale['max']:.6e} coeff_var={scale['coeff_var']:.6e}"
    )


def print_matmul_summary(name: str, payload: dict) -> None:
    print(f"\n[{name}] shape={tuple(payload['shape'])}")
    print_metric_line("  localcta_prepared_vs_bf16", payload["metrics"]["localcta_prepared_vs_bf16"])
    print_metric_line("  baseline_v5_vs_bf16", payload["metrics"]["baseline_v5_vs_bf16"])
    print_metric_line("  localcta_prepared_vs_baseline_v5", payload["metrics"]["localcta_prepared_vs_baseline_v5"])
    diag = payload["diagnostics"]
    print(
        f"  rms_amplification: localcta={diag['localcta_output_rms_amplification']:.6e} "
        f"baseline_v5={diag['baseline_output_rms_amplification']:.6e}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detailed numerics harness for CTA-local prepared vs baseline_v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regular-m", type=int, default=16384)
    parser.add_argument("--regular-n", type=int, default=5632)
    parser.add_argument("--regular-k", type=int, default=2048)
    parser.add_argument("--grouped-m", type=int, default=16384)
    parser.add_argument("--grouped-k", type=int, default=2048)
    parser.add_argument("--grouped-splits", type=parse_splits, default=parse_splits("2048,2048,2048"))
    parser.add_argument("--distribution", choices=DISTRIBUTIONS, default="normal")
    parser.add_argument("--regular-activation-dist", choices=DISTRIBUTIONS, default=None)
    parser.add_argument("--regular-weight-dist", choices=DISTRIBUTIONS, default=None)
    parser.add_argument("--grouped-activation-dist", choices=DISTRIBUTIONS, default=None)
    parser.add_argument("--grouped-weight-dist", choices=DISTRIBUTIONS, default=None)
    parser.add_argument("--tensor-scale", type=float, default=1.0)
    parser.add_argument("--sparse-spike-prob", type=float, default=2e-4)
    parser.add_argument("--sparse-spike-scale", type=float, default=32.0)
    parser.add_argument("--student-t-df", type=float, default=3.0)
    parser.add_argument("--row-spike-prob", type=float, default=0.02)
    parser.add_argument("--row-spike-scale", type=float, default=12.0)
    parser.add_argument("--chunk-spike-prob", type=float, default=0.10)
    parser.add_argument("--chunk-spike-scale", type=float, default=8.0)
    parser.add_argument("--localcta-global-scale-num", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-elems", type=int, default=8_388_608)
    parser.add_argument("--max-quantile-samples", type=int, default=2_000_000)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser


def run_experiment(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.localcta_global_scale_num is None:
        local_q.tk_localcta_reset_global_scale_num()
    else:
        local_q.tk_localcta_set_global_scale_num(float(args.localcta_global_scale_num))

    regular_activation_dist = resolve_distribution(args, args.regular_activation_dist)
    regular_weight_dist = resolve_distribution(args, args.regular_weight_dist)
    grouped_activation_dist = resolve_distribution(args, args.grouped_activation_dist)
    grouped_weight_dist = resolve_distribution(args, args.grouped_weight_dist)

    with torch.inference_mode():
        reg_A = make_tensor(
            args.regular_m, args.regular_k,
            k_dim=args.regular_k, device="cuda", dist=regular_activation_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            student_t_df=args.student_t_df,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )
        reg_B = make_tensor(
            args.regular_n, args.regular_k,
            k_dim=args.regular_k, device="cuda", dist=regular_weight_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            student_t_df=args.student_t_df,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )

        tensor_reconstruction = {
            "regular_activation": build_tensor_reconstruction_section(
                "regular_activation",
                reg_A,
                distribution=regular_activation_dist,
                epsilon=args.epsilon,
                top_k=args.top_k,
                chunk_elems=args.chunk_elems,
                max_quantile_samples=args.max_quantile_samples,
            ),
            "regular_weight": build_tensor_reconstruction_section(
                "regular_weight",
                reg_B,
                distribution=regular_weight_dist,
                epsilon=args.epsilon,
                top_k=args.top_k,
                chunk_elems=args.chunk_elems,
                max_quantile_samples=args.max_quantile_samples,
            ),
        }

        regular_output = regular_matmul_section(
            reg_A,
            reg_B,
            epsilon=args.epsilon,
            top_k=args.top_k,
            chunk_elems=args.chunk_elems,
            max_quantile_samples=args.max_quantile_samples,
            activation_recon=tensor_reconstruction["regular_activation"],
            weight_recon=tensor_reconstruction["regular_weight"],
        )

        grp_A = make_tensor(
            args.grouped_m, args.grouped_k,
            k_dim=args.grouped_k, device="cuda", dist=grouped_activation_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            student_t_df=args.student_t_df,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )
        grp_W = make_tensor(
            sum(args.grouped_splits), args.grouped_k,
            k_dim=args.grouped_k, device="cuda", dist=grouped_weight_dist,
            tensor_scale=args.tensor_scale,
            sparse_spike_prob=args.sparse_spike_prob, sparse_spike_scale=args.sparse_spike_scale,
            student_t_df=args.student_t_df,
            row_spike_prob=args.row_spike_prob, row_spike_scale=args.row_spike_scale,
            chunk_spike_prob=args.chunk_spike_prob, chunk_spike_scale=args.chunk_spike_scale,
        )

        tensor_reconstruction["grouped_activation"] = build_tensor_reconstruction_section(
            "grouped_activation",
            grp_A,
            distribution=grouped_activation_dist,
            epsilon=args.epsilon,
            top_k=args.top_k,
            chunk_elems=args.chunk_elems,
            max_quantile_samples=args.max_quantile_samples,
        )
        tensor_reconstruction["grouped_weight"] = build_tensor_reconstruction_section(
            "grouped_weight",
            grp_W,
            distribution=grouped_weight_dist,
            epsilon=args.epsilon,
            top_k=args.top_k,
            chunk_elems=args.chunk_elems,
            max_quantile_samples=args.max_quantile_samples,
        )

        grouped_output = grouped_matmul_section(
            grp_A,
            grp_W,
            args.grouped_splits,
            epsilon=args.epsilon,
            top_k=args.top_k,
            chunk_elems=args.chunk_elems,
            max_quantile_samples=args.max_quantile_samples,
            activation_recon=tensor_reconstruction["grouped_activation"],
            weight_recon=tensor_reconstruction["grouped_weight"],
        )

    return {
        "config": {
            "seed": args.seed,
            "regular_shape": [args.regular_m, args.regular_n, args.regular_k],
            "grouped_shape": [args.grouped_m, sum(args.grouped_splits), args.grouped_k],
            "grouped_splits": args.grouped_splits,
            "distributions": {
                "regular_activation": regular_activation_dist,
                "regular_weight": regular_weight_dist,
                "grouped_activation": grouped_activation_dist,
                "grouped_weight": grouped_weight_dist,
            },
            "tensor_scale": args.tensor_scale,
            "sparse_spike_prob": args.sparse_spike_prob,
            "sparse_spike_scale": args.sparse_spike_scale,
            "student_t_df": args.student_t_df,
            "row_spike_prob": args.row_spike_prob,
            "row_spike_scale": args.row_spike_scale,
            "chunk_spike_prob": args.chunk_spike_prob,
            "chunk_spike_scale": args.chunk_spike_scale,
            "localcta_global_scale_num": (
                float(args.localcta_global_scale_num)
                if args.localcta_global_scale_num is not None else None
            ),
            "epsilon": args.epsilon,
            "top_k": args.top_k,
            "chunk_elems": args.chunk_elems,
            "max_quantile_samples": args.max_quantile_samples,
        },
        "tensor_reconstruction": tensor_reconstruction,
        "regular_matmul_output": regular_output,
        "grouped_qkv_like_matmul_output": grouped_output,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_experiment(args)

    for name, payload in results["tensor_reconstruction"].items():
        print_recon_summary(name, payload)
    print_matmul_summary("regular_matmul_output", results["regular_matmul_output"])
    print_matmul_summary("grouped_qkv_like_matmul_output", results["grouped_qkv_like_matmul_output"])

    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2))
        print(f"\nfull_results_json={json_path}")

    if args.print_json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
