#!/usr/bin/env python3
"""Screen the native localCTA v4 W2-residual -> RMSNorm/FP4 boundary."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
from pathlib import Path

import torch


DEFAULT_FP4_ROOT = Path("/tmp/fp4_matmul_cde_localcta_exp_20260721")
DEFAULT_TK_ROOT = Path("/tmp/tk_cde_localcta_exp_20260721")
DEFAULT_TE_FUSED = Path(
    os.environ.get(
        "TE_FUSED_RMSNORM_EXTENSION",
        Path(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                Path.home() / ".cache" / "torch_extensions",
            )
        )
        / "py312_cu130"
        / "te_fused_rmsnorm_ext_linear"
        / "te_fused_rmsnorm_ext_linear.so",
    )
)


def _load_extension(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bytes_mismatch(lhs: torch.Tensor, rhs: torch.Tensor) -> int:
    return int((lhs.view(torch.uint8) != rhs.view(torch.uint8)).sum().item())


def _event_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples)


def _paired_ms(fn_a, fn_b, warmup: int, iterations: int, rounds: int):
    pairs: list[tuple[float, float]] = []
    for round_idx in range(rounds):
        if round_idx % 2 == 0:
            a_ms = _event_ms(fn_a, warmup, iterations)
            b_ms = _event_ms(fn_b, warmup, iterations)
        else:
            b_ms = _event_ms(fn_b, warmup, iterations)
            a_ms = _event_ms(fn_a, warmup, iterations)
        pairs.append((a_ms, b_ms))
    return pairs


def _bf16_metrics(lhs: torch.Tensor, rhs: torch.Tensor):
    maxabs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    dot = 0.0
    rhs_sq = 0.0
    for start in range(0, lhs.size(0), 1024):
        left = lhs[start : start + 1024].float()
        right = rhs[start : start + 1024].float()
        diff = left - right
        maxabs = max(maxabs, float(diff.abs().max().item()))
        diff_sq += float((diff * diff).sum().item())
        ref_sq += float((left * left).sum().item())
        rhs_sq += float((right * right).sum().item())
        dot += float((left * right).sum().item())
    return {
        "byte_mismatch": _bytes_mismatch(lhs, rhs),
        "maxabs": maxabs,
        "relative_l2": (diff_sq / max(ref_sq, 1.0e-30)) ** 0.5,
        "cosine": dot / max((ref_sq * rhs_sq) ** 0.5, 1.0e-30),
    }


def _quantize_row(tkq, value: torch.Tensor, encode_centric: bool):
    result = tkq.tk_localcta_quantize_for_gemm_final_sg_opt(
        value,
        True,
        encode_centric,
        False,
        False,
        "none",
        False,
        0,
        0,
    )
    return result[0], result[1], result[4].to(torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=14336)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--downstream-n", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--encode-centric", action="store_true")
    parser.add_argument(
        "--production-baseline",
        action="store_true",
        help="Use the fused localCTA v4 final-SG RMSNorm producer.",
    )
    parser.add_argument("--small-reference", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260721)
    device = torch.device("cuda", args.device)
    suffix = "cpython-312-aarch64-linux-gnu.so"
    gemm = _load_extension(
        "_C_nv_localcta_gemm_v3",
        DEFAULT_TK_ROOT
        / "kernels/gemm/nvfp4_b200/localCTA_epilogue_v3"
        / f"_C_nv_localcta_gemm_v3.{suffix}",
    )
    tkq = _load_extension(
        "_tk_quant_localcta_v4",
        DEFAULT_FP4_ROOT
        / "TK_quantisation/nvfp4_CTA_local_v4"
        / f"_tk_quant_localcta_v4.{suffix}",
    )
    te_fused = _load_extension(
        "te_fused_rmsnorm_ext_linear",
        DEFAULT_TE_FUSED,
    )

    M, N, K = args.m, args.n, args.k
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    residual = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    gamma = torch.randn((N,), dtype=torch.bfloat16, device=device)
    x_fp4, x_sc, x_sg = _quantize_row(tkq, x, args.encode_centric)
    w_fp4, w_sc, w_sg = _quantize_row(tkq, weight, args.encode_centric)

    out_base = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    out_cand = torch.empty_like(out_base)
    partial = torch.empty((M, N // 256), dtype=torch.float32, device=device)

    def baseline_quantize():
        if args.production_baseline:
            return tkq.tk_localcta_rmsnorm_quantize_for_gemm_final_sg_opt(
                out_base,
                gamma,
                args.epsilon,
                True,
                args.encode_centric,
                False,
                False,
                "none",
                False,
                0,
                0,
            )
        normed, inv_rms = te_fused.fused_rmsnorm_only(
            out_base, gamma, args.epsilon
        )
        quantized = tkq.tk_localcta_quantize_for_gemm_final_sg(
            normed, True, args.encode_centric
        )
        return (*quantized, inv_rms)

    def baseline():
        gemm.nvfp4_localcta_gemm_residual(
            x_fp4,
            x_sc,
            x_sg,
            w_fp4,
            w_sc,
            w_sg,
            residual,
            out_base,
        )
        return baseline_quantize()

    def candidate():
        gemm.nvfp4_localcta_gemm_residual_rms(
            x_fp4,
            x_sc,
            x_sg,
            w_fp4,
            w_sc,
            w_sg,
            residual,
            out_cand,
            partial,
        )
        return tkq.tk_localcta_rmsnorm_quantize_from_row_rms_partial_final_sg(
            out_cand,
            gamma,
            partial,
            args.epsilon,
            True,
            args.encode_centric,
        )

    base_result = baseline()
    cand_result = candidate()
    torch.cuda.synchronize()

    report: dict[str, object] = {
        "shape": [M, N, K],
        "encode_centric": args.encode_centric,
        "production_baseline": args.production_baseline,
        "output_bf16_mismatch": _bytes_mismatch(out_base, out_cand),
        "inv_rms_maxabs": float(
            (base_result[6] - cand_result[6]).abs().max().item()
        ),
        "inv_rms_byte_mismatch": _bytes_mismatch(
            base_result[6], cand_result[6]
        ),
        "row_fp4_mismatch": _bytes_mismatch(base_result[0], cand_result[0]),
        "row_scale_mismatch": _bytes_mismatch(base_result[1], cand_result[1]),
        "col_fp4_mismatch": _bytes_mismatch(base_result[2], cand_result[2]),
        "col_scale_mismatch": _bytes_mismatch(base_result[3], cand_result[3]),
        "row_sg_maxabs": float(
            (base_result[4] - cand_result[4]).abs().max().item()
        ),
        "col_sg_maxabs": float(
            (base_result[5] - cand_result[5]).abs().max().item()
        ),
    }

    if args.small_reference:
        partial_ref = (
            out_cand.float().square().reshape(M, N // 256, 256).sum(dim=-1)
        )
        inv_ref = torch.rsqrt(
            out_cand.float().square().mean(dim=-1) + args.epsilon
        )
        report["partial_maxabs_vs_torch"] = float(
            (partial - partial_ref).abs().max().item()
        )
        report["inv_maxabs_vs_torch"] = float(
            (cand_result[6] - inv_ref).abs().max().item()
        )

    if args.downstream_n:
        downstream_weight = torch.randn(
            (args.downstream_n, N), dtype=torch.bfloat16, device=device
        )
        next_fp4, next_sc, next_sg = _quantize_row(
            tkq, downstream_weight, args.encode_centric
        )
        downstream_base = torch.empty(
            (M, args.downstream_n), dtype=torch.bfloat16, device=device
        )
        downstream_cand = torch.empty_like(downstream_base)
        gemm.nvfp4_localcta_gemm(
            base_result[0],
            base_result[1],
            base_result[4],
            next_fp4,
            next_sc,
            next_sg,
            downstream_base,
        )
        gemm.nvfp4_localcta_gemm(
            cand_result[0],
            cand_result[1],
            cand_result[4],
            next_fp4,
            next_sc,
            next_sg,
            downstream_cand,
        )
        torch.cuda.synchronize()
        report["downstream_n"] = args.downstream_n
        report["downstream"] = _bf16_metrics(
            downstream_base, downstream_cand
        )

    pairs = _paired_ms(
        baseline, candidate, args.warmup, args.iterations, args.rounds
    )
    base_ms = statistics.median(pair[0] for pair in pairs)
    cand_ms = statistics.median(pair[1] for pair in pairs)

    def baseline_gemm():
        gemm.nvfp4_localcta_gemm_residual(
            x_fp4, x_sc, x_sg, w_fp4, w_sc, w_sg,
            residual, out_base,
        )

    def candidate_gemm():
        gemm.nvfp4_localcta_gemm_residual_rms(
            x_fp4, x_sc, x_sg, w_fp4, w_sc, w_sg,
            residual, out_cand, partial,
        )

    def baseline_quant():
        return baseline_quantize()

    def candidate_quant():
        return tkq.tk_localcta_rmsnorm_quantize_from_row_rms_partial_final_sg(
            out_cand,
            gamma,
            partial,
            args.epsilon,
            True,
            args.encode_centric,
        )

    stage_iterations = max(5, args.iterations // 2)
    report.update(
        {
            "baseline_ms": base_ms,
            "candidate_ms": cand_ms,
            "speedup": base_ms / cand_ms,
            "delta_ms": cand_ms - base_ms,
            "pair_ms": pairs,
            "pair_speedups": [a_ms / b_ms for a_ms, b_ms in pairs],
            "stage_ms": {
                "baseline_residual_gemm": _event_ms(
                    baseline_gemm, 2, stage_iterations
                ),
                "candidate_residual_partial_gemm": _event_ms(
                    candidate_gemm, 2, stage_iterations
                ),
                "baseline_norm_quant": _event_ms(
                    baseline_quant, 2, stage_iterations
                ),
                "candidate_reduce_norm_quant": _event_ms(
                    candidate_quant, 2, stage_iterations
                ),
            },
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
