#!/usr/bin/env python3
"""Exact CUDA equivalence gate for fused W2 and direct-QKV paired RHT carriers."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch


os.environ.setdefault("USE_TK_LOCALCTA_V3_CONTRACT", "outer")
os.environ.setdefault("USE_TK_LOCALCTA_V4_ATOMIC_FINAL_SG_PRODUCER", "1")
os.environ.setdefault("USE_TK_LOCALCTA_V4_SILU_ATOMIC_FINAL_SG_PRODUCER", "1")
os.environ.setdefault("USE_TK_LOCALCTA_V4_GEMM_VIRTUAL_RESCALE", "0")
os.environ.setdefault("USE_TK_LOCALCTA_V4_FUSED_SILU_RAW", "1")


def _load(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.uint8)


def _assert_exact(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    lhs = _bytes(actual)
    rhs = _bytes(expected)
    if torch.equal(lhs, rhs):
        return
    mismatches = int((lhs != rhs).sum().item())
    raise AssertionError(
        f"{name}: {mismatches} byte mismatches over {lhs.numel()} bytes"
    )


def _make(kind: str, rows: int, cols: int, seed: int) -> torch.Tensor:
    if kind == "random":
        generator = torch.Generator(device="cuda").manual_seed(seed)
        return torch.randn(
            (rows, cols), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        ).contiguous()
    if kind == "zeros":
        return torch.zeros((rows, cols), device="cuda", dtype=torch.bfloat16)
    if kind == "tiny":
        return torch.full(
            (rows, cols), 1.0e-8, device="cuda", dtype=torch.bfloat16
        )
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None]
    col = torch.arange(cols, device="cuda", dtype=torch.int64)[None, :]
    signed = torch.where(((row + col) & 1) == 0, 0.75, -0.5)
    if kind == "outlier":
        signed = signed.to(torch.float32)
        signed[0, 0] = 1.0e4
        signed[-1, -1] = -2.0e3
    return signed.to(torch.bfloat16).contiguous()


def _opt_col_rht(module, value: torch.Tensor, encode_centric: bool = False):
    return module.tk_localcta_quantize_for_gemm_opt(
        value, True, encode_centric, False, False, "col", True, 0, 0, "both"
    )


def _check_row(name: str, candidate, base) -> None:
    _assert_exact(f"{name}.row_fp4", candidate[0], base[0])
    _assert_exact(f"{name}.row_sc", candidate[1], base[1])
    _assert_exact(f"{name}.row_sg", candidate[4], base[4])


def _check_col(name: str, candidate, reference) -> None:
    _assert_exact(f"{name}.col_fp4", candidate[2], reference[2])
    _assert_exact(f"{name}.col_sc", candidate[3], reference[3])
    _assert_exact(f"{name}.col_sg", candidate[5], reference[5])


def _col_mismatch_counts(candidate, reference) -> tuple[int, int, int]:
    return tuple(
        int((_bytes(candidate[index]) != _bytes(reference[index])).sum().item())
        for index in (2, 3, 5)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument(
        "--te-extension",
        type=Path,
        help="independent te_fused_rmsnorm_ext_linear oracle for the W2 BF16 payload",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--qkv-only", action="store_true")
    parser.add_argument("--w2-only", action="store_true")
    args = parser.parse_args()
    if args.qkv_only and args.w2_only:
        raise ValueError("--qkv-only and --w2-only are mutually exclusive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    module = _load(args.extension.resolve(), "_tk_quant_localcta_v4")
    te_module = None
    if args.te_extension is not None:
        te_module = _load(
            args.te_extension.resolve(), "te_fused_rmsnorm_ext_linear"
        )
        if not hasattr(te_module, "fused_silu_mul_bf16_out_no_amax"):
            raise RuntimeError(
                "TE oracle lacks fused_silu_mul_bf16_out_no_amax"
            )
    required = (
        "tk_localcta_quantize_for_gemm_atomic_paired_col_rht",
        "tk_localcta_quantize_for_gemm_final_sg_paired_col_rht",
        "tk_localcta_silu_supports_paired_col_rht",
        "tk_localcta_silu_quantize_split_for_gemm_paired_col_rht",
        "tk_localcta_test_w2_transform_bf16_exact",
    )
    missing = [symbol for symbol in required if not hasattr(module, symbol)]
    if missing:
        raise RuntimeError(f"candidate extension lacks {missing}")
    if not module.tk_localcta_silu_supports_paired_col_rht():
        raise RuntimeError("candidate extension reports W2 paired RHT unsupported")
    module.tk_localcta_set_global_scale_num(448.0)

    shapes = (
        ((256, 512),)
        if args.quick
        else (
            (256, 256),
            (256, 512),
            (512, 256),
            (256, 4096),
            (512, 14336),
        )
    )
    kinds = ("random",) if args.quick else ("random", "zeros", "tiny", "signed", "outlier")
    schedules = ((1, 1),) if args.quick else ((0, 0), (0, 1), (1, 0), (1, 1))
    checks = 0
    for parallel, ring in schedules:
        os.environ["USE_TK_LOCALCTA_V4_SILU_PARALLEL_ROW_COL"] = str(parallel)
        os.environ["USE_TK_LOCALCTA_V4_SILU_H3_RING"] = str(ring)
        for rows, cols in shapes:
            for kind in kinds:
                if not args.qkv_only:
                    h1 = _make(kind, rows, cols, 1000 + checks)
                    h3 = _make(kind, rows, cols, 2000 + checks)
                    fast_modes = (True,) if args.quick else (False, True)
                    for fast_divide in fast_modes:
                        os.environ["USE_TK_LOCALCTA_V4_SILU_FAST_DIVIDE"] = (
                            "1" if fast_divide else "0"
                        )
                        precise_bf16, precise_tile_amax = (
                            module.tk_localcta_test_w2_transform_bf16_exact(
                                h1, h3, fast_divide
                            )
                        )
                        fused_bf16, tile_amax = (
                            module.tk_localcta_test_w2_transform_bf16_exact(
                                h1, h3, fast_divide, True
                            )
                        )
                        _assert_exact(
                            "w2.callfree_precise_bf16",
                            fused_bf16,
                            precise_bf16,
                        )
                        _assert_exact(
                            "w2.callfree_precise_tile_amax",
                            tile_amax,
                            precise_tile_amax,
                        )
                        if te_module is not None and fast_divide:
                            te_bf16 = torch.empty_like(h1)
                            te_module.fused_silu_mul_bf16_out_no_amax(
                                h1, h3, te_bf16
                            )
                            _assert_exact(
                                "w2.production_te_bf16", fused_bf16, te_bf16
                            )
                        expected_amax = (
                            fused_bf16.abs()
                            .view(rows // 64, 64, cols // 64, 64)
                            .amax(dim=(1, 3))
                            .float()
                        )
                        _assert_exact("w2.oracle_tile_amax", tile_amax, expected_amax)
                        w2_base = module.tk_localcta_silu_quantize_split_for_gemm(
                            h1, h3
                        )
                        w2_candidate = (
                            module.tk_localcta_silu_quantize_split_for_gemm_paired_col_rht(
                                h1, h3
                            )
                        )
                        w2_reference = _opt_col_rht(module, fused_bf16)
                        _check_row("w2", w2_candidate, w2_base)
                        if args.diagnose:
                            print(
                                "w2 col mismatch bytes",
                                (parallel, ring, rows, cols, kind, fast_divide),
                                _col_mismatch_counts(w2_candidate, w2_reference),
                            )
                        else:
                            _check_col("w2", w2_candidate, w2_reference)

                if not args.w2_only:
                    qkv_x = _make(kind, rows, cols, 3000 + checks)
                    for encode_centric in (False, True):
                        qkv_base_atomic = (
                            module.tk_localcta_quantize_for_gemm_atomic_final_sg(
                                qkv_x, True, encode_centric
                            )
                        )
                        qkv_candidate_atomic = (
                            module.tk_localcta_quantize_for_gemm_atomic_paired_col_rht(
                                qkv_x, True, encode_centric
                            )
                        )
                        qkv_reference_atomic = _opt_col_rht(
                            module, qkv_x, encode_centric
                        )
                        _check_row("qkv.atomic", qkv_candidate_atomic, qkv_base_atomic)
                        _check_col("qkv.atomic", qkv_candidate_atomic, qkv_reference_atomic)

                        qkv_base_final = module.tk_localcta_quantize_for_gemm_final_sg(
                            qkv_x, True, encode_centric
                        )
                        qkv_candidate_final = (
                            module.tk_localcta_quantize_for_gemm_final_sg_paired_col_rht(
                                qkv_x, True, encode_centric
                            )
                        )
                        qkv_reference_fallback = _opt_col_rht(
                            module, qkv_x, encode_centric
                        )
                        _check_row("qkv.final-sg", qkv_candidate_final, qkv_base_final)
                        _check_col(
                            "qkv.final-sg", qkv_candidate_final,
                            qkv_reference_fallback
                        )
                torch.cuda.synchronize()
                checks += 1

    print(f"exact fused paired-producer gate passed: {checks} cases")


if __name__ == "__main__":
    main()
