"""Prove fixed-sign H32 changes only the paired Wgrad column carriers."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("LBT_LIGHT_IMPORT", "1")

import torch


def tensor_bytes_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        raise RuntimeError("byte comparison shape or dtype mismatch")
    if not left.is_contiguous() or not right.is_contiguous():
        raise RuntimeError("byte comparison requires contiguous tensors")
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def equal_pair(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> bool:
    return tensor_bytes_equal(left[0], right[0]) and tensor_bytes_equal(
        left[1], right[1]
    )


def run_gate(rows: int, columns: int, device: str) -> None:
    from low_bits_training.quantization import mxfp4_fused_linear as mx

    if rows < 32 or columns < 32 or rows % 32 or columns % 32:
        raise ValueError("rows and columns must be positive multiples of 32")
    torch.manual_seed(42)
    x = torch.randn((rows, columns), device=device, dtype=torch.bfloat16)

    control = mx.mxfp4_quantize_row_and_col(x, 1)
    treatment = mx.mxfp4_quantize_row_and_col_opt_rht(
        x,
        1,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        rht_axes="col",
        rht_block_size=32,
        with_random_sign_mask=True,
        rng_seed=1234,
        rng_subsequence=0,
    )
    if not equal_pair(control[:2], treatment[:2]):
        raise RuntimeError("activation Fprop row copy changed under column H32")
    if equal_pair(control[2:], treatment[2:]):
        raise RuntimeError("activation Wgrad column copy did not change under H32")

    norm_weight = torch.randn((columns,), device=device, dtype=torch.bfloat16)
    rms_control = mx.mxfp4_fused_rmsnorm_quantize_row_and_col(
        x, norm_weight, 1.0e-5, 1
    )
    rms_treatment = mx.mxfp4_fused_rmsnorm_quantize_row_and_col_opt(
        x,
        norm_weight,
        1.0e-5,
        1,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        use_rht=True,
        row_with_rht=False,
        rht_block_size=32,
        with_random_sign_mask=True,
        rng_seed=1234,
        rng_subsequence=0,
    )
    if not equal_pair(rms_control[:2], rms_treatment[:2]):
        raise RuntimeError("fused RMSNorm Fprop row changed under column H32")
    if not torch.equal(rms_control[4], rms_treatment[4]):
        raise RuntimeError("fused RMSNorm inverse statistic changed under H32")
    if equal_pair(rms_control[2:4], rms_treatment[2:4]):
        raise RuntimeError("fused RMSNorm Wgrad column did not change under H32")

    h1 = torch.randn((rows, columns), device=device, dtype=torch.bfloat16)
    h3 = torch.randn((rows, columns), device=device, dtype=torch.bfloat16)
    silu_control = mx._empty_mxfp4_row_col(rows, columns, x.device)
    silu_treatment = mx._empty_mxfp4_row_col(rows, columns, x.device)
    mx.mxfp4_fused_silu_mul_quantize_row_and_col_launch_inplace(
        h1,
        h3,
        silu_control.row_fp4,
        silu_control.row_sc,
        silu_control.col_fp4,
        silu_control.col_sc,
        1,
    )
    mx.mxfp4_fused_silu_mul_quantize_row_and_col_opt_launch_inplace(
        h1,
        h3,
        silu_treatment.row_fp4,
        silu_treatment.row_sc,
        silu_treatment.col_fp4,
        silu_treatment.col_sc,
        1,
        data_stochastic_rounding=False,
        scale_stochastic_rounding=False,
        use_rht=True,
        row_with_rht=False,
        rht_block_size=32,
        with_random_sign_mask=True,
        rng_seed=1234,
        rng_subsequence=0,
    )
    if not equal_pair(
        (silu_control.row_fp4, silu_control.row_sc),
        (silu_treatment.row_fp4, silu_treatment.row_sc),
    ):
        raise RuntimeError("fused SiLU Fprop row changed under column H32")
    if equal_pair(
        (silu_control.col_fp4, silu_control.col_sc),
        (silu_treatment.col_fp4, silu_treatment.col_sc),
    ):
        raise RuntimeError("fused SiLU Wgrad column did not change under H32")

    row_kwargs = {
        "data_stochastic_rounding": True,
        "scale_stochastic_rounding": False,
        "rng_seed": 1234,
        "rng_subsequence": 0,
    }
    row_a = mx.mxfp4_quantize_for_gemm_opt(x, 1, **row_kwargs)
    row_b = mx.mxfp4_quantize_for_gemm_opt(x, 1, **row_kwargs)
    if not equal_pair(row_a, row_b):
        raise RuntimeError("dY Dgrad row-SR copy changed under exact replay")

    def quantize_column(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        return mx.mxfp4_quantize_col_only_opt_rht(
            x,
            1,
            data_stochastic_rounding=False,
            scale_stochastic_rounding=False,
            rht_axes="col",
            rht_block_size=32,
            with_random_sign_mask=True,
            rng_seed=seed,
            rng_subsequence=0,
        )

    col_a = quantize_column(1234)
    col_b = quantize_column(987654321)
    if not equal_pair(col_a, col_b):
        raise RuntimeError("fixed-sign H32 payload changed with the ignored RNG seed")

    print(
        "[MXFP4 H32 ORIENTATION PASS] block=32 fixed_sign=0x2817 "
        "fprop_row=byte_equal dgrad_row=byte_equal wgrad_columns=transformed "
        "weight_rht=off seed_invariant=true",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run_gate(args.rows, args.columns, args.device)


if __name__ == "__main__":
    main()
