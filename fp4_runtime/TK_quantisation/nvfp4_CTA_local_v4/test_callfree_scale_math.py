#!/usr/bin/env python3
"""CUDA exactness gate for the paired-RHT leaf scale division."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("_tk_quant_localcta_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bits(value: torch.Tensor) -> torch.Tensor:
    return value.view(torch.int32)


def _assert_exact_finite(
    name: str,
    precise: torch.Tensor,
    callfree: torch.Tensor,
) -> None:
    finite = torch.isfinite(precise)
    mismatch = finite & (_bits(precise) != _bits(callfree))
    if bool(mismatch.any()):
        indices = mismatch.nonzero().flatten()[:8]
        raise AssertionError(
            f"{name}: {int(mismatch.sum().item())} finite bit mismatches; "
            f"first={indices.tolist()}"
        )
    if not torch.equal(torch.isnan(precise), torch.isnan(callfree)):
        raise AssertionError(f"{name}: NaN classification mismatch")
    precise_inf = torch.isinf(precise)
    if not torch.equal(precise_inf, torch.isinf(callfree)):
        raise AssertionError(f"{name}: infinity classification mismatch")
    if bool(precise_inf.any()) and not torch.equal(
        torch.signbit(precise[precise_inf]), torch.signbit(callfree[precise_inf])
    ):
        raise AssertionError(f"{name}: infinity sign mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    module = _load(args.extension.resolve())

    # Every positive finite BF16 carrier is a possible local/outer amax.  The
    # three numerators cover outer encoding, block encoding, and reciprocal
    # scale paths used by the production specialization.
    bf16_bits = torch.arange(1, 0x7F80, device="cuda", dtype=torch.int32)
    positive_bf16 = bf16_bits.to(torch.uint16).view(torch.bfloat16).float()
    denominator = positive_bf16.repeat(3).contiguous()
    numerator = torch.cat(
        (
            torch.full_like(positive_bf16, 1493.0),
            torch.full_like(positive_bf16, 6.0),
            torch.ones_like(positive_bf16),
        )
    ).contiguous()
    precise, callfree = module.tk_localcta_test_scale_divide_callfree(
        numerator, denominator
    )
    torch.cuda.synchronize()
    _assert_exact_finite("positive-bf16-scale-domain", precise, callfree)

    # The production paired-RHT column route is decode-centric.  Preserve the
    # first operation in TE's `block_amax / fp4_max * S_enc` formula exactly;
    # the full multiply/order/cap and E4M3 conversion are covered end-to-end
    # by test_fused_paired_producers.py.
    decode_denominator = torch.full_like(positive_bf16, 6.0)
    precise, callfree = module.tk_localcta_test_scale_divide_callfree(
        positive_bf16.contiguous(), decode_denominator.contiguous()
    )
    torch.cuda.synchronize()
    _assert_exact_finite("decode-centric-block-amax-over-six", precise, callfree)

    # CUDA's exact helper is reached for exceptional rounding/range cases, not
    # merely BF16 inputs.  Sweep deterministic normal/subnormal FP32 bit
    # patterns and the products that the production row/column scale formulas
    # actually divide by.
    count = 1 << 20
    index = torch.arange(count, device="cuda", dtype=torch.int64)
    denominator_bits = (
        ((1 + (index % 254)) << 23)
        | ((index * 0x45D9F3B) & 0x7FFFFF)
    ).to(torch.int32)
    numerator_bits = (
        ((1 + ((index * 37) % 254)) << 23)
        | ((index * 0x119DE1F3) & 0x7FFFFF)
    ).to(torch.int32)
    broad_denominator = denominator_bits.view(torch.float32).contiguous()
    broad_numerator = numerator_bits.view(torch.float32).contiguous()
    precise, callfree = module.tk_localcta_test_scale_divide_callfree(
        broad_numerator, broad_denominator
    )
    torch.cuda.synchronize()
    _assert_exact_finite("broad-positive-fp32-domain", precise, callfree)

    production_amax = positive_bf16[::17].contiguous()
    production_block = positive_bf16.flip(0)[::17].contiguous()
    production_s_enc = 1493.0 / production_amax
    production_product = (production_block * production_s_enc).contiguous()
    production_numerator = torch.full_like(production_product, 6.0)
    precise, callfree = module.tk_localcta_test_scale_divide_callfree(
        production_numerator, production_product
    )
    torch.cuda.synchronize()
    _assert_exact_finite("production-block-product-domain", precise, callfree)

    edge_numerator = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, float("inf"), -float("inf"), float("nan"), 1.0],
        device="cuda",
        dtype=torch.float32,
    )
    edge_denominator = torch.tensor(
        [0.0, 0.0, 0.0, -0.0, float("inf"), float("inf"), 1.0, float("nan")],
        device="cuda",
        dtype=torch.float32,
    )
    precise, callfree = module.tk_localcta_test_scale_divide_callfree(
        edge_numerator, edge_denominator
    )
    torch.cuda.synchronize()
    _assert_exact_finite("zero-inf-nan-edges", precise, callfree)
    print(
        "call-free scale math gate passed: "
        f"bf16_cases={denominator.numel()} broad_fp32_cases={count} "
        f"decode_cases={decode_denominator.numel()} "
        f"product_cases={production_product.numel()} "
        f"edge_cases={edge_numerator.numel()}"
    )


if __name__ == "__main__":
    main()
