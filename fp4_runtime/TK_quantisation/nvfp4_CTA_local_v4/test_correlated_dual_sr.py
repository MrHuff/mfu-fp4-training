#!/usr/bin/env python3
"""Validate correlated fast data SR for dual-view localCTA quantization."""

import os
from pathlib import Path
import sys

import torch


os.environ["USE_TK_LOCALCTA_V4_FAST_DATA_SR"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tk_quant_localcta_v4 as tkq


def _unpack_fp4(tensor: torch.Tensor) -> torch.Tensor:
    packed = tensor.view(torch.uint8)
    return torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(
        packed.shape[0], -1
    )


def _correlation(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs = lhs.float()
    rhs = rhs.float()
    lhs = lhs - lhs.mean()
    rhs = rhs - rhs.mean()
    return float(
        (lhs * rhs).mean() /
        (lhs.std(unbiased=False) * rhs.std(unbiased=False))
    )


def _assert_axis_isolation(values: torch.Tensor) -> None:
    outputs = {}
    for axes in ("none", "row", "col", "both"):
        outputs[axes] = tkq.tk_localcta_quantize_for_gemm_opt(
            values,
            True,
            True,
            axes != "none",
            False,
            "none",
            False,
            123,
            456,
            axes,
        )
    torch.cuda.synchronize()

    def same(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
        return torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))

    assert not same(outputs["row"][0], outputs["none"][0])
    assert same(outputs["row"][2], outputs["none"][2])
    assert same(outputs["col"][0], outputs["none"][0])
    assert not same(outputs["col"][2], outputs["none"][2])
    assert not same(outputs["both"][0], outputs["none"][0])
    assert not same(outputs["both"][2], outputs["none"][2])


def main() -> None:
    size = 512
    row_index = torch.arange(size, device="cuda")[:, None]
    col_index = torch.arange(size, device="cuda")[None, :]

    # Every row and column quantization block sees the same maximum, making
    # row/column FP4 payload codes directly comparable.
    block_max = (row_index % 16) == (col_index % 16)
    values = torch.full(
        (size, size), 0.28, device="cuda", dtype=torch.bfloat16
    )
    values[block_max] = 1.0
    _assert_axis_isolation(values)

    samples = []
    previous = None
    change_rates = []
    for _ in range(64):
        outputs = tkq.tk_localcta_quantize_for_gemm_opt(
            values,
            True,
            True,
            True,   # data stochastic rounding
            False,  # scale stochastic rounding
            "none",
            False,
            123,
            456,
        )
        row_codes = _unpack_fp4(outputs[0])
        col_codes = _unpack_fp4(outputs[2]).T.contiguous()
        assert torch.equal(row_codes, col_codes), (
            "dual-view data SR used inconsistent row/column rounding draws"
        )
        if previous is not None:
            change_rates.append(float((row_codes != previous).float().mean()))
        previous = row_codes
        samples.append((row_codes == 4) & ~block_max)

    stochastic_samples = torch.stack(samples)
    torch.cuda.synchronize()
    assert sum(change_rates) / len(change_rates) > 0.01, (
        "fast data SR did not advance across calls"
    )

    correlations = []
    for row_offset, col_offset in ((0, 2), (2, 0), (0, 4), (4, 0), (0, 8), (8, 0)):
        lhs = stochastic_samples[:, : size - row_offset, : size - col_offset]
        rhs = stochastic_samples[:, row_offset:, col_offset:]
        valid = (
            ~block_max[: size - row_offset, : size - col_offset]
            & ~block_max[row_offset:, col_offset:]
        )
        correlations.append(_correlation(lhs[:, valid], rhs[:, valid]))

    assert max(abs(value) for value in correlations) < 0.02, (
        f"correlated data SR has structured spatial noise: {correlations}"
    )
    print(
        "localCTA correlated dual-view data SR: "
        f"axes=PASS agreement=PASS advancing=PASS max_spatial_corr="
        f"{max(abs(value) for value in correlations):.6f}"
    )


if __name__ == "__main__":
    main()
