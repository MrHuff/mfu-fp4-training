#!/usr/bin/env python3
"""Validate correlated data SR for dual-view MXFP4 quantization."""

from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import mxfp4_quant_v4 as mxfp4


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
        (lhs * rhs).mean()
        / (lhs.std(unbiased=False) * rhs.std(unbiased=False))
    )


def main() -> None:
    size = 512
    row_index = torch.arange(size, device="cuda")[:, None]
    col_index = torch.arange(size, device="cuda")[None, :]

    # Every row and column MX block sees the same maximum, making the payload
    # codes directly comparable across the two layouts.
    block_max = (row_index % 32) == (col_index % 32)
    values = torch.full(
        (size, size), 0.3, device="cuda", dtype=torch.bfloat16
    )
    values[block_max] = 1.0

    samples = []
    previous = None
    change_rates = []
    for _ in range(64):
        outputs = mxfp4.mxfp4_quantize_row_and_col_opt(
            values,
            0,
            True,   # data stochastic rounding
            False,  # scale stochastic rounding
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
        "data SR did not advance across calls"
    )

    correlations = []
    offsets = ((0, 2), (2, 0), (0, 4), (4, 0), (0, 8), (8, 0))
    for row_offset, col_offset in offsets:
        lhs = stochastic_samples[:, : size - row_offset, : size - col_offset]
        rhs = stochastic_samples[:, row_offset:, col_offset:]
        valid = (
            ~block_max[: size - row_offset, : size - col_offset]
            & ~block_max[row_offset:, col_offset:]
        )
        correlations.append(_correlation(lhs[:, valid], rhs[:, valid]))

    max_spatial_correlation = max(abs(value) for value in correlations)
    assert max_spatial_correlation < 0.02, (
        f"correlated data SR has structured spatial noise: {correlations}"
    )
    print(
        "MXFP4 correlated dual-view data SR: "
        "agreement=PASS advancing=PASS "
        f"max_spatial_corr={max_spatial_correlation:.6f}"
    )


if __name__ == "__main__":
    main()
