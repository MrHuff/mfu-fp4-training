#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch


def _expected_scale_mean(sigma: float, block_size: int, cutoff_scale: float):
    n = torch.distributions.Normal(loc=0.0, scale=1.0)
    pow2s = torch.arange(-20, 21).exp2()
    y = n.cdf(pow2s * cutoff_scale / sigma)
    y = y * 2 - 1
    z = y**block_size
    probs = z[1:] - z[:-1]
    expected = (probs * pow2s[:-1]).sum()
    return expected


def _best_sigma(rho: float, block_size: int, cutoff_scale, tolerance: float = 1e-6):
    """
    Perform a binary search to find a value for sigma that gives the right scale mean.
    """
    lower_sigma = rho / 4
    upper_sigma = rho * 2

    lower_rho = _expected_scale_mean(lower_sigma, block_size, cutoff_scale)
    upper_rho = _expected_scale_mean(upper_sigma, block_size, cutoff_scale)

    assert lower_rho < rho < upper_rho, (
        lower_rho,
        rho,
        upper_rho,
        lower_sigma,
        upper_sigma,
    )

    while upper_sigma - lower_sigma >= tolerance:
        new_sigma = (lower_sigma + upper_sigma) / 2
        new_rho = _expected_scale_mean(new_sigma, block_size, cutoff_scale)

        if new_rho > rho:
            upper_sigma = new_sigma
            upper_rho = new_rho
        else:
            lower_sigma = new_sigma
            lower_rho = new_rho

    return (lower_sigma + upper_sigma) / 2


def create_lut(
    n_lut_entries: int, scale_dtype, data_dtype, scale_rounding_fn, block_size: int
):
    if scale_dtype != torch.float8_e8m0fnu:
        raise ValueError("LUT-and-LERP only supports E8M0 scale tensors")

    cutoff_scales = {
        torch.float8_e4m3fn: {
            "ocp": 1.0,
            "ceil": 0.5,
            "even": 0.96875,
            "rceil": 0.875,
            "cublas_ceil": 0.875,
        },
        torch.float8_e5m2: {
            "ocp": 1.0,
            "ceil": 0.5,
            "even": 0.9375,
            "rceil": 0.875,
            "cublas_ceil": 0.875,
        },
    }

    cutoff_scale = cutoff_scales[data_dtype][scale_rounding_fn]

    return [
        _best_sigma(2 ** (i / (n_lut_entries)), block_size, cutoff_scale)
        for i in range(n_lut_entries + 1)
    ]
