#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import pytest
from collections import OrderedDict

import torch

from low_bits_training.experiments.mx_norm.mx_norm import (
    mx_quantise,
    get_largest_pow2,
    calculate_correction_factors,
    build_fixed_point_iter_eval_from_lut,
    MXRMSNorm,
    swap_norm_with_custom_norm,
)

from low_bits_training.models.llama3 import llama3_gc_configs, Transformer


def test__mx_norm_impl_fixed_point_iter_eval_from_lut():
    def estimate_sigma_from_scale(
        scale: torch.Tensor,
        initial_correction_factor: torch.Tensor,
        all_corrections: torch.Tensor,
        iterations: int,
        mxdtype,
    ) -> torch.Tensor:
        """
        Deprecated implementation
        """

        correction_factor = initial_correction_factor
        sigma_hat = (
            scale.mul(2 ** get_largest_pow2(mxdtype)).mean(dim=-2) / correction_factor
        ).flatten()

        for _ in range(iterations):
            sigma_index = (
                ((sigma_hat.log2().fmod(1.0)) * 256).round().int().remainder(256)
            )
            new_correction_factor = all_corrections[sigma_index]
            sigma_hat *= correction_factor / new_correction_factor
            correction_factor = new_correction_factor

        sigma_hat = sigma_hat.reshape(scale.shape[:-2])[..., None, None]
        return sigma_hat

    # hyperparameters
    block_size = 32
    iters = 4
    data_dtype = torch.float8_e4m3fn
    hidden_dim = 1024

    # precomputed correction factors
    corrections_data, initial_correction_factor = calculate_correction_factors(
        block_size=block_size, hidden_dim=hidden_dim, n_lut_entries=256
    )
    lut = torch.tensor(corrections_data)

    # new implementation
    new_impl = build_fixed_point_iter_eval_from_lut(initial_correction_factor, k=iters)

    # generate data
    N, D = 16, 1024
    torch.manual_seed(1472)
    x = torch.randn(N, D) * torch.arange(1, N + 1)[:, None].sqrt()

    scale, _ = mx_quantise(
        x, scale_rounding_fn="ocp", block_size=block_size, dtype=data_dtype
    )
    ref = estimate_sigma_from_scale(
        scale=scale,
        initial_correction_factor=initial_correction_factor,
        all_corrections=lut,
        iterations=iters,
        mxdtype=data_dtype,
    )
    new = new_impl(2 ** get_largest_pow2(data_dtype) * scale, lut)

    assert ref.sub(new).abs().max() < 1e-4


def test__swap_norm_with_mx_rmsnorm():
    model = torch.nn.Sequential(
        OrderedDict(
            [
                ("fc", torch.nn.Linear(128, 256, bias=False, dtype=torch.bfloat16)),
                ("norm", torch.nn.RMSNorm((256,), eps=1e-5)),
                (
                    "seq",
                    torch.nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "fc",
                                    torch.nn.Linear(
                                        256, 256, bias=False, dtype=torch.bfloat16
                                    ),
                                ),
                                ("norm", torch.nn.RMSNorm((256,), eps=1e-5)),
                            ]
                        )
                    ),
                ),
            ]
        )
    )
    swap_norm_with_custom_norm(
        model,
        norm_cls=MXRMSNorm,
        dim=256,
        block_size=32,
        eps=1e-5,
        scale_rounding_fn="ocp",
        scale_dtype=torch.float8_e8m0fnu,
        data_dtype=torch.float8_e4m3fn,
        sigma_absmax_mapping_fn="fixed_point_iter_with_lut",
    )
    assert not isinstance(model[0], MXRMSNorm)
    assert isinstance(model[1], MXRMSNorm)
    assert isinstance(model[2][1], MXRMSNorm)


@pytest.mark.parametrize(
    "method", ["fixed_point_iter_with_lut", "lut_and_lerp", "linear_scale", "mean_absmax"]
)
def test__swap_norm_with_mx_rmsnorm_llama(method):
    model_args = llama3_gc_configs["debugmodel"]
    model_args.vocab_size = 1024
    model = Transformer(model_args)
    swap_norm_with_custom_norm(
        model,
        norm_cls=MXRMSNorm,
        dim=model_args.dim,
        eps=model_args.norm_eps,
        block_size=32,
        scale_rounding_fn="ocp",
        scale_dtype=torch.float8_e8m0fnu,
        data_dtype=torch.float8_e4m3fn,
        sigma_absmax_mapping_fn=method,
    )
    assert isinstance(model.norm, MXRMSNorm)
    assert isinstance(model.layers["1"].attention_norm, MXRMSNorm)
    assert isinstance(model.layers["1"].ffn_norm, MXRMSNorm)
