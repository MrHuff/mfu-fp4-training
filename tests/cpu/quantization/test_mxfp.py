#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#

import torch
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
)
from collections import OrderedDict

from low_bits_training.utils import JobConfig
from low_bits_training.quantization.mxfp import (
    to_mxfp_dtype,
    MXLinearConfig,
    MXGemmKernelChoice,
    swap_linear_with_quantization_linear,
    MXLinearGeneral,
    ScaleCalculationMode,
)


import torch.nn as nn


def test__mxfp_job_config__default_values():
    config = JobConfig()
    assert "mxfp" not in config.model.converters
    assert config.mxfp.activation_dtype == "e4m3"
    assert config.mxfp.weight_dtype == "e4m3"
    assert config.mxfp.gradient_dtype == "e5m2"
    assert config.mxfp.block_size == 32


def test__to_mxfp_dtype__proper_result():
    assert to_mxfp_dtype("e4m3") == torch.float8_e4m3fn
    assert to_mxfp_dtype("e5m2") == torch.float8_e5m2
    assert to_mxfp_dtype("e3m2") == DTYPE_FP6_E3M2
    assert to_mxfp_dtype("e2m3") == DTYPE_FP6_E2M3
    assert to_mxfp_dtype("e2m1") == torch.float4_e2m1fn_x2


def test__mxfp_linear_config__from_job_config():
    job_config = JobConfig()
    mx_config = MXLinearConfig.from_job_config(job_config)

    assert isinstance(mx_config, MXLinearConfig)
    assert mx_config.adtype == torch.float8_e4m3fn
    assert mx_config.wdtype == torch.float8_e4m3fn
    assert mx_config.gdtype == torch.float8_e5m2
    assert mx_config.block_size == 32
    assert mx_config.scale_rounding_fn == ScaleCalculationMode.FLOOR
    assert mx_config.swap_filter_fn == "all_but_output"
    assert mx_config.gemm_kernel_choice == MXGemmKernelChoice.EMULATED


def test__swap_linear_with_quantization_linear__no_filter():
    mx_config = MXLinearConfig(
        torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fn, block_size=64
    )
    model = nn.Sequential(
        nn.Linear(128, 256, bias=False, dtype=torch.bfloat16), nn.ReLU()
    )
    assert isinstance(model[0], torch.nn.Linear)

    swap_linear_with_quantization_linear(
        model,
        MXLinearGeneral,
        filter_fn=None,
        mx_config=mx_config,
    )
    # MXFP linear layer with proper params.
    assert isinstance(model[0], MXLinearGeneral)
    assert model[0].mx_config == mx_config
    # Not modifying other layer.
    assert isinstance(model[1], nn.ReLU)
    # Make sure it runs!
    input = torch.randn((128, 128), dtype=torch.bfloat16)
    model(input)


def test__swap_linear_with_quantization_linear__with_filter():
    def filter_fn(mod, fqn):
        return "proj" not in fqn

    mx_config = MXLinearConfig(
        torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fn, block_size=64
    )
    model = nn.Sequential(
        OrderedDict(
            [
                ("qkv", nn.Linear(128, 256, bias=False, dtype=torch.bfloat16)),
                ("proj", nn.Linear(256, 10, bias=False, dtype=torch.bfloat16)),
            ]
        )
    )

    swap_linear_with_quantization_linear(
        model,
        MXLinearGeneral,
        filter_fn=filter_fn,
        mx_config=mx_config,
    )
    # Selective swap of Linear layers.
    assert isinstance(model[0], MXLinearGeneral)
    assert model[0].mx_config == mx_config
    assert not isinstance(model[1], MXLinearGeneral)
