#
# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
#
import torch
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
)

from low_bits_training.utils import JobConfig
from low_bits_training.quantization.mxfp import (
    to_mxfp_dtype,
    swap_linear_with_quantization_linear,
    MXLinearGeneral,
)


def test__mxfp_job_config__default_values():
    config = JobConfig.make_default()
    assert not config.mxfp.enable_mxfp_linear
    assert config.mxfp.activation_dtype == "e4m3"
    assert config.mxfp.weight_dtype == "e4m3"
    assert config.mxfp.gradient_dtype == "e5m2"
    assert config.mxfp.block_size == 32


def test__to_mxfp_dtype__proper_result():
    assert to_mxfp_dtype("e4m3") == torch.float8_e4m3fn
    assert to_mxfp_dtype("e5m2") == torch.float8_e5m2
    assert to_mxfp_dtype("e3m2") == DTYPE_FP6_E3M2
    assert to_mxfp_dtype("e2m3") == DTYPE_FP6_E2M3
    assert to_mxfp_dtype("e2m1") == DTYPE_FP4


def test__swap_linear_with_quantization_linear__no_filter():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x)

    model = TinyModel()
    assert isinstance(model.linear, torch.nn.Linear)
    swap_linear_with_quantization_linear(
        model,
        MXLinearGeneral,
        None,
        adtype=torch.float8_e4m3fn,
        wdtype=torch.float8_e4m3fn,
        gdtype=torch.float8_e4m3fn,
        block_size=64,
    )
    # MXFP linear layer with proper params.
    assert isinstance(model.linear, MXLinearGeneral)
    # TODO: use different dtypes when supported.
    assert model.linear.adtype == torch.float8_e4m3fn
    assert model.linear.wdtype == torch.float8_e4m3fn
    assert model.linear.gdtype == torch.float8_e4m3fn
    assert model.linear.block_size == 64

    # Make sure it runs!
    input = torch.randn((128, 128), dtype=torch.bfloat16)
    model(input)


def test__swap_linear_with_quantization_linear__with_filter():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 256)
            self.proj = torch.nn.Linear(256, 10)

        def forward(self, x):
            return self.proj(self.linear(x))

    def filter_fn(mod, fqn):
        return "proj" not in fqn

    model = TinyModel()
    swap_linear_with_quantization_linear(
        model,
        MXLinearGeneral,
        filter_fn,
        adtype=torch.float8_e4m3fn,
        wdtype=torch.float8_e4m3fn,
        gdtype=torch.float8_e4m3fn,
        block_size=64,
    )
    # Selective swap of Linear layers.
    assert isinstance(model.linear, MXLinearGeneral)
    assert isinstance(model.proj, torch.nn.Linear)


