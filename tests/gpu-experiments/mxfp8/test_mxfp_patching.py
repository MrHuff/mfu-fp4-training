#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#


import torch
import numpy as np
import pytest

from low_bits_training.experiments.mxfp8.mxfp_patching import (
    to_mx_ao,
    ScaleCalculationMode,
    ScaleCalculationModeAO,
    to_mx,
    scale_round_nearest,
    scale_round_cublas_ceil,
    mx_linear,
    MXLinearConfig,
    DTYPE_FP6_E2M3,
)

import low_bits_training

device = "cuda"


def test__to_mx__equivalent_to_torchao_original():
    scaling_mode = ScaleCalculationMode.EVEN

    def to_mx_ao_fn(v):
        return to_mx_ao(
            v,
            elem_dtype=torch.float8_e4m3fn,
            block_size=len(v),
            scaling_mode=ScaleCalculationModeAO.EVEN,
        )[0]

    def to_mx_fn(v):
        return to_mx(
            v,
            elem_dtype=torch.float8_e4m3fn,
            block_size=len(v),
            scaling_mode=scaling_mode,
        )[0]

    data_in = torch.tensor([1, 2, 4, 7.75], device=device, dtype=torch.bfloat16)

    # Torch compile our version.
    to_mx_fn = torch.compile(to_mx_fn)
    # to_mx_ao_fn = torch.compile(to_mx_ao_fn)

    scale_mx_orig = to_mx_ao_fn(data_in)
    scale_mx = to_mx_fn(data_in)
    # Rounding up `scale` to 8
    assert scale_mx_orig.view(torch.uint8).cpu() == 122
    assert scale_mx.view(torch.uint8).cpu() == 122

    # assert torch.equal(data_mx, data_mx_orig)


@pytest.mark.parametrize(
    "inval, exp_scale",
    [(256, 256), (447, 256), (448, 256), (463, 256), (464, 512), (512, 512)],
)
def test__scale_round_nearest__proper_rounding_e4m3(inval, exp_scale):
    dtype = torch.float8_e4m3fn
    inval = torch.tensor(inval, dtype=torch.float32).to(device)
    scale = scale_round_nearest(inval, dtype)
    assert scale.cpu() == np.float32(exp_scale)


@pytest.mark.parametrize(
    "inval, exp_scale",
    [(57344, 32768), (57345, 32768), (61439, 32768), (61440, 65536)],
)
def test__scale_round_nearest__proper_rounding_e5m2(inval, exp_scale):
    dtype = torch.float8_e5m2
    inval = torch.tensor(inval, dtype=torch.float32).to(device)
    scale = scale_round_nearest(inval, dtype)
    assert scale.cpu() == np.float32(exp_scale)


@pytest.mark.parametrize(
    "inval, exp_scale",
    [(256, 256), (447, 256), (448, 256), (449, 512), (464, 512), (512, 512)],
)
def test__scale_round_cublas_ceil__proper_rounding_e4m3(inval, exp_scale):
    dtype = torch.float8_e4m3fn
    inval = torch.tensor(inval, dtype=torch.float32).to(device)
    scale = scale_round_cublas_ceil(inval, dtype)
    assert scale.cpu() == np.float32(exp_scale)


@pytest.mark.parametrize(
    "inval, exp_scale",
    [(57344, 32768), (57345, 65536), (61439, 65536), (61440, 65536)],
)
def test__scale_round_cublas_ceil__proper_rounding_e5m2(inval, exp_scale):
    dtype = torch.float8_e5m2
    inval = torch.tensor(inval, dtype=torch.float32).to(device)
    scale = scale_round_cublas_ceil(inval, dtype)
    assert scale.cpu() == np.float32(exp_scale)


def test__mx_linear_core__proper_patching():
    assert low_bits_training.quantization.mxfp.mx_linear is mx_linear
    assert (
        low_bits_training.quantization.mxfp.ScaleCalculationMode is ScaleCalculationMode
    )


def test__scale_calculation_mode__compatible_with_torchao():
    values_ao = [v.name for v in ScaleCalculationModeAO]
    values = [v.name for v in ScaleCalculationMode]
    assert values[: len(values_ao)] == values_ao


def test__mx_linear__working_call():
    x = torch.rand((16, 32), dtype=torch.bfloat16, device=device)
    w = torch.rand((48, 32), dtype=torch.bfloat16, device=device)
    mx_config = MXLinearConfig(
        DTYPE_FP6_E2M3,
        DTYPE_FP6_E2M3,
        DTYPE_FP6_E2M3,
        scale_rounding_fn=ScaleCalculationMode.EVEN,
    )
    mx_linear(x, w, None, mx_config)


# def test__ml_dtypes():
#     ml_dtypes.float8_e4m3fn
#     v = ml_dtypes.float8_e4m3fn(8)
#     print(v, np.nextafter(v, 0))
#     print(ml_dtypes.float8_e4m3fn(7.7499))
#     print(ml_dtypes.float8_e4m3fn(7.75))
#     # assert False
