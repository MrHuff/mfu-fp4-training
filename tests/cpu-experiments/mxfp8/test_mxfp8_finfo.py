#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import ml_dtypes
import numpy as np

import torch
import pytest


def test__finfo__float8_e5m3():
    from low_bits_training.experiments.mxfp8.finfo import finfo as mx_finfo

    finfo = mx_finfo("float8_e5m3")

    assert finfo.maxexp == 16
    assert finfo.minexp == -14

    assert finfo.max == 2 ** (16 - 1) * (1 + 0.5 + 0.25 + 0.125)
    assert finfo.eps == 0.125

    assert finfo.smallest_normal == 2**-14
    assert finfo.smallest_subnormal == 2**-17


def test__finfo__numpy_ml_dtypes_compatible():
    from low_bits_training.experiments.mxfp8.finfo import finfo as mx_finfo

    assert mx_finfo(ml_dtypes.bfloat16) is ml_dtypes.finfo(ml_dtypes.bfloat16)
    assert mx_finfo(np.float16) is ml_dtypes.finfo(np.float16)


def test__from_torch_to_ml_dtypes__e4m3fn():
    from low_bits_training.experiments.mxfp8.finfo import finfo as mx_finfo
    from low_bits_training.experiments.mxfp8 import from_torch_to_ml_dtypes
    from torchao.prototype.mx_formats.mx_tensor import (
        F8E4M3_MAX_POW2,
        MBITS_F8_E4M3,
        F8E4M3_MAX,
    )

    fi = mx_finfo(from_torch_to_ml_dtypes(torch.float8_e4m3fn))
    assert fi.nmant == MBITS_F8_E4M3
    assert fi.max == F8E4M3_MAX
    assert fi.maxexp - 1 == F8E4M3_MAX_POW2


def test__from_torch_to_ml_dtypes__e5m2():
    from low_bits_training.experiments.mxfp8.finfo import finfo as mx_finfo
    from low_bits_training.experiments.mxfp8 import from_torch_to_ml_dtypes

    from torchao.prototype.mx_formats.mx_tensor import (
        F8E5M2_MAX_POW2,
        MBITS_F8_E5M2,
        F8E5M2_MAX,
    )

    fi = mx_finfo(from_torch_to_ml_dtypes(torch.float8_e5m2))
    assert fi.nmant == MBITS_F8_E5M2
    assert fi.max == F8E5M2_MAX
    assert fi.maxexp - 1 == F8E5M2_MAX_POW2


def test__from_torch_to_ml_dtypes__e2m1():
    from low_bits_training.experiments.mxfp8.finfo import finfo as mx_finfo
    from low_bits_training.experiments.mxfp8 import from_torch_to_ml_dtypes

    from torchao.prototype.mx_formats.mx_tensor import (
        F4_E2M1_MAX_POW2,
        MBITS_F4_E2M1,
        F4_E2M1_MAX,
    )

    fi = mx_finfo(from_torch_to_ml_dtypes(torch.float4_e2m1fn_x2))
    assert fi.nmant == MBITS_F4_E2M1
    assert fi.max == F4_E2M1_MAX
    assert fi.maxexp - 1 == F4_E2M1_MAX_POW2


@pytest.mark.parametrize(
    "dtype_in",
    [
        ml_dtypes.bfloat16,
        ml_dtypes.float8_e4m3fn,
        ml_dtypes.float8_e5m2,
        ml_dtypes.float4_e2m1fn,
        np.float16,
        np.float32,
    ],
)
def test__from_ml_dtypes_to_torch__round_trip(dtype_in):
    from low_bits_training.experiments.mxfp8 import (
        from_torch_to_ml_dtypes,
        from_ml_dtypes_to_torch,
    )

    torch_dtype = from_ml_dtypes_to_torch(dtype_in)
    dtype_out = from_torch_to_ml_dtypes(torch_dtype)
    assert dtype_out is dtype_in
