# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
import functools

import torch
import ml_dtypes
import numpy as np

MLDtype = np.finfo
TorchDType = torch.dtype | str

############################################################################
# Standardised checkpoint statistics
############################################################################


class _Float8E5M3MachArLike:
    def __init__(self):
        minexp = -14
        nmant = 3
        self.smallest_normal = np.float32(2**minexp)
        self.smallest_subnormal = np.float32(2 ** (minexp - nmant))


class finfo(ml_dtypes.finfo):
    """Extension of `finfo` with additional dtypes."""

    __doc__ = ml_dtypes.finfo.__doc__

    @staticmethod
    def _float8_e5m3_finfo():
        fpdtype = np.float32

        def float_to_str(f):
            return "%6.2e" % float(f)

        tiny = fpdtype(2**-14)
        resolution = 0.1
        eps = fpdtype(0.125)
        epsneg = fpdtype(0.125)
        max_ = fpdtype(2 ** (16 - 1) * (1 + 0.5 + 0.25 + 0.125))

        obj = object.__new__(np.finfo)
        obj.dtype = "float8_e5m3"
        obj.bits = 8
        obj.eps = fpdtype(eps)
        obj.epsneg = fpdtype(epsneg)
        # TODO: check
        obj.machep = -2
        obj.negep = -3
        obj.max = fpdtype(max_)
        obj.min = fpdtype(0.0)
        obj.nexp = 5
        obj.nmant = 3
        obj.iexp = obj.nexp
        # Same as FP16, E5M2
        obj.maxexp = 16
        obj.minexp = -14
        obj.precision = 1
        obj.resolution = fpdtype(resolution)
        # pylint: disable=protected-access
        # obj.tiny = fpdtype(tiny)
        # Matching FP16, E5M2 formulas.
        obj._machar = _Float8E5M3MachArLike()
        obj.smallest_subnormal = fpdtype(2 ** (obj.minexp - obj.nmant))

        obj._str_tiny = float_to_str(tiny)
        obj._str_smallest_normal = float_to_str(tiny)
        obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
        obj._str_max = float_to_str(max_)
        obj._str_epsneg = float_to_str(epsneg)
        obj._str_eps = float_to_str(eps)
        obj._str_resolution = float_to_str(resolution)
        # pylint: enable=protected-access
        return obj

    _finfo_factory_map = {
        "float8_e5m3": _float8_e5m3_finfo,
    }

    def __new__(cls, dtype):
        # New dtypes added.
        if dtype in cls._finfo_factory_map:
            return cls._finfo_factory_map[dtype]()

        return super().__new__(cls, dtype)


@functools.cache
def get_dtype_map() -> dict[TorchDType, MLDtype]:
    """Map torch dtypes to ml_dtypes"""
    import torch
    from torchao.prototype.mx_formats.mx_tensor import (
        DTYPE_FP6_E2M3,
        DTYPE_FP6_E3M2,
        DTYPE_FP4,
    )

    return {
        torch.float32: finfo(np.float32),
        torch.float16: finfo(np.float16),
        torch.bfloat16: finfo(ml_dtypes.bfloat16),
        torch.float8_e4m3fn: finfo(ml_dtypes.float8_e4m3fn),
        torch.float8_e5m2: finfo(ml_dtypes.float8_e5m2),
        DTYPE_FP6_E2M3: finfo(ml_dtypes.float6_e2m3fn),
        DTYPE_FP6_E3M2: finfo(ml_dtypes.float6_e3m2fn),
        DTYPE_FP4: finfo(ml_dtypes.float4_e2m1fn),
    }


def from_torch_to_ml_dtypes(torch_dtype: TorchDType) -> MLDtype:
    """From Torch/TorchAO to ML dtypes."""
    dtypes_map = get_dtype_map()
    assert torch_dtype in dtypes_map, f"Unknown Torch(AO) dtype: '{torch_dtype}'."
    return dtypes_map[torch_dtype]
