#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import warnings

import numpy as np
import torch
from torch import nn

from torchtitan.tools.logging import logger
from torchtitan.protocols.model_converter import ModelConverter, register_model_converter

import math
from typing import Tuple, Dict, Union, List, Callable, Optional

from torchao.prototype.mx_formats.mx_tensor import to_mx

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims

from low_bits_training.experiments.mx_norm import lut_and_lerp
from low_bits_training.models import get_model_config
from low_bits_training.quantization.mxfp import ScaleCalculationMode

dtypes: Dict[str, torch.dtype] = {
    "e4m3": torch.float8_e4m3fn,
    "e5m2": torch.float8_e5m2,
    "e8m0": torch.float8_e8m0fnu,
}

_MX_ROUNDING_MODES = {
    "ocp": ScaleCalculationMode.FLOOR,
    "floor": ScaleCalculationMode.FLOOR,
    "ceil": ScaleCalculationMode.CEIL,
    "even": ScaleCalculationMode.EVEN,
    "rceil": ScaleCalculationMode.CUBLAS_CEIL,
    "cublas_ceil": ScaleCalculationMode.CUBLAS_CEIL,
    "nearest": ScaleCalculationMode.NEAREST_TIES_UP,
}


def get_largest_pow2(dtype):
    """
    Get the largest power of 2 for the given dtype.
    """
    return int(math.floor(math.log2(torch.finfo(dtype).max)))


if len(_MX_ROUNDING_MODES) < len(ScaleCalculationMode):
    raise Exception(
        "_MX_ROUNDING_MODES does not contain all available modes "
        f"(found: {_MX_ROUNDING_MODES.keys()}, available: {list(ScaleCalculationMode.__members__.keys())})"
    )


def mx_quantise(
    x: torch.Tensor,
    scale_rounding_fn: str,
    block_size: int = 32,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantises a tensor to MX format, represented as a tuple of (scale, data).
    """

    scaling_mode = _MX_ROUNDING_MODES.get(scale_rounding_fn)

    if scaling_mode is None:
        raise ValueError(
            f"{scale_rounding_fn} is not a valid scale rounding mode/function"
        )

    scale, data = to_mx(
        data_hp=x,
        elem_dtype=dtype,
        block_size=block_size,
        scaling_mode=scaling_mode,
    )

    scale_shape = x.shape[:-1] + (x.shape[-1] // block_size, 1)
    scale = scale.reshape(scale_shape).to(torch.float32)
    data_shape = x.shape[:-1] + (x.shape[-1] // block_size, block_size)
    data = data.reshape(data_shape).to(x.dtype)

    return scale, data


def mx_dequantise(scale, data, orig_shape, dtype=torch.float32):
    """
    Dequantises a tensor from MX format, represented as a tuple of (scale, data).
    """
    dequantised = data * scale
    return dequantised.reshape(*orig_shape).to(dtype)


def build_fixed_point_iter_eval_from_lut(initial: float, k: int = 4):
    def eval_fn(x: torch.Tensor, lut: torch.Tensor):
        n_lut_entries = lut.shape[0]
        c = initial
        y = x.mean(dim=-2).div(c).flatten()
        for _ in range(k):
            y_idx = (
                y.log2()
                .fmod(1.0)
                .mul(n_lut_entries)
                .round()
                .int()
                .remainder(n_lut_entries)
            )
            new_c = lut[y_idx]
            y = y.mul(c / new_c)
            c = new_c
        y = y.reshape(x.shape[:-2])[..., None, None]
        return y

    return eval_fn


def build_lut_and_lerp_function():
    def eval_fn(x: torch.Tensor, lut: torch.Tensor):
        lut_n_entries = lut.numel() - 1
        y = x.mean(dim=-2).flatten()

        log2 = y.log2()
        log2_floor = log2.floor()
        log2_frac = log2 - log2_floor

        lut_index_as_float = log2_frac * lut_n_entries

        lut_smaller_index = lut_index_as_float.floor()
        lut_lerp_coefficient = lut_index_as_float - lut_smaller_index

        lut_smaller_index = lut_smaller_index.long()
        lut_larger_index = lut_smaller_index + 1

        output = (
            lut[lut_smaller_index] * (1 - lut_lerp_coefficient)
            + lut[lut_larger_index] * lut_lerp_coefficient
        )
        output *= log2_floor.exp2()
        output = output.reshape(x.shape[:-2])[..., None, None]

        return output

    return eval_fn


def build_linear_scale_function(correction_factor):
    def eval_fn(x: torch.Tensor):
        return x.mean(dim=-2, keepdim=True) * correction_factor

    return eval_fn


def build_mean_absmax_function():
    def eval_fn(x: torch.Tensor):
        return x.mean(dim=-2, keepdim=True)

    return eval_fn


def mx_norm_forward(
    x: torch.Tensor,
    lut: Optional[torch.Tensor],
    eps: float,
    block_size: int,
    sigma_absmax_mapping_fn: Callable,
    scale_rounding_fn: str,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
    data_dtype: torch.dtype = torch.float8_e4m3fn,
):
    """
    MX Norm forward pass.

    Quantises input to MX, estimates RMS from scale tensor, normalises scales,
    and dequantises back to original dtype
    """
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"Tensor of shape {x.shape} cannot be divided into blocks of size {block_size}"
        )
    orig_shape = x.shape
    orig_dtype = x.dtype

    if scale_dtype != torch.float8_e8m0fnu:
        raise NotImplementedError("Only e8m0 scale_dtype implemented at the moment")

    scale, data = mx_quantise(
        x,
        block_size=block_size,
        scale_rounding_fn=scale_rounding_fn,
        dtype=data_dtype,
    )
    if lut is not None:
        sigma_hat = sigma_absmax_mapping_fn(
            scale.mul(2 ** get_largest_pow2(data_dtype)), lut
        )
    else:
        sigma_hat = sigma_absmax_mapping_fn(scale.mul(2 ** get_largest_pow2(data_dtype)))
    scale = scale / (sigma_hat + eps)  # normalise scales only

    output = mx_dequantise(scale, data, orig_shape=orig_shape, dtype=orig_dtype)
    return output


def rms_norm_backward(x, eps, grad_output):
    """
    RMS Norm backward pass.

    Used as a straight through estimator for MX Norm
    """
    m = x.square().mean(dim=-1, keepdim=True) + eps
    p = torch.rsqrt(m)

    dot_term = torch.einsum("...d,...d->...", grad_output, x).unsqueeze(-1)
    x_grad = (-1 / x.shape[-1]) * (m**-1.5) * x * dot_term + p * grad_output

    return x_grad


class MXRMSNormFunction(torch.autograd.Function):
    """
    Approximation to RMSNorm calculated using the MX scale tensor.

    For a fixed block size, we can empirically validate that if `X_i ~ N(0, s^2) (i.i.d.)`
    (where `0 <= i < [block size]`) then `E(mx_scale(X)) ~= f(s) * s` for some function `f` that takes values in a
    relatively narrow range. We can then get a first approximation of `s` as `s_0 = mean(mx_scale(X)) / mean(f)`, then
    iterate on this as `s_i = mean(mx_scale(X)) / f(s_{i-1})`.

    The backward pass is computed as in exact RMSNorm.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        lut,
        block_size,
        eps,
        sigma_absmax_mapping_fn,
        scale_rounding_fn,
        scale_dtype,
        data_dtype,
    ):
        ctx.save_for_backward(x, torch.tensor(eps))
        output = mx_norm_forward(
            x=x,
            lut=lut,
            eps=eps,
            block_size=block_size,
            sigma_absmax_mapping_fn=sigma_absmax_mapping_fn,
            scale_rounding_fn=scale_rounding_fn,
            scale_dtype=scale_dtype,
            data_dtype=data_dtype,
        )

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad_output,) = grad_outputs
        x, eps = ctx.saved_tensors

        x_grad = rms_norm_backward(x, eps, grad_output)

        # inputs to forward() are x, block_size, eps; only x should have a gradient
        return x_grad, None, None, None, None, None, None, None


def calculate_correction_factors(block_size, hidden_dim, n_lut_entries):
    """
    Compute the correction factor LUT and initial correction factor for the fixed-point iteration version of MXRMSNorm.
    """

    correction_factors = []

    eps = 1e-6

    # Dummy second argument for LUT
    def absmax_mean(x, _):
        return x.mean(dim=-2, keepdim=True)

    for i in range(-4 * n_lut_entries, 4 * n_lut_entries):
        mult = 2 ** (i / n_lut_entries)
        data = torch.randn(16384, hidden_dim, device="cuda") * mult
        # initial correction of 1.0 + no iterations -> MX-Norm with no correction
        output = MXRMSNormFunction.apply(
            data,
            [],  # lut can be empty
            block_size,
            eps,
            absmax_mean,
            "ocp",
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
        )

        expected = torch.nn.functional.rms_norm(
            data, normalized_shape=(hidden_dim,), eps=eps
        )

        ratios = output[:, 0] / expected[:, 0]

        if (
            ratios.isnan().sum() + ratios.isinf().sum() + ratios.eq(0).sum()
            > ratios.numel() * 0.01
        ):
            warnings.warn(f">1% of values for {i = } were NaN, Inf, or 0")

        ratios = ratios[~ratios.isnan()]
        ratios = ratios[~ratios.isinf()]
        ratios = ratios[ratios.ne(0)]

        gmean_error = ratios.log().mean().exp()

        correction_factors.append(gmean_error.item())

    correction_factors = np.exp(
        np.mean(-np.log(np.reshape(correction_factors, [8, n_lut_entries])), axis=0)
    )

    initial_correction_factor = np.exp(np.mean(np.log(correction_factors))).item()

    return correction_factors, initial_correction_factor


class MXRMSNorm(nn.Module):
    def __init__(
        self,
        dim,
        block_size,
        eps,
        elementwise_affine=True,
        scale_rounding_fn="ocp",  # only ocp is supported
        scale_dtype=torch.float8_e8m0fnu,
        data_dtype=torch.float8_e4m3fn,
        sigma_absmax_mapping_fn: str = "fixed_point_iter_with_lut",  # only fixed_point_iter_with_lut is supported
        n_fixed_point_iterations: int = 4,
        n_lut_entries: int = 256,
        device=None,
        dtype=None,
    ):
        if dim % block_size != 0:
            raise ValueError(
                f"Dimension {dim} of MXRMSNorm not divisible by block size {block_size}."
            )
        super().__init__()
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty((dim,), device=device, dtype=dtype))
        else:
            self.weight = None
        self.block_size = block_size
        self.eps = eps
        self.scale_rounding_fn = scale_rounding_fn
        self.scale_dtype = scale_dtype
        self.data_dtype = data_dtype

        self.sigma_absmax_mapping_fn_name = sigma_absmax_mapping_fn

        if sigma_absmax_mapping_fn == "fixed_point_iter_with_lut":
            correction_factors, self.initial_correction_factor = (
                calculate_correction_factors(
                    block_size=self.block_size,
                    hidden_dim=dim,
                    n_lut_entries=n_lut_entries,
                )
            )

            self._lut = torch.tensor(correction_factors, device=device)
            self.register_buffer("lut", self._lut)
            self.sigma_absmax_mapping_fn = build_fixed_point_iter_eval_from_lut(
                self.initial_correction_factor, k=n_fixed_point_iterations
            )

        elif sigma_absmax_mapping_fn == "lut_and_lerp":
            self._lut = torch.tensor(
                lut_and_lerp.create_lut(
                    n_lut_entries=n_lut_entries,
                    scale_dtype=scale_dtype,
                    data_dtype=data_dtype,
                    scale_rounding_fn=scale_rounding_fn,
                    block_size=block_size,
                ),
                device=device,
            )
            self.register_buffer("lut", self._lut)
            self.sigma_absmax_mapping_fn = build_lut_and_lerp_function()

        elif sigma_absmax_mapping_fn == "linear_scale":
            # In this case "n_lut_entries" is the number of points used to compute the average correction
            _, correction_factor = calculate_correction_factors(
                block_size=self.block_size, hidden_dim=dim, n_lut_entries=n_lut_entries
            )
            self.sigma_absmax_mapping_fn = build_linear_scale_function(
                correction_factor=correction_factor
            )
            self.lut = None

        elif sigma_absmax_mapping_fn == "mean_absmax":
            self.sigma_absmax_mapping_fn = build_mean_absmax_function()
            self.lut = None
        else:
            raise NotImplementedError(
                "MXRMSNorm only supports '`fixed_point_iter_with_lut' and 'lut_and_lerp' for sigma_absmax_mapping_fn, "
                f"got {sigma_absmax_mapping_fn}"
            )

    def forward(self, x: torch.Tensor):
        normed_x: torch.Tensor = MXRMSNormFunction.apply(
            x,
            self.lut,
            self.block_size,
            self.eps,
            self.sigma_absmax_mapping_fn,
            self.scale_rounding_fn,
            self.scale_dtype,
            self.data_dtype,
        )  # type: ignore
        if self.weight is not None:
            return self.weight * normed_x
        else:
            return normed_x

    def reset_parameters(self):
        nn.init.ones_(self.weight)

        if self.sigma_absmax_mapping_fn_name == "fixed_point_iter_with_lut":
            self.lut.data.copy_(self._lut)

        if self.sigma_absmax_mapping_fn_name == "lut_and_lerp":
            self.lut.data.copy_(self._lut)


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Implementation used by torchtitan before they switched to torch.nn.RMSNorm

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        elementwise_affine (bool): If True, the layer will have learnable parameters. Default is True.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            return output * self.weight
        else:
            return output

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore


def _replace(model: nn.Module, replacement_fn: Callable[[nn.Module], nn.Module]):
    for name, child in model.named_children():
        if isinstance(child, nn.RMSNorm):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            _replace(child, replacement_fn)


def swap_norm_with_custom_norm(model: nn.Module, norm_cls: nn.Module, **kwargs):
    """
    Swaps RMSNorm layers in the model with MXRMSNorm or custom RMSNorm layers
    """
    _replace(model, lambda mod: norm_cls(**kwargs))


class MXRMSNormConverter(ModelConverter):
    """
    Converts RMSNorm layers to MXRMSNorm layers
    """

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        model_config = get_model_config(job_config.model.name, job_config.model.flavor)

        self._dim: int = int(model_config.dim)
        self._eps: float = float(model_config.norm_eps)
        self._elementwise_affine: bool = job_config.mx_rmsnorm.elementwise_affine
        self._block_size: int = int(job_config.mx_rmsnorm.block_size)
        self._scale_rounding_fn: str = job_config.mx_rmsnorm.scale_rounding_fn
        self._scale_dtype: torch.dtype = dtypes.get(
            job_config.mx_rmsnorm.scale_dtype, torch.float8_e8m0fnu
        )
        self._data_dtype: torch.dtype = dtypes.get(
            job_config.mx_rmsnorm.data_dtype, torch.float8_e4m3fn
        )
        self._sigma_absmax_mapping_fn: str = job_config.mx_rmsnorm.sigma_absmax_mapping_fn
        self._n_fixed_point_iterations: int = (
            job_config.mx_rmsnorm.n_fixed_point_iterations
        )
        self._n_lut_entries: int = job_config.mx_rmsnorm.n_lut_entries

    def convert(self, model: nn.Module):
        logger.info(
            f"MXRMSNorm converter (eps={self._eps}, dim={self._dim}, "
            f"elementwise_affine={self._elementwise_affine}, "
            f"block_size={self._block_size}, "
            f"scale_rounding={self._scale_rounding_fn}, "
            f"scale_dtype={self._scale_dtype}, "
            f"data_dtype={self._data_dtype}, "
            f"sigma_absmax_mapping={self._sigma_absmax_mapping_fn}, "
            f"n_fixed_point_iterations={self._n_fixed_point_iterations}, "
            f"n_lut_entries={self._n_lut_entries})",
        )
        if self._sigma_absmax_mapping_fn == "rms":
            swap_norm_with_custom_norm(
                model,
                norm_cls=RMSNorm,
                dim=self._dim,
                eps=self._eps,
                elementwise_affine=self._elementwise_affine,
            )
        else:
            swap_norm_with_custom_norm(
                model,
                norm_cls=MXRMSNorm,
                dim=self._dim,
                block_size=self._block_size,
                eps=self._eps,
                elementwise_affine=self._elementwise_affine,
                scale_rounding_fn=self._scale_rounding_fn,
                scale_dtype=self._scale_dtype,
                data_dtype=self._data_dtype,
                sigma_absmax_mapping_fn=self._sigma_absmax_mapping_fn,
                n_fixed_point_iterations=self._n_fixed_point_iterations,
                n_lut_entries=self._n_lut_entries,
            )

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        # Nothing to do
        pass


# Register the MXRMSNormConverter
register_model_converter(MXRMSNormConverter, "mx_rmsnorm")
