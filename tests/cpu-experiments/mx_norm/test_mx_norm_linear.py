#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import pytest
from unittest.mock import Mock
import torch

from low_bits_training.quantization.mxfp import MXLinearGeneral
from low_bits_training.models.llama3 import llama3_gc_configs, Transformer
from low_bits_training.config import ConfigManager

from torchtitan.distributed.parallel_dims import ParallelDims
from torchao.prototype.mx_formats.mx_tensor import MXTensor, to_dtype


def test_monkeypatch():
    import low_bits_training.experiments.mx_norm as mxn
    import low_bits_training.quantization.mxfp as mxfp
    import torchao.prototype.mx_formats as mxao

    assert mxao.mx_tensor.to_mx == mxn.mxfp_patching.to_mx
    assert mxfp.mx_linear == mxn.mxfp_patching.mx_linear
    assert mxfp.ScaleCalculationMode == mxn.mxfp_patching.ScaleCalculationMode


def test__swap_norm_linear_with_mx_norm_linear_llama():
    from low_bits_training.experiments.mx_norm.norm_linear import (
        swap_norm_linear_with_mx_norm_linear,
        MXNorm,
        MXNormLinear,
        NormMode,
        Reduction,
    )

    model_args = llama3_gc_configs["debugmodel"]
    model_args.vocab_size = 1024
    model = Transformer(model_args)

    swap_norm_linear_with_mx_norm_linear(
        model,
        norm_linear_cls=MXNormLinear,
        filter_fn=lambda mod, fqn: fqn
        not in ["output", "norm", "attention.wo", "feed_forward.w_out"],
        adtype=torch.float8_e4m3fn,
        wdtype=torch.float8_e4m3fn,
        gdtype=torch.float8_e5m2,
        block_size=32,
        scale_rounding="rceil",
        reduction=Reduction.MEAN,
        clamp_val=32,
        norm_mode=NormMode.POST,
        norm_kwargs=dict(eps=1e-5, n_lut_entries=256),
    )
    assert isinstance(model.norm, torch.nn.RMSNorm)
    assert isinstance(model.output, torch.nn.Linear)
    assert isinstance(model.layers["1"].attention_norm, MXNorm)
    assert isinstance(model.layers["1"].ffn_norm, MXNorm)
    assert isinstance(model.layers["1"].attention.wq, MXNormLinear)
    assert isinstance(model.layers["1"].attention.wo, torch.nn.Linear)


@pytest.mark.parametrize("method", ["pre", "post"])
def test__mx_norm_linear_converter(method):
    from low_bits_training.experiments.mx_norm.norm_linear import (
        MXNorm,
        MXNormLinear,
        MXNormLinearConverter,
    )

    config = ConfigManager().parse_args([])

    config.model.converters = ["mx_norm_linear"]
    config.mxfp.block_size = 32
    config.mxfp.scale_rounding_fn = "rceil"
    config.mxfp.activation_dtype = "e4m3"
    config.mxfp.weight_dtype = "e4m3"
    config.mxfp.gradient_dtype = "e5m2"
    config.mx_norm_linear.norm_mode = method
    config.mx_norm_linear.n_lut_entries = 256

    model_args = llama3_gc_configs["debugmodel"]
    model_args.vocab_size = 1024
    model = Transformer(model_args)

    converter = MXNormLinearConverter(config, Mock(spec=ParallelDims))

    converter.convert(model)
    assert isinstance(model.norm, torch.nn.RMSNorm)
    assert isinstance(model.output, torch.nn.Linear)
    assert isinstance(model.layers["1"].attention_norm, MXNorm)
    assert isinstance(model.layers["1"].ffn_norm, MXNorm)
    assert isinstance(model.layers["1"].attention.wqkv, MXNormLinear)
    assert isinstance(model.layers["1"].attention.wo, MXLinearGeneral)
    assert isinstance(model.layers["1"].feed_forward.w_in, MXNormLinear)
    assert isinstance(model.layers["1"].feed_forward.w_out, MXLinearGeneral)


def generate_inputs(block_size):
    scale = 1.0
    normal = torch.distributions.Normal(0.0, scale=scale)

    n_sigmas = 2**10
    hidden_dim = 2**11
    block_count = hidden_dim // block_size

    x = normal.sample((n_sigmas, block_count * block_size))
    sigmas = torch.logspace(-3.5, 3.5, n_sigmas, base=2)
    x = x * sigmas[:, None]
    return x


def _check(target, actual, tol):
    error = ((target - actual) / (target + 1e-5)).abs().mean()
    assert error < tol


@pytest.mark.parametrize(
    "norm_mode,reduction", [(n, r) for n in ["pre", "post"] for r in ["mean", "rms"]]
)
def test__to_mx_with_norm_approximation(norm_mode, reduction):
    from low_bits_training.experiments.mx_norm.mxfp_patching import to_mx
    from low_bits_training.experiments.mx_norm.norm_linear import (
        ScaleCalculationMode,
        Reduction,
        NormMode,
        to_mx_with_norm,
        get_max_abs_pre_round_norm_scaling,
    )

    from low_bits_training.experiments.mx_norm import lut_and_lerp

    torch.manual_seed(1472)
    block_size = 2**5
    x = generate_inputs(block_size)
    hidden_dim = x.shape[-1]
    elem_dtype = torch.float8_e4m3fn
    norm_eps = 1e-6
    norm_kwargs = {"eps": norm_eps}

    with torch.no_grad():
        ref_rms = torch.sqrt(torch.mean(x.square(), keepdim=True, dim=1) + norm_eps)
        normed_x = torch.nn.functional.rms_norm(
            x, normalized_shape=(hidden_dim,), eps=norm_eps
        )

    to_mx_kwargs = dict(
        elem_dtype=elem_dtype,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.CUBLAS_CEIL,
        pack_fp6=False,
    )
    to_dtype_kwargs = dict(
        elem_dtype=elem_dtype,
        block_size=block_size,
        target_dtype=torch.float32,
        pack_fp6=False,
    )

    ref_x_scale, ref_x_data = to_mx(normed_x, **to_mx_kwargs)

    if norm_mode == "post":
        norm_kwargs["lut"] = torch.tensor(
            lut_and_lerp.create_lut(
                n_lut_entries=256,
                scale_dtype=torch.float8_e8m0fnu,
                data_dtype=torch.float8_e4m3fn,
                scale_rounding_fn="rceil",
                block_size=block_size,
            ),
        )
    elif norm_mode == "pre":
        norm_kwargs["scale"] = torch.tensor(
            get_max_abs_pre_round_norm_scaling(block_size, Reduction[reduction.upper()]),
        )

    new_x_scale, new_x_data, new_rms = to_mx_with_norm(
        x,
        norm_mode=NormMode[norm_mode.upper()],
        reduction=Reduction[reduction.upper()],
        norm_kwargs=norm_kwargs,
        **to_mx_kwargs,
    )

    ref_x = to_dtype(ref_x_data, ref_x_scale, **to_dtype_kwargs)
    new_x = to_dtype(new_x_data, new_x_scale, **to_dtype_kwargs)

    tol = 0.1
    _check(ref_rms, new_rms, tol)
    _check(ref_x, new_x, tol)


@pytest.mark.parametrize(
    "norm_mode,reduction", [(n, r) for n in ["pre", "post"] for r in ["mean", "rms"]]
)
def test__mx_mm_with_affine_norm(norm_mode, reduction):
    from low_bits_training.experiments.mx_norm.norm_linear import (
        MXGemmKernelChoice,
        mx_mm as mx_mm_lp,
        get_max_abs_pre_round_norm_scaling,
        Reduction,
    )
    from low_bits_training.experiments.mx_norm import lut_and_lerp
    from low_bits_training.experiments.mx_norm.mxfp_patching import (
        mx_mm as mx_mm_pb,
        ScaleCalculationMode,
    )

    torch.manual_seed(1472)
    block_size = 2**5
    batch_size = 2**2
    with torch.no_grad():
        x = generate_inputs(block_size)
        hidden_dim = x.shape[-1]
        x = x.reshape(batch_size, x.shape[0] // batch_size, hidden_dim)
    x.requires_grad_(True)
    out_dim = hidden_dim // 2
    norm_eps = 1e-6
    norm_kwargs = {"eps": norm_eps}

    in_elem_dtype = torch.float8_e4m3fn
    w_elem_dtype = torch.float8_e4m3fn
    grad_elem_dtype = torch.float8_e5m2

    norm_weight = torch.randn((hidden_dim,))
    weight = torch.randn((out_dim, hidden_dim))
    with torch.no_grad():
        norm_weight = norm_weight * 0.1 + 1
        weight = weight * hidden_dim**-0.5
    norm_weight.requires_grad_(True)
    weight.requires_grad_(True)

    normed_x = torch.nn.functional.rms_norm(
        x, normalized_shape=(hidden_dim,), eps=norm_eps
    )

    with torch.no_grad():
        # test norm_weight fused with weight
        out = mx_mm_pb.apply(
            normed_x,
            norm_weight * weight,
            in_elem_dtype,
            w_elem_dtype,
            grad_elem_dtype,
            block_size,
            MXGemmKernelChoice.EMULATED,
            ScaleCalculationMode.CUBLAS_CEIL,
        )

    _out = mx_mm_pb.apply(
        normed_x * norm_weight,
        weight,
        in_elem_dtype,
        w_elem_dtype,
        grad_elem_dtype,
        block_size,
        MXGemmKernelChoice.EMULATED,
        ScaleCalculationMode.CUBLAS_CEIL,
    )

    grad = torch.randn_like(_out)
    _out.backward(grad)

    ref_out = out.detach().clone()
    ref_x_grad = x.grad.detach().clone()
    ref_weight_grad = weight.grad.detach().clone()
    ref_norm_weight_grad = norm_weight.grad.detach().clone()

    x.grad.zero_()
    weight.grad.zero_()
    norm_weight.grad.zero_()

    norm_kwargs = {"eps": norm_eps}
    if norm_mode == "post":
        norm_kwargs["lut"] = torch.tensor(
            lut_and_lerp.create_lut(
                n_lut_entries=256,
                scale_dtype=torch.float8_e8m0fnu,
                data_dtype=torch.float8_e4m3fn,
                scale_rounding_fn="rceil",
                block_size=block_size,
            ),
        )
    elif norm_mode == "pre":
        norm_kwargs["scale"] = torch.tensor(
            get_max_abs_pre_round_norm_scaling(block_size, Reduction[reduction.upper()]),
        )

    normed_x = torch.nn.functional.rms_norm(
        x, normalized_shape=(hidden_dim,), eps=norm_eps
    )

    normed_x_mx = MXTensor.to_mx(
        normed_x.reshape(-1, hidden_dim),
        elem_dtype=in_elem_dtype,
        block_size=block_size,
        scaling_mode=ScaleCalculationMode.CUBLAS_CEIL,
    )

    out = mx_mm_lp.apply(
        normed_x,
        normed_x_mx,
        weight,
        norm_weight,
        in_elem_dtype,
        w_elem_dtype,
        grad_elem_dtype,
        block_size,
        MXGemmKernelChoice.EMULATED,
        ScaleCalculationMode.CUBLAS_CEIL,
    )

    out.backward(grad)

    new_out = out.detach().clone()

    assert x.grad.dtype in [torch.bfloat16, torch.float32]

    new_x_grad = x.grad.detach().clone()
    new_weight_grad = weight.grad.detach().clone()
    new_norm_weight_grad = norm_weight.grad.detach().clone()

    _check(ref_out, new_out, 1e-5)
    _check(ref_x_grad, new_x_grad, 1e-5)
    _check(ref_weight_grad, new_weight_grad, 0.4)  # difference due to fused norm_weight
    _check(ref_norm_weight_grad, new_norm_weight_grad, 1e-5)


@pytest.mark.parametrize("norm_mode", ["pre", "post"])
def test_mx_norm_mx_mm_against_old_impl(norm_mode):
    from low_bits_training.experiments.mx_norm.norm_linear import (
        MXGemmKernelChoice,
        NormMode,
        Reduction,
        mx_mm,
        mx_norm,
        get_max_abs_pre_round_norm_scaling,
        to_mx_with_norm,
    )
    from low_bits_training.experiments.mx_norm import lut_and_lerp
    from low_bits_training.experiments.mx_norm.mxfp_patching import (
        ScaleCalculationMode,
    )
    from typing import Dict, Any

    # =============================================== #
    # ==== Old implementation to compare against ==== #
    # =============================================== #

    @torch._dynamo.allow_in_graph
    class mx_mm_with_affine_norm(torch.autograd.Function):
        # There are three gemms in a forward + backward of a Linear layer:
        #
        # 1.       input @ weight_t    = output     (forward pass)
        # 2. grad_output @ weight      = grad_input (backward pass)
        # 3.     input_t @ grad_output = grad_weight (backward pass)
        #
        # input, weight and grad_output can have each their own MX element dtype.

        @staticmethod
        def forward(
            ctx,
            input_hp: torch.Tensor,
            weight_hp: torch.Tensor,
            norm_weight_hp: torch.Tensor,
            in_elem_dtype: Any,
            w_elem_dtype: Any,
            grad_elem_dtype: Any,
            block_size: int,
            gemm_kernel_choice: MXGemmKernelChoice,
            scale_rounding: ScaleCalculationMode,
            norm_mode: NormMode,
            norm_kwargs: Dict[str, Any],
        ):
            ctx.in_elem_dtype = in_elem_dtype
            ctx.w_elem_dtype = w_elem_dtype
            ctx.grad_elem_dtype = grad_elem_dtype
            ctx.block_size = block_size
            ctx.gemm_kernel_choice = gemm_kernel_choice
            ctx.scale_rounding = scale_rounding
            ctx.norm_mode = norm_mode
            ctx.norm_kwargs = norm_kwargs

            # OCP MX scale calculation (default)
            # scaling_mode = ScaleCalculationMode.FLOOR,
            # Improved rounding.
            act_scaling_mode = scale_rounding
            w_scaling_mode = scale_rounding
            # act_scaling_mode = ScaleCalculationMode.EVEN
            # w_scaling_mode = ScaleCalculationMode.FLOOR

            # input @ weight_t = output
            input_orig_shape = input_hp.shape
            input_hp_r = input_hp.reshape(-1, input_orig_shape[-1])

            input_mx_r_dim0_scale, input_mx_r_dim0_data, rms_estimate = to_mx_with_norm(
                input_hp_r,
                in_elem_dtype,
                block_size,
                scaling_mode=act_scaling_mode,
                norm_mode=norm_mode,
                norm_kwargs=norm_kwargs,
            )

            ctx.save_for_backward(input_hp, weight_hp, norm_weight_hp, rms_estimate)

            input_mx_r_dim0 = MXTensor(
                input_mx_r_dim0_data,
                input_mx_r_dim0_scale,
                in_elem_dtype,
                block_size,
                input_hp_r.dtype,
                gemm_kernel_choice=gemm_kernel_choice,
                pack_fp6=False,
                act_quant_kwargs=None,
            )

            # Incorporate norm weight into linear weight
            fused_weight_hp = weight_hp * norm_weight_hp

            weight_mx_dim0 = MXTensor.to_mx(
                fused_weight_hp,
                w_elem_dtype,
                block_size,
                scaling_mode=w_scaling_mode,
                gemm_kernel_choice=gemm_kernel_choice,
            )
            output = torch.mm(input_mx_r_dim0, weight_mx_dim0.t())
            output = output.reshape(*input_orig_shape[:-1], output.shape[-1])

            return output

        @staticmethod
        def backward(ctx, grad_output_hp: torch.Tensor):
            input_hp, weight_hp, norm_weight_hp, rms_estimate = ctx.saved_tensors
            weight_hp_t_c = weight_hp.t().contiguous()
            in_elem_dtype = ctx.in_elem_dtype
            w_elem_dtype = ctx.w_elem_dtype
            grad_elem_dtype = ctx.grad_elem_dtype
            block_size = ctx.block_size
            gemm_kernel_choice = ctx.gemm_kernel_choice
            scale_rounding = ctx.scale_rounding

            grad_output_orig_shape = grad_output_hp.shape
            grad_output_hp_r = grad_output_hp.reshape(-1, grad_output_orig_shape[-1])

            input_hp_orig_shape = input_hp.shape
            input_hp_r = input_hp.reshape(-1, input_hp_orig_shape[-1])

            # OCP MX scale calculation (default)
            # scaling_mode = ScaleCalculationMode.FLOOR,
            # Improved rounding.
            act_scaling_mode = scale_rounding
            w_scaling_mode = scale_rounding
            grad_scaling_mode = scale_rounding

            # grad_output @ weight = grad_input
            grad_output_mx_dim0 = MXTensor.to_mx(
                grad_output_hp_r,
                grad_elem_dtype,
                block_size,
                scaling_mode=grad_scaling_mode,
                gemm_kernel_choice=gemm_kernel_choice,
            )  # mx, [B, out]
            weight_mx_dim1 = MXTensor.to_mx(
                weight_hp_t_c,
                w_elem_dtype,
                block_size,
                scaling_mode=w_scaling_mode,
                gemm_kernel_choice=gemm_kernel_choice,
            )  # mx, [in, out]
            grad_mm_input = torch.mm(
                grad_output_mx_dim0, weight_mx_dim1.t()
            )  # high_prec, [B, in]

            # input_t @ grad_output = grad_weight
            grad_output_mx_dim1 = MXTensor.to_mx(
                grad_output_hp_r.t().contiguous(),
                grad_elem_dtype,
                block_size,
                scaling_mode=grad_scaling_mode,
                gemm_kernel_choice=gemm_kernel_choice,
            )  # [out, B]

            inv_rms = rms_estimate.reciprocal()

            normed_input_hp_r = input_hp_r / rms_estimate
            input_t_mx_dim0_tmp = MXTensor.to_mx(
                normed_input_hp_r.t().contiguous(),
                in_elem_dtype,
                block_size,
                scaling_mode=act_scaling_mode,
                gemm_kernel_choice=gemm_kernel_choice,
            )
            input_t_mx_dim0 = input_t_mx_dim0_tmp.t()

            grad_weight = torch.mm(grad_output_mx_dim1, input_t_mx_dim0) * norm_weight_hp
            grad_norm_weight = torch.sum(normed_input_hp_r * grad_mm_input, dim=0)

            grad_out = grad_mm_input * norm_weight_hp
            delta = torch.einsum("...d,...d->...", grad_out, input_hp_r).unsqueeze(-1)
            grad_input = (-1 / input_hp_r.shape[-1]) * inv_rms.pow(
                3
            ) * input_hp_r * delta + inv_rms * grad_out
            grad_input = grad_input.reshape(input_hp_orig_shape)
            return (
                grad_input,
                grad_weight,
                grad_norm_weight,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    # =============================================== #

    torch.manual_seed(1472)
    block_size = 2**5
    batch_size = 2**2
    with torch.no_grad():
        x = generate_inputs(block_size)
        hidden_dim = x.shape[-1]
        x = x.reshape(batch_size, x.shape[0] // batch_size, hidden_dim)
    x.requires_grad_(True)
    out_dim = hidden_dim // 2
    norm_eps = 1e-6
    norm_kwargs = {"eps": norm_eps}

    in_elem_dtype = torch.float8_e4m3fn
    w_elem_dtype = torch.float8_e4m3fn
    grad_elem_dtype = torch.float8_e5m2

    norm_weight = torch.randn((hidden_dim,))
    weight = torch.randn((out_dim, hidden_dim))
    with torch.no_grad():
        norm_weight = norm_weight * 0.1 + 1
        weight = weight * hidden_dim**-0.5
    norm_weight.requires_grad_(True)
    weight.requires_grad_(True)

    norm_kwargs = {"eps": norm_eps}
    if norm_mode == "post":
        norm_kwargs["lut"] = torch.tensor(
            lut_and_lerp.create_lut(
                n_lut_entries=256,
                scale_dtype=torch.float8_e8m0fnu,
                data_dtype=torch.float8_e4m3fn,
                scale_rounding_fn="rceil",
                block_size=block_size,
            ),
        )
    elif norm_mode == "pre":
        norm_kwargs["scale"] = torch.tensor(
            get_max_abs_pre_round_norm_scaling(block_size, Reduction.MEAN),
        )

    out = mx_mm_with_affine_norm.apply(
        x,
        weight,
        norm_weight,
        in_elem_dtype,
        w_elem_dtype,
        grad_elem_dtype,
        block_size,
        MXGemmKernelChoice.EMULATED,
        ScaleCalculationMode.CUBLAS_CEIL,
        NormMode[norm_mode.upper()],
        norm_kwargs,
    )

    grad = torch.randn_like(out)
    out.backward(grad)

    ref_out = out.detach().clone()
    ref_x_grad = x.grad.detach().clone()
    ref_weight_grad = weight.grad.detach().clone()
    ref_norm_weight_grad = norm_weight.grad.detach().clone()

    x.grad.zero_()
    weight.grad.zero_()
    norm_weight.grad.zero_()

    normed_x, normed_x_mx = mx_norm.apply(
        x,
        in_elem_dtype,
        block_size,
        ScaleCalculationMode.CUBLAS_CEIL,
        NormMode[norm_mode.upper()],
        Reduction.MEAN,
        None,
        norm_kwargs,
        MXGemmKernelChoice.EMULATED,
    )

    out = mx_mm.apply(
        normed_x,
        normed_x_mx,
        weight,
        norm_weight,
        in_elem_dtype,
        w_elem_dtype,
        grad_elem_dtype,
        block_size,
        MXGemmKernelChoice.EMULATED,
        ScaleCalculationMode.CUBLAS_CEIL,
    )

    out.backward(grad)

    new_out = out.detach().clone()
    new_x_grad = x.grad.detach().clone()
    new_weight_grad = weight.grad.detach().clone()
    new_norm_weight_grad = norm_weight.grad.detach().clone()

    _check(ref_out, new_out, 1e-3)
    _check(ref_norm_weight_grad, new_norm_weight_grad, 1e-3)
    _check(ref_weight_grad, new_weight_grad, 1e-3)
    _check(ref_x_grad, new_x_grad, 1e-3)
