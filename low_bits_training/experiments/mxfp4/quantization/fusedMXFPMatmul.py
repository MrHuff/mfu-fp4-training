#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
from quantization.dimensionQuantisationClass import (
    Optional,
    EXMYQuantization,
    blockify,
    new_to_mx,
    E2M1Quantizer,
    E2M1QuantizerSR,
    QvalueQuantizer,
    SplineGradModule,
    MXFPscalingModule,
)
from torchao.prototype.mx_formats.mx_linear import (
    replace_with_custom_fn_if_matches_filter,
    _is_linear,
)
from quantization.MXFPconfig import MXLinearDimConfig
from quantization.mx_ops_dim import DimensionMXTensor, mx_matmul
from torch.nn import Linear
import torch
import numpy as np
from scipy.linalg import hadamard


def mx_tensor_nan_check(mx_tensor: DimensionMXTensor, tensor_name: str):
    if torch.isnan(mx_tensor._data).any().item():
        print(f"{tensor_name} quantisation tensor has nans")

    if torch.isnan(mx_tensor._scale_fp).any().item():
        print(f"{tensor_name} scale_fp tensor has nans")

    if torch.isnan(mx_tensor._scale_lp).any().item():
        print(f"{tensor_name} scale_lp tensor has nans")


def e8m0_to_float(exponent_8bit):
    if exponent_8bit == 0:
        return 0.0  # No subnormals!
    elif exponent_8bit == 255:
        return np.nan  # Invalid (could also assert/error)
    else:
        return 2.0 ** (exponent_8bit - 127)


def e4m3_to_float(exponent_4bit, mantissa_3bit):
    # E4M3 format (4-bit exponent, 3-bit mantissa)
    if exponent_4bit == 0:
        # Subnormal number (optional - depends on implementation)
        return (mantissa_3bit / 8.0) * 2.0 ** (-6)  # Minimum exponent -6
    else:
        # Normal number: 1.mantissa * 2^exponent
        return (1.0 + mantissa_3bit / 8.0) * 2.0 ** (exponent_4bit - 7)  # Bias = 7


def getScaleGrid(config):
    if config.scale_type == "E8M0":
        valid_exponents = range(1, 255)
        e8m0_values = [e8m0_to_float(e) for e in valid_exponents]
        return np.array(e8m0_values[:20]), np.linspace(0, e8m0_values[20], 40)
    elif config.scale_type == "E4M3":
        # Generate all possible E4M3 values
        e4m3_values = []
        for exp in range(16):  # 4-bit exponent (0-15)
            for mant in range(8):  # 3-bit mantissa (0-7)
                e4m3_values.append(e4m3_to_float(exp, mant))
        return np.array(e4m3_values[:20]), np.linspace(0, e4m3_values[20], 40)
    else:
        raise ValueError(f"Unsupported scale type: {config.scale_type}")


class STEscaleGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad: torch.Tensor,
        lp_tensor: DimensionMXTensor,
        q_prime: torch.nn.Module,
        X: torch.Tensor,
    ):
        return grad


class TensorScalingGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad: torch.Tensor,
        lp_tensor: DimensionMXTensor,
        X: torch.Tensor,
        grad_type: str,
    ):
        if grad_type == "absmax":
            return grad + lp_tensor._global_abs_mask.reshape(grad.shape) * (
                lp_tensor.to_dtype(X.dtype) - X / lp_tensor._g * grad
            )
        elif grad_type == "ignore":
            return grad
        else:
            return grad + (lp_tensor.to_dtype(X.dtype) - X / lp_tensor._g * grad)


class AbsMaxGradFPScale(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad: torch.Tensor,
        lp_tensor: DimensionMXTensor,
        q_prime: torch.nn.Module,
        X: torch.Tensor,
    ):
        reshaped_grad, grad_orig_shape = blockify(
            grad, lp_tensor._block_dim, lp_tensor._block_size
        )
        g_reshaped, _ = blockify(
            lp_tensor._data, lp_tensor._block_dim, lp_tensor._block_size
        )

        scaled_reshaped_grad = (
            reshaped_grad
            * lp_tensor._scale_fp
            / lp_tensor._scale_lp
            * lp_tensor._max_abs_mask
        )
        scaled_reshaped_grad_2 = (
            q_prime(lp_tensor._scale_fp, 1.0)
            / lp_tensor._scale_lp**2
            * torch.logical_not(lp_tensor._max_abs_mask)
            * g_reshaped
            * (lp_tensor._scale_fp / lp_tensor._max_abs * torch.sign(g_reshaped))
        )

        return (scaled_reshaped_grad + scaled_reshaped_grad_2).reshape(grad_orig_shape)

        # TODO, figure smooth quantisation for scale and additional term!


class SoftMaxGradFPScale(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad: torch.Tensor,
        lp_tensor: DimensionMXTensor,
        q_prime: torch.nn.Module,
        X: torch.Tensor,
    ):
        reshaped_grad, grad_orig_shape = blockify(
            grad, lp_tensor._block_dim, lp_tensor._block_size
        )
        g_reshaped, _ = blockify(
            lp_tensor._data, lp_tensor._block_dim, lp_tensor._block_size
        )
        X_reshaped, _ = blockify(X, lp_tensor._block_dim, lp_tensor._block_size)
        scaled_reshaped_grad = reshaped_grad * lp_tensor._scale_fp / lp_tensor._scale_lp
        dn = (
            -lp_tensor._scale_lp
            / lp_tensor._max_abs
            * lp_tensor._sm
            * torch.sign(X_reshaped)
        )
        scaled_reshaped_grad_2 = (
            X_reshaped * reshaped_grad / lp_tensor._scale_lp
            - g_reshaped * q_prime(lp_tensor._scale_fp, 1.0) / lp_tensor._scale_lp**2
        )
        # propagate softmax somehow, have to save the softmax annoyingly

        return (scaled_reshaped_grad + dn * scaled_reshaped_grad_2).reshape(
            grad_orig_shape
        )
        # return (scaled_reshaped_grad+scaled_reshaped_grad_2).reshape(grad_orig_shape)


class AbsMaxGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad: torch.Tensor,
        lp_tensor: DimensionMXTensor,
        q_prime: torch.nn.Module,
        X: torch.Tensor,
    ):
        reshaped_grad, grad_orig_shape = blockify(
            grad, lp_tensor._block_dim, lp_tensor._block_size
        )
        g_reshaped, _ = blockify(
            lp_tensor._data, lp_tensor._block_dim, lp_tensor._block_size
        )

        scaled_reshaped_grad_2 = (
            q_prime(lp_tensor._scale_fp, 1.0)
            / lp_tensor._scale_lp
            * torch.logical_not(lp_tensor._max_abs_mask)
            * (
                lp_tensor._scale_fp
                / lp_tensor._max_abs
                * torch.sign(g_reshaped)
                / lp_tensor._scale_lp
                * g_reshaped
                - lp_tensor._scale_fp * reshaped_grad
            )
        )

        return (reshaped_grad + scaled_reshaped_grad_2).reshape(grad_orig_shape)


class SoftMaxGrad(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad: torch.Tensor,
        lp_tensor: DimensionMXTensor,
        q_prime: torch.nn.Module,
        X: torch.Tensor,
    ):
        X_reshaped, _ = blockify(X, lp_tensor._block_dim, lp_tensor._block_size)
        g_reshaped, _ = blockify(
            lp_tensor._data, lp_tensor._block_dim, lp_tensor._block_size
        )
        reshaped_grad, grad_orig_shape = blockify(
            grad, lp_tensor._block_dim, lp_tensor._block_size
        )
        dn = (
            -lp_tensor._scale_lp
            / lp_tensor._max_abs
            * lp_tensor._sm
            * torch.sign(X_reshaped)
        )
        scaled_reshaped_grad_2 = (
            (X_reshaped * reshaped_grad - g_reshaped / lp_tensor._scale_lp)
            * q_prime(lp_tensor._scale_fp, 1.0)
            / lp_tensor._scale_lp
        )
        # propagate softmax somehow, have to save the softmax annoyingly

        return (reshaped_grad + dn * scaled_reshaped_grad_2).reshape(grad_orig_shape)
        # return (scaled_reshaped_grad+scaled_reshaped_grad_2).reshape(grad_orig_shape)


class STEquantisationGrad(torch.nn.Module):
    def __init__(selfe):
        super().__init__()

    def forward(self, data_hp_scaled, g):
        return g


class BaselineQuantisationGrad(torch.nn.Module):
    def __init__(self, quant_values, k, dtype, lb, ub):
        super().__init__()
        self.k = k
        self.dtype = dtype
        self.quant = EXMYQuantization(quant_values=quant_values, dtype=dtype)
        self.lb = lb
        self.ub = ub

    def forward(self, data_hp_scaled, g):
        max_mask = self.quant.max > data_hp_scaled
        # Find the index where x falls into the interval
        quant = self.quant.to(data_hp_scaled.device)
        idx = torch.bucketize(data_hp_scaled, quant.quant_values) - 1
        idx = torch.clamp(idx, 0, len(quant.deltas) - 1)

        # Extract corresponding values
        mid = quant.deltas[idx] / 2
        x_diff = data_hp_scaled - quant.centers[idx]
        abs_x_diff = torch.abs(x_diff)
        nonzero_mask = abs_x_diff != 0
        grad_output = torch.zeros_like(x_diff)
        grad_output[nonzero_mask] = (
            mid[nonzero_mask]
            * (1 / self.k)
            * abs_x_diff[nonzero_mask] ** (1 / self.k - 1)
        )
        return (grad_output * torch.logical_not(max_mask) + max_mask) * g


class SigmoidQuantisationGrad(torch.nn.Module):
    def __init__(self, quant_values, temperature, dtype, lb=0.3, ub=1.5):
        super().__init__()
        self.temperature = temperature
        self.dtype = dtype
        self.quant = EXMYQuantization(quant_values, dtype)
        self.lb = lb
        self.ub = ub

    def forward(self, data_hp_scaled, g):
        max_mask = self.quant.max > data_hp_scaled
        # Find the index where x falls into the interval
        quant = self.quant.to(data_hp_scaled.device)

        idx = torch.bucketize(data_hp_scaled, quant.quant_values) - 1
        idx = torch.clamp(idx, 0, len(quant.deltas) - 1)

        # Extract corresponding values
        x_diff = (data_hp_scaled - quant.centers[idx]) * quant.sigmoidDeltas[idx]

        sigmoid_values = torch.sigmoid(x_diff / self.temperature)
        grad = ((sigmoid_values * (1 - sigmoid_values)) / self.temperature * 12).clip(
            self.lb, self.ub
        )
        return (grad * torch.logical_not(max_mask) + max_mask) * g


class LinearSplineQuantisationGrad(torch.nn.Module):
    def __init__(self, quant_values, quant_func, x_test, dtype, lb, ub):
        super().__init__()
        self.dtype = dtype
        self.spline = SplineGradModule(
            quant_func=quant_func, x_test=x_test, dtype=dtype, K=1
        )
        self.quant = EXMYQuantization(quant_values, dtype)
        self.lb = lb
        self.ub = ub

    def forward(self, data_hp_scaled, g):
        max_mask = self.quant.max > data_hp_scaled
        # --- Existing setup code ---
        spline = self.spline.to(data_hp_scaled.device)
        indices = torch.searchsorted(spline.knots, data_hp_scaled, right=True) - 1
        indices = torch.clamp(indices, 0, len(spline.knots) - 2)
        offset = data_hp_scaled - spline.knots[indices]
        c_ind = spline.coeffs[:, indices]

        # --- START of the corrected logic ---

        # 1. Compute the correct coefficients for the derivative polynomial.
        # For a polynomial of degree `m`, the derivative's coefficients are derived
        # by multiplying the original coefficients (C_0, C_1, ...) by their corresponding
        # powers (m, m-1, ...).
        num_coeffs = c_ind.shape[0]
        degree = num_coeffs - 1

        if degree < 1:
            # If the spline is constant (degree 0), the derivative is zero.
            poly_derivative = torch.zeros_like(data_hp_scaled)
        else:
            # Create multipliers for the coefficients, i.e., [m, m-1, ..., 1]
            # We reshape it to work with coefficient tensors of any dimension.
            power_multipliers = torch.arange(
                degree, 0, -1, device=c_ind.device, dtype=c_ind.dtype
            )
            power_multipliers = power_multipliers.view(-1, *[1] * data_hp_scaled.dim())

            # The coefficients for the derivative polynomial.
            # We take all but the last coefficient of the original polynomial.
            deriv_coeffs = c_ind[:-1] * power_multipliers

            # 2. Evaluate the derivative polynomial using a fast, vectorized method.
            # This replaces the slow Python for-loop and works on tensors of any shape.

            # Create a matrix of the powers of the offset `h`: [h^(m-1), h^(m-2), ..., h^0]
            deriv_degree = degree - 1

            # Prepare exponent tensor, reshaping for broadcasting with the input tensor shape
            exponents = torch.arange(
                deriv_degree, -1, -1, device=c_ind.device, dtype=c_ind.dtype
            )
            exponents = exponents.view(-1, *[1] * data_hp_scaled.dim())

            # Broadcast `offset` to compute all powers at once. `offset` has shape (*),
            # `offset.unsqueeze(0)` has shape (1, *), `offset_powers` has shape (deriv_degree+1, *)
            offset_powers = offset.unsqueeze(0) ** exponents

            # Multiply coefficients by their corresponding offset powers and sum them up.
            # This is the vectorized equivalent of Horner's method.
            poly_derivative = torch.sum(deriv_coeffs * offset_powers, dim=0)

        # --- END of the corrected logic ---

        return (
            poly_derivative.clip(self.lb, self.ub) * torch.logical_not(max_mask)
            + max_mask
        ) * g


class FusedMXFPMatMul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        config: MXLinearDimConfig,
        quantgrad: torch.nn.Module,
        scalegrad: torch.nn.Module,
        tensorscalegrad: torch.nn.Module,
        q_prime: torch.nn.Module,
        scalingModule: torch.nn.Module,
        fp4_quantizer_tied: torch.nn.Module,
        fp4_quantizer_SR: torch.nn.Module,
    ):
        fp4_quantiser = (
            fp4_quantizer_tied
            if config.use_approx["SR"] in ["backwards", None, "IntelFP4", "NewIntelFP4"]
            else fp4_quantizer_SR
        )
        A_mx_lp, _, A_scaled_hp_lp = new_to_mx(
            A, scalingModule, config.gemm_kernel_choice, fp4_quantiser
        )
        B_mx_lp, _, B_scaled_hp_lp = new_to_mx(
            B, scalingModule, config.gemm_kernel_choice, fp4_quantiser
        )
        ctx.config = config
        ctx.quantgrad = quantgrad
        ctx.scalegrad = scalegrad
        ctx.tensorscalegrad = tensorscalegrad
        ctx.q_prime = q_prime
        ctx.scalingModule = scalingModule
        ctx.fp4_quantizer_tied = fp4_quantizer_tied
        ctx.fp4_quantizer_SR = fp4_quantizer_SR
        ctx.A_mx_lp = A_mx_lp
        ctx.B_mx_lp = B_mx_lp
        ctx.save_for_backward(A_scaled_hp_lp, B_scaled_hp_lp, A, B)

        return mx_matmul(A_mx_lp, B_mx_lp)

    @staticmethod
    def backward(ctx, grad_output):
        config = ctx.config
        quantgrad = ctx.quantgrad
        scalegrad = ctx.scalegrad
        tensorscalegrad = ctx.tensorscalegrad
        q_prime = ctx.q_prime
        scalingModule = ctx.scalingModule
        fp4_quantizer_tied = ctx.fp4_quantizer_tied
        fp4_quantizer_SR = ctx.fp4_quantizer_SR
        A_mx_lp = ctx.A_mx_lp
        B_mx_lp = ctx.B_mx_lp
        fp4_quantiser = (
            fp4_quantizer_SR
            if config.use_approx["SR"] in ["backwards", "all", "IntelFP4"]
            else fp4_quantizer_tied
        )
        A_scaled_hp_lp, B_scaled_hp_lp, A, B = ctx.saved_tensors
        if config.use_approx["use_hadamard"] in ["all", "backward"]:
            H = (
                torch.tensor(hadamard(config.block_size), dtype=A.dtype, device=A.device)
                / config.block_size**0.5
            )
            S = (torch.randn(config.block_size, device=A.device) > 0).to(A.dtype) * 2 - 1
            HS_back = H.T * S

            grad_output_H = (
                grad_output.reshape(-1, config.block_size) @ (HS_back).T
            ).reshape(grad_output.shape)
            grad_output_T = (
                HS_back @ grad_output.reshape(config.block_size, -1)
            ).reshape(grad_output.shape)
            B_ = (HS_back @ B.reshape(config.block_size, -1)).reshape(B.shape)
            A_ = (HS_back @ A.reshape(config.block_size, -1)).reshape(A.shape)

            B_lp_T, _, _ = new_to_mx(
                B_.T.contiguous(),
                scalingModule,
                config.gemm_kernel_choice,
                fp4_quantiser=fp4_quantiser,
            )
            A_lp, _, _ = new_to_mx(
                A_.contiguous(),
                scalingModule,
                config.gemm_kernel_choice,
                fp4_quantiser=fp4_quantiser,
            )
            grad_T_mx_lp, _, _ = new_to_mx(
                grad_output_T.T.contiguous(),
                scalingModule,
                config.gemm_kernel_choice,
                fp4_quantiser=fp4_quantiser,
            )
            grad_mx_lp, _, _ = new_to_mx(
                grad_output_H.contiguous(),
                scalingModule,
                config.gemm_kernel_choice,
                fp4_quantiser=fp4_quantiser,
            )
            gradA = mx_matmul(
                grad_mx_lp, B_lp_T
            )  # z x m  #TODO - something wrong when quantising input tensor with E4M3 scaling factor
            gradB = mx_matmul(grad_T_mx_lp, A_lp).t()  # m x y
        else:
            if config.use_approx["SR"] in ["backwards", "all"]:
                grad_T_mx_lp, _, _ = new_to_mx(
                    grad_output.t().contiguous(),
                    scalingModule,
                    config.gemm_kernel_choice,
                    fp4_quantiser=fp4_quantiser,
                )
                gradA = mx_matmul(
                    B_mx_lp, grad_T_mx_lp
                ).t()  # z x m  #TODO - something wrong when quantising input tensor with E4M3 scaling factor
                gradB = mx_matmul(grad_T_mx_lp, A_mx_lp).t()  # m x y
            elif config.use_approx["SR"] == "IntelFP4":
                A_mx_lp_SR, _, _ = new_to_mx(
                    A, scalingModule, config.gemm_kernel_choice, fp4_quantizer_SR
                )
                grad_T_mx_lp, _, _ = new_to_mx(
                    grad_output.t().contiguous(),
                    scalingModule,
                    config.gemm_kernel_choice,
                    fp4_quantiser=fp4_quantiser,
                )
                gradA = mx_matmul(
                    B_mx_lp, grad_T_mx_lp
                ).t()  # z x m  #TODO - something wrong when quantising input tensor with E4M3 scaling factor
                grad_T_mx_lp, _, _ = new_to_mx(
                    grad_output.t().contiguous(),
                    scalingModule,
                    config.gemm_kernel_choice,
                    fp4_quantiser=fp4_quantiser,
                )
                gradB = mx_matmul(grad_T_mx_lp, A_mx_lp_SR).t()  # m x y
                # basically do 4 extra quantisations!
            elif config.use_approx["SR"] == "NewIntelFP4":
                A_mx_lp_SR, _, _ = new_to_mx(
                    A, scalingModule, config.gemm_kernel_choice, fp4_quantizer_SR
                )
                grad_T_mx_lp, _, _ = new_to_mx(
                    grad_output.t().contiguous(),
                    scalingModule,
                    config.gemm_kernel_choice,
                    fp4_quantiser=fp4_quantiser,
                )
                gradA = mx_matmul(
                    B_mx_lp, grad_T_mx_lp
                ).t()  # z x m  #TODO - something wrong when quantising input tensor with E4M3 scaling factor
                gradB = mx_matmul(grad_T_mx_lp, A_mx_lp_SR).t()  # m x y

        gradA = quantgrad(A_scaled_hp_lp, gradA)
        gradB = quantgrad(B_scaled_hp_lp, gradB)

        # abs-max scaling adjustment + slicing/view adjustment

        gradA = scalegrad(gradA, A_mx_lp, q_prime, A)  # alpha adjustment...
        gradB = scalegrad(gradB, B_mx_lp, q_prime, B)

        if config.use_approx["use_tensor_scaling"]:
            gradA = tensorscalegrad(
                gradA, A_mx_lp, A, config.use_approx["tensor_scaling_grad_est"]
            )
            gradB = tensorscalegrad(
                gradB, B_mx_lp, B, config.use_approx["tensor_scaling_grad_est"]
            )
        # print(gradA.abs().max())
        # print(gradB.abs().max())
        return gradA, gradB, None, None, None, None, None, None, None, None


class MXLinearDimFused(torch.nn.Linear):
    """
    Linear layer with the compute happening in emulate MX. Currently the MX
    matmul is emulated since there is no hardware support yet. Activations,
    weights and grads are casted to MX and back to high precision for each
    matmul.

    Input, weight and grad_output can have each their own MX element dtype.
    """

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        config: Optional[MXLinearDimConfig] = MXLinearDimConfig(),
    ):
        # TODO(before land): remove this
        assert isinstance(config, MXLinearDimConfig)
        mod.__class__ = MXLinearDimFused
        mod.config = config
        mod.tensorscalegrad = TensorScalingGrad()
        mod.quantiser_FP4 = E2M1Quantizer()
        mod.quantiser_FP4_SR = E2M1QuantizerSR()
        e2m1range = np.array(
            [
                -6.0,
                -4.0,
                -3.0,
                -2.0,
                -1.5,
                -1.0,
                -0.5,
                0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
            ],
            dtype=np.float32,
        )
        e2m1range_tensor = torch.from_numpy(e2m1range)
        if config.use_approx["stepGradient"] == "STE":
            mod.quantgrad = STEquantisationGrad()
        elif config.use_approx["stepGradient"] == "baseline":
            mod.quantgrad = BaselineQuantisationGrad(
                quant_values=e2m1range_tensor,
                k=config.use_approx["k"],
                dtype=config.use_approx["dtype"],
                lb=config.use_approx["lb"],
                ub=config.use_approx["ub"],
            )
        elif config.use_approx["stepGradient"] == "spline":
            q_class = QvalueQuantizer(e2m1range)

            mod.quantgrad = LinearSplineQuantisationGrad(
                quant_values=e2m1range_tensor,
                quant_func=q_class.quantize,
                x_test=np.linspace(-6, 6, 40),
                dtype=config.use_approx["dtype"],
                lb=config.use_approx["lb"],
                ub=config.use_approx["ub"],
            )
        elif config.use_approx["stepGradient"] == "sigmoid":
            mod.quantgrad = SigmoidQuantisationGrad(
                quant_values=e2m1range_tensor,
                temperature=config.use_approx["temperature"],
                dtype=config.use_approx["dtype"],
                lb=config.use_approx["lb"],
                ub=config.use_approx["ub"],
            )

        if config.use_approx["smooth"] == "STE":
            mod.scalegrad = STEscaleGrad()
        elif config.use_approx["smooth"] == "absmax":
            if config.fp_scale_factor:
                mod.scalegrad = AbsMaxGradFPScale()
            else:
                mod.scalegrad = AbsMaxGrad()
        elif (
            config.use_approx["smooth"] == "softsoftmax"
            or config.use_approx["smooth"] == "hardsoftmax"
        ):
            if config.fp_scale_factor:
                mod.scalegrad = SoftMaxGradFPScale()
            else:
                mod.scalegrad = SoftMaxGrad()

        scaleRange, x_test = getScaleGrid(config)
        scaleRange_tensor = torch.from_numpy(scaleRange)

        if config.use_approx["qGradient"] == "STE":
            mod.q_prime = STEquantisationGrad()
        elif config.use_approx["qGradient"] == "baseline":
            mod.q_prime = BaselineQuantisationGrad(
                quant_values=scaleRange_tensor,
                k=config.use_approx["k"],
                dtype=config.use_approx["dtype"],
                lb=config.use_approx["lb"],
                ub=config.use_approx["ub"],
            )
        elif config.use_approx["qGradient"] == "spline":
            q_class = QvalueQuantizer(scaleRange)
            mod.q_prime = LinearSplineQuantisationGrad(
                quant_values=scaleRange_tensor,
                quant_func=q_class.quantize,
                x_test=x_test,
                dtype=config.use_approx["dtype"],
                lb=config.use_approx["lb"],
                ub=config.use_approx["ub"],
            )
        elif config.use_approx["qGradient"] == "sigmoid":
            mod.q_prime = SigmoidQuantisationGrad(
                quant_values=scaleRange_tensor,
                temperature=config.use_approx["temperature"],
                dtype=config.use_approx["dtype"],
                lb=config.use_approx["lb"],
                ub=config.use_approx["ub"],
            )

        mod.scalingModule = MXFPscalingModule(
            elem_dtype=config.elem_dtype,
            block_size=config.block_size,
            block_dim=config.block_dim,
            roundMode=config.roundMode,
            scale_type=config.scale_type,
            use_approx=config.use_approx,
            fp_scale_factor=config.fp_scale_factor,
            use_fp32_scaling=config.use_fp32_scaling,
        )

        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x_shape = x.shape
        in_features = self.in_features
        out_features = self.out_features
        # Reshape x to 2D for matmul: (..., F_in) -> (N_flat, F_in)
        if len(original_x_shape) == 1:  # Input is (F_in)
            # This case is less common for transformer layers but supported by nn.Linear
            x_reshaped = x.unsqueeze(0)  # (1, F_in)
            output_prefix_shape = []  # For reshaping back to (F_out)
        elif len(original_x_shape) == 2:  # Input is (N, F_in)
            x_reshaped = x  # (N, F_in)
            output_prefix_shape = [
                original_x_shape[0]
            ]  # For reshaping back to (N, F_out)
        elif len(original_x_shape) == 3:  # Input is (B, S, F_in)
            x_reshaped = x.reshape(-1, in_features)  # (B*S, F_in)
            output_prefix_shape = [
                original_x_shape[0],
                original_x_shape[1],
            ]  # For (B,S,F_out)
        else:
            # For inputs with more than 3 dimensions, torch.nn.Linear flattens all dims before in_features
            x_reshaped = x.reshape(-1, in_features)
            # Collect all leading dimensions for reshaping the output
            output_prefix_shape = list(original_x_shape[:-1])

        # Handle autocast
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_dtype(
                x.device.type if x.device.type != "mps" else "cpu"
            )
            x_for_matmul = x_reshaped.to(autocast_dtype)
            w = self.weight.to(autocast_dtype)
        else:
            x_for_matmul = x_reshaped
            w = self.weight

        config = self.config
        HS = None  # Hadamard sketch tensor

        # Apply Hadamard transform if configured
        # This part assumes block_size divides the feature dimension (in_features)
        if config.use_approx["use_hadamard"] in ["forward", "all"]:
            if in_features < config.block_size:
                # Skip Hadamard when feature size too small
                x_input_to_fusedmatmul = x_for_matmul
                w_input_to_fusedmatmul = w
            else:
                if in_features % config.block_size != 0:
                    raise ValueError(
                        f"Input feature dimension ({in_features}) must be divisible by block_size ({config.block_size}) for Hadamard transform."
                    )
                if w.shape[1] % config.block_size != 0:  # w is [F_out, F_in]
                    raise ValueError(
                        f"Weight's input feature dimension ({w.shape[1]}) must be divisible by block_size ({config.block_size}) for Hadamard transform."
                    )

                _HS_device = x_for_matmul.device
                _HS_dtype = x_for_matmul.dtype

                H_val = torch.tensor(
                    hadamard(config.block_size), dtype=_HS_dtype, device=_HS_device
                ) / (config.block_size**0.5)
                S_val = (torch.randn(config.block_size, device=_HS_device) > 0).to(
                    _HS_dtype
                ) * 2 - 1
                HS = H_val.T * S_val  # [block_size, block_size]

                # Transform input
                x_hadamard_input_view = x_for_matmul.view(-1, config.block_size)
                x_transformed = (x_hadamard_input_view @ HS.T).view(x_for_matmul.shape)

                # Transform weights
                w_hadamard_input_view = w.view(-1, config.block_size)
                w_transformed = (w_hadamard_input_view @ HS.T).view(w.shape)

                x_input_to_fusedmatmul = x_transformed
                w_input_to_fusedmatmul = w_transformed
        else:
            x_input_to_fusedmatmul = x_for_matmul
            w_input_to_fusedmatmul = w
        # Perform the fused matrix multiplication
        # FusedMXFPMatMul.apply expects A of shape (M, K) and B of shape (K, N)
        # x_input_to_fusedmatmul is (N_flat, F_in)
        # w_input_to_fusedmatmul.t() is (F_in, F_out)
        y_flat = FusedMXFPMatMul.apply(
            x_input_to_fusedmatmul.contiguous(),
            w_input_to_fusedmatmul.t().contiguous(),  # Pass (F_in, F_out)
            config,
            self.quantgrad,
            self.scalegrad,
            self.tensorscalegrad,
            self.q_prime,
            self.scalingModule,
            self.quantiser_FP4,
            self.quantiser_FP4_SR,
        )
        # y_flat will have shape (N_flat, F_out)

        # Reshape y_flat back to match original input's batch/sequence dimensions
        if not output_prefix_shape:  # Original input was 1D (F_in)
            y = y_flat.squeeze(0)  # Output (F_out)
        else:
            y = y_flat.view(*output_prefix_shape, out_features)

        if self.bias is not None:
            y = y + self.bias
        return y


def swap_linear_with_mx_linear_fused(
    model,
    *,
    config: Optional[MXLinearDimConfig] = None,
    filter_fn=None,
):
    if filter_fn is None:
        combined_filter_fn = _is_linear
    else:

        def __fn(mod, fqn):
            return _is_linear(mod, fqn) and filter_fn(mod, fqn)

        combined_filter_fn = __fn
    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXLinearDimFused.from_float(mod, config=config),
        combined_filter_fn,
    )


def swap_llama_mlp_to_mxlinear(
    model,
    config: Optional[MXLinearDimConfig] = None,
    include_lm_head: bool = False,
):
    def mlp_linear_filter_fn(mod, fqn: str) -> bool:
        mlp_keywords = [
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ]
        if include_lm_head:
            mlp_keywords.append("lm_head")
        return isinstance(mod, Linear) and any(k in fqn for k in mlp_keywords)

    replace_with_custom_fn_if_matches_filter(
        model,
        lambda mod: MXLinearDimFused.from_float(mod, config=config),
        mlp_linear_filter_fn,
    )
