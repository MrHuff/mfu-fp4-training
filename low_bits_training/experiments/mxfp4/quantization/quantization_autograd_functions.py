#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, PPoly
import gfloat
from gfloat.formats import format_info_ocp_e2m1
from quantization.customTorchQuantiser import round_ndarray


class ReshapeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shape):
        ctx.original_shape = x.shape
        return x.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class E2M1QuantizerSR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Register e2m1 values as buffer (so they move with the model between CPU/GPU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast e2m1 values to match input dtype

        srnumbits = 16
        srbits = torch.randint_like(x, 2**srnumbits)
        rvs = round_ndarray(
            format_info_ocp_e2m1,
            x,
            gfloat.RoundMode.Stochastic,
            sat=True,
            srbits=srbits,
            srnumbits=srnumbits,
        )

        return rvs.to(x.dtype)


class E2M1Quantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Register e2m1 values as buffer (so they move with the model between CPU/GPU)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast e2m1 values to match input dtype
        rvs = round_ndarray(
            format_info_ocp_e2m1, x, gfloat.RoundMode.TiesToEven, sat=True
        )
        return rvs


class QvalueQuantizer:
    def __init__(self, q_values):
        self.q_values = q_values

    def quantize(self, x):
        idx = np.abs(x[:, None] - self.q_values).argmin(axis=1)
        return self.q_values[idx]


class EXMYQuantization(torch.nn.Module):
    def __init__(self, quant_values, dtype):
        super().__init__()
        quant_values = quant_values.to(dtype)

        centers = (quant_values[:-1] + quant_values[1:]) / 2  # 14 elements
        deltas = quant_values[1:] - quant_values[:-1]  # 14 elements
        sigmoidDeltas = 6 / (deltas / 2)  # interval length
        min = quant_values[0]
        max = quant_values[-1]
        # Register as buffers so they move with .to(device) or .cuda()
        self.register_buffer("quant_values", quant_values)
        self.register_buffer("centers", centers)
        self.register_buffer("deltas", deltas)
        self.register_buffer("sigmoidDeltas", sigmoidDeltas)
        self.register_buffer("min", min)
        self.register_buffer("max", max)

    def forward(self, x):
        # No operation in forward (this is just for storage/manipulation)
        return x


class STEClamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b):
        return torch.clamp(x, a, b)  # hard clip in forward pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None  # full passthrough, ignore clamp


class SafeDiv(torch.autograd.Function):
    FACTOR = 32.0

    @staticmethod
    def forward(ctx, numerator, denominator):
        dtype = numerator.dtype
        finfo = torch.finfo(dtype)
        tiny = finfo.tiny

        # Avoid small magnitudes for stability
        safe_denominator = torch.where(
            torch.abs(denominator) < tiny,
            SafeDiv.FACTOR * tiny * torch.sign(denominator),
            denominator,
        )

        ratio = numerator / safe_denominator
        ctx.save_for_backward(ratio, safe_denominator)
        return ratio

    @staticmethod
    def backward(ctx, grad_output):
        ratio, safe_denominator = ctx.saved_tensors
        dtype = grad_output.dtype
        finfo = torch.finfo(dtype)

        # Raw gradients
        grad_numerator = grad_output / safe_denominator
        grad_denominator = -grad_numerator * ratio

        # Clamp to max finite value to avoid FP16 overflows
        max_val = finfo.max / SafeDiv.FACTOR
        grad_numerator = torch.clamp(grad_numerator, min=-max_val, max=max_val)
        grad_denominator = torch.clamp(grad_denominator, min=-max_val, max=max_val)
        return grad_numerator, grad_denominator


class SplineFunctionGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, knots, coeffs):
        """
        Compute the spline value at x using piecewise polynomial coefficients.
        This implementation uses a fully vectorized polynomial evaluation.
        """
        # Find the interval index for each element in x
        indices = torch.searchsorted(knots, x, right=True) - 1
        indices = torch.clamp(indices, 0, len(knots) - 2)

        # Select the coefficients for the corresponding intervals
        c_ind = coeffs[:, indices]

        # Calculate the offset from the start of the interval
        offset = x - knots[indices]

        # Save necessary tensors for backward pass
        ctx.save_for_backward(offset, c_ind)

        # Vectorized polynomial evaluation (replaces Horner's method loop)
        degree = coeffs.shape[0] - 1

        # Create a matrix of the powers of the offset: [h^m, h^(m-1), ..., h^0]
        # Reshape exponents for broadcasting
        exponents = torch.arange(degree, -1, -1, device=x.device, dtype=x.dtype)
        exponents = exponents.view(-1, *[1] * x.dim())  # Shape (degree+1, 1, 1...)

        # Broadcast `offset` to compute all powers at once
        # `offset` has shape (*), `offset.unsqueeze(0)` has shape (1, *)
        # `offset_powers` will have shape (degree+1, *)
        offset_powers = offset.unsqueeze(0) ** exponents

        # Multiply coefficients by their corresponding offset powers and sum them up
        result = torch.sum(c_ind * offset_powers, dim=0)

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the gradient of the spline.
        This implementation correctly calculates the derivative of the piecewise polynomial.
        """
        offset, c_ind = ctx.saved_tensors

        # If the polynomial is P(h) = C_0*h^m + C_1*h^(m-1) + ... + C_m,
        # its derivative is P'(h) = m*C_0*h^(m-1) + (m-1)*C_1*h^(m-2) + ... + C_(m-1).

        num_coeffs = c_ind.shape[0]
        degree = num_coeffs - 1

        if degree < 1:
            # If the spline is constant (degree 0), the derivative is zero.
            poly_derivative = torch.zeros_like(grad_output)
        else:
            # Create multipliers for the coefficients, i.e., [m, m-1, ..., 1]
            # We reshape it to work with coefficient tensors of any dimension.
            power_multipliers = torch.arange(
                degree, 0, -1, device=c_ind.device, dtype=c_ind.dtype
            )
            power_multipliers = power_multipliers.view(-1, *[1] * grad_output.dim())

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
            exponents = exponents.view(-1, *[1] * grad_output.dim())

            # Broadcast `offset` to compute all powers at once. `offset` has shape (*),
            # `offset.unsqueeze(0)` has shape (1, *), `offset_powers` has shape (deriv_degree+1, *)
            offset_powers = offset.unsqueeze(0) ** exponents

            # Multiply coefficients by their corresponding offset powers and sum them up.
            # This is the vectorized equivalent of Horner's method.
            poly_derivative = torch.sum(deriv_coeffs * offset_powers, dim=0)

        grad_x = poly_derivative * grad_output

        return grad_x, None, None


# Example usage remains the same
class SplineGradModule(torch.nn.Module):
    def __init__(self, quant_func, x_test, dtype=torch.float32, K=1):
        super().__init__()
        y_hard = quant_func(x_test)

        # Use NumPy for SciPy compatibility
        # Create the B-spline object as before
        spline = InterpolatedUnivariateSpline(x_test, y_hard, k=K)

        # Convert to a Piecewise Polynomial (PPoly)
        ppoly = PPoly.from_spline(spline._eval_args)

        # --- THE FIX IS HERE ---
        # Instead of spline.get_knots(), use the breakpoints from the PPoly object.
        # ppoly.x are the correct start/end points for the intervals in ppoly.c
        knots = ppoly.x

        # Now, the knots and coeffs are correctly aligned.
        self.register_buffer("knots", torch.tensor(knots, dtype=dtype))
        self.register_buffer("coeffs", torch.tensor(ppoly.c, dtype=dtype))

    def forward(self, x):
        return SplineFunctionGrad.apply(x, self.knots, self.coeffs)


class STEAbsmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim):
        ctx.dim = dim
        ctx.save_for_backward(x)

        return x.abs().amax(dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class HardAbsMaxWithSoftGrad(torch.autograd.Function):
    """Version 2: Hard max(|x|) forward, smooth LSE-based backward."""

    @staticmethod
    def forward(ctx, x, alpha, dim):
        max_abs_x = x.abs().amax(dim=dim)
        dim = torch.tensor(dim)  # Ensure temperature is a tensor
        alpha = torch.tensor(alpha, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, alpha, dim)
        return max_abs_x

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha, dim = ctx.saved_tensors
        weights = torch.softmax(alpha * x.abs(), dim=dim.item())  # Use softmax directly
        grad_x = weights * torch.sign(x) * grad_output.unsqueeze(-1)
        return grad_x, None, None  # No gradient for alpha


class SmoothQuantizeSigmoidOptimizedFunctionFP4(torch.autograd.Function):
    e2m1_values = torch.tensor(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        dtype=torch.float32,
    )  # Indexed 0 to 14, 15 elements!
    centers = (e2m1_values[:-1] + e2m1_values[1:]) / 2
    deltas = e2m1_values[1:] - e2m1_values[:-1]
    sigmoidDeltas = 6 / (deltas / 2)  # interval length

    @staticmethod
    def forward(ctx, x, temperature):
        temperature = torch.tensor(
            temperature, dtype=torch.float32, device=x.device
        )  # Ensure temperature is a tensor

        idx = (
            torch.bucketize(x, SmoothQuantizeSigmoidOptimizedFunctionFP4.e2m1_values) - 1
        )
        idx = torch.clamp(
            idx, 0, len(SmoothQuantizeSigmoidOptimizedFunctionFP4.centers) - 1
        )

        x_diff = (
            x - SmoothQuantizeSigmoidOptimizedFunctionFP4.centers[idx]
        ) * SmoothQuantizeSigmoidOptimizedFunctionFP4.sigmoidDeltas[idx]
        sigmoid_values = torch.sigmoid(x_diff / temperature)
        quantized_values = (
            sigmoid_values
            * SmoothQuantizeSigmoidOptimizedFunctionFP4.e2m1_values[idx + 1]
            + (1 - sigmoid_values)
            * SmoothQuantizeSigmoidOptimizedFunctionFP4.e2m1_values[idx]
        )

        ctx.save_for_backward(sigmoid_values, temperature)
        return quantized_values

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_values, temperature = ctx.saved_tensors

        grad_input = (
            grad_output * (sigmoid_values * (1 - sigmoid_values)) / temperature * 12
        )

        return grad_input, None  # No gradient for temperature


class BaselineSmoothQuantFP4(torch.autograd.Function):
    e2m1_values = torch.tensor(
        [
            -6.0,
            -4.0,
            -3.0,
            -2.0,
            -1.5,
            -1.0,
            -0.5,
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
        ],
        dtype=torch.float32,
    )  # Indexed 0 to 14, 15 elements!
    centers = (e2m1_values[:-1] + e2m1_values[1:]) / 2
    deltas = e2m1_values[1:] - e2m1_values[:-1]

    @staticmethod
    def forward(ctx, x, k):
        # Store context variables for backward pass

        k_ = torch.tensor(
            k, dtype=torch.float32, device=x.device
        )  # Ensure temperature is a tensor

        # Find the index where x falls into the interval
        idx = torch.bucketize(x, BaselineSmoothQuantFP4.e2m1_values) - 1
        idx = torch.clamp(idx, 0, len(BaselineSmoothQuantFP4.deltas) - 1)

        # Extract corresponding values
        mid = BaselineSmoothQuantFP4.deltas[idx] / 2
        base = BaselineSmoothQuantFP4.e2m1_values[idx]
        x_diff = x - BaselineSmoothQuantFP4.centers[idx]
        abs = torch.abs(x_diff)
        pol = abs ** (1 / k_)
        # Compute the output y
        ctx.save_for_backward(mid, abs, pol, k_)

        # Return the result
        return mid * (1 + torch.sign(x_diff) * pol) + base

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve stored context variables
        mid, abs, pol, k = ctx.saved_tensors
        # Compute the gradient of y with respect to x
        grad_x = grad_output * (mid * 1 / k * pol / abs)
        # Return the gradient for x and other input tensors (which don't require gradients)
        return grad_x, None, None, None, None


class SmoothQuantizer:
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, x):
        return SmoothQuantizeSigmoidOptimizedFunctionFP4.apply(x, self.temperature)


if __name__ == "__main__":
    # Example usage:
    quantizer = SmoothQuantizer(temperature=1.0)
    x = torch.tensor([0.3, -2.2, 1.7, 3.5], dtype=torch.float32, requires_grad=True)
    quantized_x = quantizer(x)
    quantized_x.backward(torch.ones_like(x))
    print(x)
