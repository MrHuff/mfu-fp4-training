#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, PPoly
import torch

# Define quantization levels
e2m1_values = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
)


# Hard quantization function
def quantize_hard(x):
    idx = np.abs(x[:, None] - e2m1_values).argmin(axis=1)
    return e2m1_values[idx]


# Generate test values
x_test = np.linspace(-6, 6, 50)
y_hard = quantize_hard(x_test)
K = 1
# Fit a cubic spline
spline = InterpolatedUnivariateSpline(x_test, y_hard, k=K)

# Apply smooth function
y_smooth = spline(x_test)

# Plot the functions
plt.figure(figsize=(8, 5))
plt.plot(
    x_test,
    y_smooth,
    label="Smooth Quantization (Cubic Spline)",
    linestyle="--",
    color="b",
)
plt.scatter(e2m1_values, e2m1_values, color="r", label="Hard Quantization Points")
plt.xlabel("Input x")
plt.ylabel("Quantized Output")
plt.legend()
plt.title("Smooth vs Hard Quantization using Cubic Spline Interpolation")
plt.grid()
plt.savefig("interpolate.png")

plt.clf()

spline_deriv = spline.derivative()

# Apply the smooth function and its derivative
y_derivative = spline_deriv(x_test)

plt.figure(figsize=(8, 5))
plt.plot(
    x_test,
    y_derivative,
    label="Smooth Quantization (Cubic Spline) deriv",
    linestyle="--",
    color="b",
)
plt.xlabel("Input x")
plt.ylabel("Splint derivative")
plt.legend()
plt.title("Cubic spline derivative")
plt.grid()
plt.savefig("interpolate_derivative.png")
plt.clf()
knots = spline.get_knots()
coeffs = spline.get_coeffs()

t, c, k = spline._eval_args  # t = knots, c = coefficients, k = spline order
# Convert to piecewise polynomial form
ppoly = PPoly.from_spline((t, c, k))
for i in range(len(knots) - 1):
    coefs = ppoly.c[:, i]  # Coefficients in descending powers


x_test_torch = torch.from_numpy(np.linspace(-6, 6, 200)).float()
x_test_torch.requires_grad = True

y_torch = spline(x_test_torch)

lloss = y_torch.sum().backward()
grad_s = x_test_torch.grad


plt.figure(figsize=(8, 5))
plt.plot(
    x_test_torch.detach(),
    y_torch.detach(),
    label="Smooth Quantization (Cubic Spline)",
    linestyle="--",
    color="b",
)
plt.scatter(e2m1_values, e2m1_values, color="r", label="Hard Quantization Points")
plt.xlabel("Input x")
plt.ylabel("Quantized Output")
plt.legend()
plt.title("Smooth vs Hard Quantization using Cubic Spline Interpolation")
plt.grid()
plt.savefig("interpolate_torch.png")

plt.clf()

spline_deriv = spline.derivative()

# Apply the smooth function and its derivative
y_derivative = spline_deriv(x_test)

plt.figure(figsize=(8, 5))
plt.plot(
    x_test_torch.detach(),
    grad_s,
    label="Smooth Quantization (Cubic Spline) deriv",
    linestyle="--",
    color="b",
)
plt.xlabel("Input x")
plt.ylabel("Splint derivative")
plt.legend()
plt.title("Cubic spline derivative")
plt.grid()
plt.savefig("interpolate_derivative_torch.png")
