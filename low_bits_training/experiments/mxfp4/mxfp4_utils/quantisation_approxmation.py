#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import numpy as np
import matplotlib.pyplot as plt
from quantization.quantization_autograd_functions import (
    SmoothQuantizeSigmoidOptimizedFunctionFP4,
    BaselineSmoothQuantFP4,
    SplineGradModule,
    QvalueQuantizer,
)
import torch
import copy

# --- Data and Function Definitions (from your script) ---
e2m1_values = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=np.float32,
)


def quantize_hard(x):
    """Hard quantization: rounds x to the nearest value in e2m1_values."""
    idx = np.abs(x[:, None] - e2m1_values).argmin(axis=1)
    return e2m1_values[idx]


if __name__ == "__main__":
    k = 5
    x = np.linspace(-6, 6, 1000)  # Increased resolution for smoother plots
    x_tensor = torch.tensor(x, requires_grad=True)

    # --- Compute Quantization Functions and Gradients ---
    # Baseline
    y_g = BaselineSmoothQuantFP4.apply(x_tensor, k)
    y_g.sum().backward()
    grad_g = copy.deepcopy(x_tensor.grad)
    x_tensor.grad.zero_()

    # Hard Quantization (for plotting)
    y_h = quantize_hard(x)

    # Sigmoid Approximation
    temp = 0.5
    y_s_sigmoid = SmoothQuantizeSigmoidOptimizedFunctionFP4.apply(x_tensor, temp)
    y_s_sigmoid.sum().backward()
    grad_sigmoid = copy.deepcopy(x_tensor.grad)
    x_tensor.grad.zero_()

    # Spline Approximation
    q_class = QvalueQuantizer(e2m1_values)
    spline = SplineGradModule(
        quant_func=q_class.quantize,
        x_test=np.linspace(-6, 6, 40),
        dtype=x_tensor.dtype,
        K=1,
    )
    y_s_spline = spline(x_tensor)
    y_s_spline.sum().backward()
    grad_spline = copy.deepcopy(x_tensor.grad)
    x_tensor.grad.zero_()

    # --- Plot 1: Differentiable Quantization Approximations ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Improvement: Add horizontal lines for E2M1 levels for context
    for level in e2m1_values:
        ax.axhline(level, color="gray", linestyle=":", linewidth=0.7, alpha=0.6)

    # Improvement: Use plt.step for the hard quantization function
    ax.step(
        x,
        y_h,
        label="Hard Quantization (E2M1)",
        where="post",
        color="black",
        linewidth=2.5,
        zorder=5,
    )

    # Plot the smooth approximations with improved styling
    ax.plot(
        x,
        y_g.detach(),
        label="Baseline Approx.",
        color="#d62728",
        linestyle="--",
        linewidth=2,
        zorder=4,
    )
    ax.plot(
        x,
        y_s_sigmoid.detach(),
        label=rf"Sigmoid Approx. (T={temp})",
        color="#2ca02c",
        linestyle="-.",
        linewidth=2,
        zorder=3,
    )
    ax.plot(
        x,
        y_s_spline.detach(),
        label="Spline Approx.",
        color="#9467bd",
        linestyle=":",
        linewidth=2.5,
        zorder=2,
    )

    ax.set_title("Differentiable Approximations of E2M1 Quantization", fontsize=16)
    ax.set_xlabel("Input Value ($x$)", fontsize=12)
    ax.set_ylabel("Quantized Value ($Q(x)$)", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x.min(), x.max())

    plt.tight_layout()
    plt.savefig(
        f"approx_quantisation_k={k}_torch_improved.png", bbox_inches="tight", dpi=300
    )
    plt.clf()

    # --- Plot 2: Gradients of the Approximations ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot gradients with matching styles from Plot 1
    ax.plot(
        x,
        grad_g.detach(),
        label=f"Baseline Gradient (k={k})",
        color="#d62728",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        x,
        grad_sigmoid.detach(),
        label=f"Sigmoid Gradient (T={temp})",
        color="#2ca02c",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        x,
        grad_spline.detach(),
        label="Spline Gradient",
        color="#9467bd",
        linestyle="-",
        linewidth=1,
    )

    # Improvement: Add a line at y=0 for reference
    ax.axhline(1, color="black", linestyle="-", linewidth=2, label="STE (Hard Quant.)")

    ax.set_title("Gradients of Differentiable Quantization Functions", fontsize=16)
    ax.set_xlabel("Input Value ($x$)", fontsize=12)
    ax.set_ylabel(r"Gradient ($\frac{dQ}{dx}$)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x.min(), x.max())
    # Improvement: Cap y-axis for better visualization if gradients spike
    ax.set_ylim(-0.5, 20)  # Adjust this range as needed based on your results

    plt.tight_layout()
    plt.savefig(
        f"approx_quantisation_k={k}_torch_grad_improved.png", bbox_inches="tight", dpi=300
    )
    plt.clf()

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot gradients with matching styles from Plot 1
    ax.plot(
        x,
        grad_g.detach(),
        label=f"Baseline Gradient (k={k})",
        color="#d62728",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        x,
        grad_sigmoid.clip(1.0, 25).detach(),
        label=f"Sigmoid Gradient (T={temp})",
        color="#2ca02c",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        x,
        grad_spline.clip(1.0, 25).detach(),
        label="Spline Gradient",
        color="#9467bd",
        linestyle="-",
        linewidth=1,
    )

    # Improvement: Add a line at y=0 for reference
    ax.axhline(1, color="black", linestyle="-", linewidth=2, label="STE (Hard Quant.)")

    ax.set_title(
        "Clipped Gradients of Differentiable Quantization Functions", fontsize=16
    )
    ax.set_xlabel("Input Value ($x$)", fontsize=12)
    ax.set_ylabel(r"Gradient ($\frac{dQ}{dx}$)", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(x.min(), x.max())
    # Improvement: Cap y-axis for better visualization if gradients spike
    ax.set_ylim(-0.5, 20)  # Adjust this range as needed based on your results

    plt.tight_layout()
    plt.savefig(
        f"approx_quantisation_k={k}_torch_grad_improved_clipped.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.clf()
