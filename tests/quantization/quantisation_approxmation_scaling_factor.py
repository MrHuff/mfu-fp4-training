import numpy as np
import matplotlib.pyplot as plt
from low_bits_training.quantization.quantization_autograd_functions import SmoothQuantizeSigmoidOptimizedFunctionFP4, BaselineSmoothQuantFP4, SplineGradModule
from low_bits_training.quantization.ExMy import *

import torch
import copy
import gfloat

#TODO: Redo reconstruction and matmul errors!
#TODO: analyse gradient errors for matmul!!!!

def dummyE8(x):
    exponent = torch.where(
        torch.isnan(x),
        0xFF,  # Handle biased exponent for nan
        # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
        (
            torch.clamp(
                torch.ceil(torch.log2(x)),
                min=-127,
                max=127,
            )
            + 127
        ).to(torch.uint8),
    )

    descale_fp = torch.where(
        exponent == 0, 1.0, torch.exp2(exponent.to(torch.float32)-127)
    )
    return descale_fp


if __name__ == "__main__":
    quantizer43 = ExMy_new(4, 3, roundMode=gfloat.RoundMode.TiesToEven)
    quantizer80 = ExMy_new(8, 0, roundMode=gfloat.RoundMode.TiesToEven)

    x = np.linspace(0, 0.1, 1000)
    x_tensor = torch.tensor(x, requires_grad=True)

    y43 = quantizer43.dummy_round(x_tensor)
    y80 = quantizer80.dummy_round(x_tensor)
    y80_dummy = dummyE8(x_tensor)

    # --- Plot 1: Quantisation Function ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, y80.detach(), label=r"E8M0 quantisation")
    plt.plot(x, y43.detach(), label=r"E4M3 quantisation")
    # plt.plot(x, y80_dummy.detach(), label=r"E8M0 nvidia")

    plt.legend()
    plt.xlabel('s')
    plt.ylabel('q(s)')
    plt.title('Quantisation Function Comparison') # Added a title for clarity
    plt.grid(True)

    # --- Improvements for Plot 1 ---
    # 1. Removes excess whitespace around the plot before saving
    plt.tight_layout()
    # 2. Saves the figure, trimming any remaining whitespace around the plot borders
    plt.savefig('EXMY_quantisaiton_func.pdf', bbox_inches='tight', dpi=300)
    plt.clf()


    # --- Plot 2: Divergence Function ---
    plt.figure(figsize=(10, 6))
    plt.plot(x, x_tensor.detach()/y80.detach(), label=r"E8M0 divergence")
    plt.plot(x, x_tensor.detach()/y43.detach(), label=r"E4M3 divergence")

    # --- Improvements for Plot 2 ---
    # 1. Adds a thick, red, dashed horizontal line at y=1 for reference
    plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Ideal (s/s_q = 1)')

    plt.legend()
    plt.xlabel('s')
    plt.ylabel(r'$s/s_q$') # Using LaTeX for the label
    plt.title('Quantisation Divergence') # Added a title for clarity
    plt.grid(True)

    # 2. Removes excess whitespace around the plot
    plt.tight_layout()
    # 3. Saves the figure, trimming any remaining whitespace
    plt.savefig('EXMY_divergence_func.pdf', bbox_inches='tight', dpi=300)
    plt.clf()




# --- Function Definitions (defined once at the top) ---

def e8m0_to_float(exponent_8bit):
    """Converts an E8M0 exponent to its float representation."""
    if exponent_8bit == 0:
        return 0.0
    # Assuming standard E8M0 with no NaNs/Infs for this range
    return 2.0 ** (exponent_8bit - 127)

def e4m3_to_float(exponent_4bit, mantissa_3bit):
    """Converts an E4M3 exponent and mantissa to its float representation."""
    # Subnormal numbers
    if exponent_4bit == 0:
        return (mantissa_3bit / 8.0) * 2.0 ** (-6)
    # Normal numbers
    else:
        return (1.0 + mantissa_3bit / 8.0) * 2.0 ** (exponent_4bit - 7)

def quantize_to_nearest(x_continuous, levels):
    """For each value in x, finds the nearest value in the provided levels."""
    # Expand dims to allow broadcasting for efficient distance calculation
    x_expanded = x_continuous[:, np.newaxis]
    # Find the index of the level with the minimum absolute difference
    indices = np.abs(x_expanded - levels).argmin(axis=1)
    return levels[indices]


# --- Plot 1: E8M0 Quantization ---

# Data Generation
valid_exponents_e8m0 = range(1, 255)
e8m0_values = [e8m0_to_float(e) for e in valid_exponents_e8m0]
# Using the first 20 positive values for a clear plot
e8m0_quantization_levels = np.array(e8m0_values[:20])

# Generate a high-resolution continuous input for the x-axis
x_continuous_e8m0 = np.linspace(0, e8m0_quantization_levels.max() * 1.05, 2000)
# Apply the quantization function to get the y-values
y_quantized_e8m0 = quantize_to_nearest(x_continuous_e8m0, e8m0_quantization_levels)

# Plotting
plt.figure(figsize=(12, 8))

# --- Improvement: Use plt.step for accurate visualization of a step function ---
plt.step(x_continuous_e8m0, y_quantized_e8m0, where='post', label='Quantization Function Q(x)')

plt.plot(x_continuous_e8m0, x_continuous_e8m0, 'r--', label='Identity Function (No Quantization)', alpha=0.7)
plt.scatter(e8m0_quantization_levels, e8m0_quantization_levels, color='green', zorder=5, s=40, label='E8M0 Quantization Levels')

plt.title('E8M0 Quantization Function for Scaling Factors', fontsize=16)
plt.xlabel('Input Scaling Factor (Continuous)', fontsize=12)
plt.ylabel('Output Scaling Factor (Quantized)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Improvement: Tighten layout and save with high DPI ---
plt.tight_layout()
plt.savefig('e8m0_scaling_improved.pdf', bbox_inches='tight', dpi=300)
plt.clf() # Clear the figure for the next plot


# --- Plot 2: E4M3 Quantization ---

# Data Generation
e4m3_values = []
for exp in range(16):
    for mant in range(8):
        e4m3_values.append(e4m3_to_float(exp, mant))
# Filter unique values and sort them
e4m3_values = sorted(list(set(e4m3_values)))
# Using the first 20 positive values for a clear plot
e4m3_quantization_levels = np.array(e4m3_values[:20])

# Generate a high-resolution continuous input for the x-axis
x_continuous_e4m3 = np.linspace(0, e4m3_quantization_levels.max() * 1.05, 2000)
# Apply the quantization function
y_quantized_e4m3 = quantize_to_nearest(x_continuous_e4m3, e4m3_quantization_levels)

# Plotting
plt.figure(figsize=(12, 8))

# --- Improvement: Use plt.step for accurate visualization ---
plt.step(x_continuous_e4m3, y_quantized_e4m3, where='post', label='Quantization Function Q(x)')

plt.plot(x_continuous_e4m3, x_continuous_e4m3, 'r--', label='Identity Function (No Quantization)', alpha=0.7)
plt.scatter(e4m3_quantization_levels, e4m3_quantization_levels, color='green', zorder=5, s=40, label='E4M3 Quantization Levels')

plt.title('E4M3 Quantization Function for Scaling Factors', fontsize=16)
plt.xlabel('Input Scaling Factor (Continuous)', fontsize=12)
plt.ylabel('Output Scaling Factor (Quantized)', fontsize=12)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# --- Improvement: Tighten layout and save with high DPI ---
plt.tight_layout()
plt.savefig('e4m3_scaling_improved.pdf', bbox_inches='tight', dpi=300)
plt.clf()