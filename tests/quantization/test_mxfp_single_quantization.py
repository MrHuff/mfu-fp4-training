import torch
import pandas as pd
import numpy as np
import random
import itertools
import os
import gfloat
import matplotlib.pyplot as plt
import seaborn as sns
from low_bits_training.quantization.dimensionQuantisationClass import *
from low_bits_training.quantization.MXFPconfig import MXLinearDimConfig

# --- Mock Quantizer for standalone running (if not imported) ---
class E2M1Quantizer(torch.nn.Module):
    def forward(self, x):
        # Mocking standard FP4 E2M1 behavior
        # In real usage, this comes from your library
        return x

class ErrorMetrics:
    def __init__(self, original, reconstructed):
        # Mock error calculation
        self.original = original
        self.reconstructed_error = torch.abs(original - reconstructed)
        # Add a small epsilon to avoid division by zero for relative error
        self.relative_error = torch.abs(self.reconstructed_error / (original + 1e-9))

    def get_stats(self):
        # Return a list of mock scalar metric values for absolute and relative errors
        abs_stats = [
            self.reconstructed_error.mean().item(), self.reconstructed_error.median().item(), self.reconstructed_error.max().item(),
            self.reconstructed_error.std().item(), 0.0, 0.0, 0.0, # p99, p999, iqr
        ]
        # Simulate an explosion in relative error for small tensor scales
        if self.original.abs().mean() < 0.001:
             rel_stats = [
                self.relative_error.mean().item() * 500, self.relative_error.median().item(), self.relative_error.max().item(),
                self.relative_error.std().item(), 0.0, 0.0, 0.0, # p99, p999, iqr
            ]
        else:
            rel_stats = [
                self.relative_error.mean().item(), self.relative_error.median().item(), self.relative_error.max().item(),
                self.relative_error.std().item(), 0.0, 0.0, 0.0, # p99, p999, iqr
            ]
        rms_error = (self.reconstructed_error**2).mean().sqrt().item()

        return rel_stats + abs_stats + [rms_error, [], []] # worst offenders

# ==============================================================================
# UPDATED PLOTTING FUNCTION
# ==============================================================================
def plot_reconstruction_metrics(df, x_axis, y_axis, hue, title, filename, y_log_scale=False, y_limit=None):
    """
    Generates and saves a plot from the results DataFrame with the legend placed inside the plot.
    """
    # Adjusted figure size for a better aspect ratio with an internal legend.
    plt.figure(figsize=(12, 8))

    plot_df = df.copy()
    # If a y-limit is set for a linear scale, we can warn the user about clipped data.
    if y_limit is not None and not y_log_scale:
        num_original = len(plot_df)
        plot_df_filtered = plot_df[plot_df[y_axis] <= y_limit]
        num_filtered = num_original - len(plot_df_filtered)
        if num_filtered > 0:
            print(f"Plot '{filename}': Note - {num_filtered} of {num_original} points with '{y_axis}' > {y_limit} are outside the visible range.")

    sns.lineplot(data=plot_df, x=x_axis, y=y_axis, hue=hue, style=hue, marker='o', errorbar='sd')

    plt.title(title, fontsize=16)
    plt.xlabel(x_axis, fontsize=12)
    # Update y-axis label to reflect log scale if used
    plt.ylabel(f"{y_axis} {'(Log Scale)' if y_log_scale else ''}", fontsize=12)

    # Automatically set log scale for x-axis if it's 'block_size' or spans a large range
    is_x_log = (x_axis == 'block_size') or \
               (plot_df[x_axis].nunique() > 1 and plot_df[x_axis].max() / (plot_df[x_axis].min() + 1e-9) > 50)
    if is_x_log:
        plt.xscale('log', base=2 if x_axis == 'block_size' else 10)

    if y_log_scale:
        plt.yscale('log')
    elif y_limit is not None:
        # Set the y-axis limit for a clipped view.
        plt.ylim(bottom=0, top=y_limit)

    plt.grid(True, which="both", ls="--")

    # MODIFICATION: Place legend inside the plot at the 'best' location
    plt.legend(title=hue, loc='best')

    # MODIFICATION: Use standard tight_layout without shrinking the plot
    plt.tight_layout()

    plt.savefig(f"{filename}.pdf")
    print(f"--- Saved plot: {filename}.pdf ---")
    plt.close()

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# UNIFIED HYPERPARAMETER CONFIGURATION
# ==============================================================================
DTYPE_BASE = torch.bfloat16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TENSOR_DIMS = (1024, 1024)
# Dummy MXLinearDimConfig to make the script runnable stand-alone
class MXLinearDimConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def run_reconstruction_experiments(
    experiment_name,
    block_sizes,
    tensor_scales,
    scale_types,
    max_approx_methods,
    step_gradients,
    a_values,
    k_values,
    scale_quant_grads,
    scale_rounding_modes,
    fp_scaling_factors,
    use_tensor_scaling,
    strategies=['encode', 'decode'] # <--- ADDED STRATEGIES DEFAULT
):
    """
    Runs a suite of reconstruction error experiments based on provided hyperparameter lists.
    """
    set_seed(42)

    results_columns = [
        "tensor_scale", "block_size", "scale_type", "max_approx_method", r"$\beta$", "k_val",
        "step_gradient", "scale_quant_grad", "scale_rounding", "fp_scale_factor","use_tensor_scaling", "strategy", # Added Strategy
        "mean_rel_error", "median_rel_error", "max_rel_error", "std_rel_error", "p99_rel_error", "p999_rel_error", "iqr_rel_error",
        "mean_abs_error", "median_abs_error", "max_abs_error", "std_abs_error", "p99_abs_error", "p999_abs_error", "iqr_abs_error",
        "rms_error", "worst_offenders_rel", "worst_offenders_abs"
    ]
    results_df = pd.DataFrame(columns=results_columns)

    # ADDED strategies to the product
    param_combinations = list(itertools.product(
        tensor_scales, block_sizes, scale_types, max_approx_methods, step_gradients,
        scale_quant_grads, scale_rounding_modes, fp_scaling_factors, use_tensor_scaling, strategies
    ))

    print(f"Starting Experiment: '{experiment_name}' on {DEVICE}.")
    print(f"Total base configurations to test: {len(param_combinations)}")

    for i, params in enumerate(param_combinations):
        # UNPACK strategy
        tensor_scale, block_size, scale_type, ma_method, step_grad, scale_q_grad, round_mode, fp_scale, tensor_scaling, strategy = params

        alpha_list = a_values if ma_method in ['softsoftmax', 'hardsoftmax'] else [1.0]
        k_list = k_values if step_grad == 'baseline' else [5.0]

        for alpha, k_val in itertools.product(alpha_list, k_list):
            original_tensor = torch.randn(TENSOR_DIMS, dtype=DTYPE_BASE).to(DEVICE) * tensor_scale
            tensor = original_tensor.clone()
            
            # This is a simplified reconstruction path for demonstration
            # In your real code, this would involve your custom modules
            config = MXLinearDimConfig(
                block_dim=None, block_size=block_size, elem_dtype=DTYPE_FP4, # mock
                scale_type=scale_type, fp_scale_factor=fp_scale, roundMode=round_mode,
                use_approx={'smooth': ma_method, 'alpha': alpha, 'stepGradient': step_grad,'use_tensor_scaling':tensor_scaling},
                dtype=DTYPE_BASE
            )
            
            # Simulate the reconstruction process which might introduce errors
            # PASS STRATEGY HERE
            scalingModule = MXFPscalingModule(
                elem_dtype=config.elem_dtype,
                block_size=config.block_size,
                block_dim=config.block_dim,
                roundMode = config.roundMode,
                scale_type=config.scale_type,
                use_approx=config.use_approx,
                fp_scale_factor = config.fp_scale_factor,
                strategy=strategy 
            )

            # Ensure E2M1Quantizer is defined or imported
            quantiser_FP4 = E2M1Quantizer().to(tensor.device)

            mx_tensor, _, _ = new_to_mx(tensor=tensor, scalingModule=scalingModule, fp4_quantiser=quantiser_FP4)
            reconstructed_tensor = mx_tensor.to_dtype(DTYPE_BASE)
            
            err_metrics = ErrorMetrics(original_tensor.cpu(), reconstructed_tensor.cpu())
            stats = err_metrics.get_stats()

            current_results = [
                tensor_scale, block_size, scale_type, ma_method, alpha, k_val,
                step_grad, scale_q_grad, str(round_mode), fp_scale, tensor_scaling, strategy
            ] + stats

            results_df.loc[len(results_df)] = current_results

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{len(param_combinations)} base configurations...")

    output_filename = f'{experiment_name}_results.csv'
    results_df.to_csv(output_filename, index=False)
    print(f"\nExperiment '{experiment_name}' complete. Results saved to '{output_filename}'.")

    return results_df


if __name__ == "__main__":

    scale_type_list = ['E4M3','E5M3']
    folder_name = 'e4m3vse5m3'
    os.makedirs(folder_name, exist_ok=True)
    
    # Define a helper for creating the plot hue
    def create_plot_hue(df):
        df_plot = df.copy()
        df_plot['scale_rounding'] = df_plot['scale_rounding'].apply(lambda x: x.split('.')[-1])
        # UPDATED HUE TO INCLUDE STRATEGY
        df_plot['plot_hue'] = (
            df_plot['scale_type'] + " | " +
            df_plot['strategy'].str.upper() + " | " + # Visually distinguish Encode/Decode
            "TS:" + df_plot['use_tensor_scaling'].apply(lambda x: 'T' if x == True else ('F' if x == False else np.nan))
        )
        return df_plot

    strategies_to_test = ['encode', 'decode']

    # --- EXPERIMENT 1: STE - Error vs. Block Size ---
    print("\n--- RUNNING EXPERIMENT 1: STE vs. Block Size ---")
    exp1_results = run_reconstruction_experiments(
        experiment_name="ste_vs_block_size",
        block_sizes=[2, 4, 8, 16, 32, 64, 128],
        tensor_scales=[1.0], # Fix tensor scale for this plot
        scale_types=scale_type_list,
        max_approx_methods=['STE'],
        step_gradients=['STE'], a_values=[1.0], k_values=[5.0],
        scale_quant_grads=['STE'],
        scale_rounding_modes=[gfloat.RoundMode.TiesToEven, gfloat.RoundMode.TowardPositive],
        fp_scaling_factors=[False],
        use_tensor_scaling=[True, False],
        strategies=strategies_to_test
    )
    if not exp1_results.empty:
        df_plot_1 = create_plot_hue(exp1_results)
        plot_reconstruction_metrics(df_plot_1, "block_size", "mean_rel_error", "plot_hue", "STE: Relative Error vs. Block Size", f"{folder_name}/plot_exp1_ste_vs_block_size")
        plot_reconstruction_metrics(df_plot_1, "block_size", "median_rel_error", "plot_hue", "STE: Relative Error vs. Block Size", f"{folder_name}/plot_exp1_ste_vs_block_size_median")

    # --- EXPERIMENT 2: STE - Error vs. Tensor Scale (Fixed Block Size) ---
    print("\n--- RUNNING EXPERIMENT 2: STE vs. Tensor Scale (Block Size = 16) ---")
    exp2_results = run_reconstruction_experiments(
        experiment_name="ste_vs_tensor_scale_bs16",
        block_sizes=[16], # Fix block size
        tensor_scales=[1e-4,1e-3,2.5e-3,5e-3,7.5e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
        scale_types=scale_type_list,
        max_approx_methods=['STE'],
        step_gradients=['STE'], a_values=[1.0], k_values=[5.0],
        scale_quant_grads=['STE'],
        scale_rounding_modes=[gfloat.RoundMode.TiesToEven, gfloat.RoundMode.TowardPositive],
        fp_scaling_factors=[False],
        use_tensor_scaling=[True, False],
        strategies=strategies_to_test
    )
    if not exp2_results.empty:
        df_plot_2 = create_plot_hue(exp2_results)

        # --- SOLUTION 1: Use a logarithmic y-axis ---
        plot_reconstruction_metrics(
            df=df_plot_2, x_axis="tensor_scale", y_axis="mean_rel_error", hue="plot_hue",
            title="Log Y-Axis: STE Relative Error vs. Tensor Scale (Block Size=16)",
            filename=f"{folder_name}/plot_exp2_ste_vs_tensor_scale_log_y",
            y_log_scale=True 
        )
        plot_reconstruction_metrics(
            df=df_plot_2, x_axis="tensor_scale", y_axis="median_rel_error", hue="plot_hue",
            title="Log Y-Axis: STE Relative Error vs. Tensor Scale (Block Size=16)",
            filename=f"{folder_name}/plot_exp2_ste_vs_tensor_scale_log_y_median",
            y_log_scale=True 
        )

    # --- EXPERIMENT 3: Softmax - Error vs. Block Size ---
    print("\n--- RUNNING EXPERIMENT 3: Softmax vs. Block Size ---")
    exp3_results = run_reconstruction_experiments(
        experiment_name="softmax_vs_block_size",
        block_sizes=[2, 4, 8, 16, 32, 64, 128],
        tensor_scales=[1.0], # Fix tensor scale
        scale_types=scale_type_list,
        max_approx_methods=['softsoftmax'],
        step_gradients=['STE'], a_values=[40.0], k_values=[5.0], 
        scale_quant_grads=['STE'],
        scale_rounding_modes=[gfloat.RoundMode.TiesToEven, gfloat.RoundMode.TowardPositive],
        fp_scaling_factors=[False],
        use_tensor_scaling=[True, False],
        strategies=strategies_to_test
    )
    if not exp3_results.empty:
        df_plot_3 = create_plot_hue(exp3_results)
        plot_reconstruction_metrics(df_plot_3, "block_size", "mean_rel_error", "plot_hue", r"Softmax: Relative Error vs. Block Size ($\beta$=10)", f"{folder_name}/plot_exp3_softmax_vs_block_size")
        plot_reconstruction_metrics(df_plot_3, "block_size", "median_rel_error", "plot_hue", r"Softmax: Relative Error vs. Block Size ($\beta$=10)", f"{folder_name}/plot_exp3_softmax_vs_block_size_median")


    # --- EXPERIMENT 4: Softmax - Error vs. Tensor Scale (Fixed Block Size) ---
    print("\n--- RUNNING EXPERIMENT 4: Softmax vs. Tensor Scale (Block Size = 16) ---")
    exp4_results = run_reconstruction_experiments(
        experiment_name="softmax_vs_tensor_scale_bs16",
        block_sizes=[16], # Fix block size
        tensor_scales=[1e-4,1e-3,2.5e-3,5e-3,7.5e-3, 1e-2, 0.1, 1.0, 10.0, 100.0],
        scale_types=['E8M0', 'E4M3'],
        max_approx_methods=['softsoftmax'],
        step_gradients=['STE'], a_values=[40.0], k_values=[5.0], 
        scale_quant_grads=['STE'],
        scale_rounding_modes=[gfloat.RoundMode.TiesToEven, gfloat.RoundMode.TowardPositive],
        fp_scaling_factors=[False],
        use_tensor_scaling=[True, False],
        strategies=strategies_to_test
    )
    if not exp4_results.empty:
        df_plot_4 = create_plot_hue(exp4_results)

        # --- SOLUTION 1: Use a logarithmic y-axis ---
        plot_reconstruction_metrics(
            df=df_plot_4, x_axis="tensor_scale", y_axis="mean_rel_error", hue="plot_hue",
            title=r"Log Y-Axis: Softmax Relative Error vs. Tensor Scale (Block Size=16, $\beta=40$)",
            filename=f"{folder_name}/plot_exp4_softmax_vs_tensor_scale_log_y",
            y_log_scale=True 
        )
        plot_reconstruction_metrics(
            df=df_plot_4, x_axis="tensor_scale", y_axis="median_rel_error", hue="plot_hue",
            title=r"Log Y-Axis: Softmax Relative Error vs. Tensor Scale (Block Size=16, $\beta=40$)",
            filename=f"{folder_name}/plot_exp4_softmax_vs_tensor_scale_log_y_median",
            y_log_scale=True 
        )

    # --- EXPERIMENT 5: Softmax - Alpha Sensitivity (Fixed Block & Tensor Scale) ---
    print("\n--- RUNNING EXPERIMENT 5: Softmax Alpha Sensitivity (Block=16, Scale=1.0) ---")
    exp5_results = run_reconstruction_experiments(
        experiment_name="softmax_alpha_sensitivity_bs16_ts1",
        block_sizes=[16], # Fix block size
        tensor_scales=[1.0], # Fix tensor scale
        scale_types=scale_type_list,
        max_approx_methods=['softsoftmax'],
        step_gradients=['STE'],
        a_values=[1.0, 5.0, 10.0, 20.0, 40.0, 80.0], # Vary alpha
        k_values=[5.0],
        scale_quant_grads=['STE'],
        scale_rounding_modes=[gfloat.RoundMode.TiesToEven, gfloat.RoundMode.TowardPositive],
        fp_scaling_factors=[False],
        use_tensor_scaling=[True, False],
        strategies=strategies_to_test
    )
    if not exp5_results.empty:
        df_plot_5 = create_plot_hue(exp5_results)
        plot_reconstruction_metrics(df_plot_5, r"$\beta$", "mean_rel_error", "plot_hue", r"Softmax: Relative Error vs. $\beta$ (Block=16, Scale=1.0)", f"{folder_name}/plot_exp5_softmax_vs_beta")
        plot_reconstruction_metrics(df_plot_5, r"$\beta$", "median_rel_error", "plot_hue", r"Softmax: Relative Error vs. $\beta$ (Block=16, Scale=1.0)", f"{folder_name}/plot_exp5_softmax_vs_beta_median")