#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import os
import pickle
from glob import glob
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import ast  # Used for safely parsing string representations of lists

# --- IMPORTANT: Add Necessary Imports Here --- #
# You need to import the building blocks for your config object.
from quantization.MXFPconfig import MXLinearDimConfig, DTYPE_FP4
import gfloat
import torch
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Intercept tensor storage loading and force CPU mapping
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def save_dataset_configs(datasets, job_name, dtype):
    os.makedirs(job_name, exist_ok=True)
    for idx, ds in enumerate(datasets):
        dict_save = {
            "meta_config": {
                "dataset": ds,
                "dtype": dtype,
                "optimiser": "optim.Adam(self.model.parameters(), lr=lr)",
                "loss_scaling": False,
            },
            "MXLinearDimConfig": None,
        }
        out_path = os.path.join(job_name, f"config_baseline_{ds}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(dict_save, f)


def get_torch_dtype(dtype_str):
    """Converts a string to a torch.dtype object."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    # Return the torch.dtype if found, otherwise return the original string
    return dtype_map.get(str(dtype_str).replace("torch.", ""), dtype_str)


def get_round_mode(round_str):
    """Converts a string to a gfloat.RoundMode object."""
    round_map = {
        "RoundMode.TiesToEven": gfloat.RoundMode.TiesToEven,
        "RoundMode.TowardPositive": gfloat.RoundMode.TowardPositive,
        "TiesToEven": gfloat.RoundMode.TiesToEven,
        "TowardPositive": gfloat.RoundMode.TowardPositive,
        # Add other modes if you used them
    }
    return round_map.get(str(round_str), round_str)


def clean_value(val):
    """Converts 'N/A' or 'None' strings back to Python's None."""
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str) and val in ["N/A", "None", "nan"]:
        return None
    return val


def reconstruct_runnable_config(row: pd.Series):
    """
    Rebuilds the original nested config dictionary from a flat row of a DataFrame.
    """
    # 1. Reconstruct the 'use_approx' nested dictionary
    use_approx = {
        "smooth": clean_value(row["smooth"]),
        "alpha": float(row["alpha"]),
        "stepGradient": clean_value(row["stepGradient"]),
        "temperature": float(row["temperature"]),
        "k": float(row["k"]),
        "dtype": get_torch_dtype(row["dtype"]),
        "lb": float(row["lb"]),
        "ub": float(row["ub"]),
        "use_hadamard": clean_value(row["use_hadamard"]),
        "qGradient": clean_value(row["qGradient"]),
        "SR": clean_value(row["SR"]),
        "use_tensor_scaling": bool(row["use_tensor_scaling"]),
        "tensor_scaling_grad_est": clean_value(row["tensor_scaling_grad_est"]),
    }

    # 2. Reconstruct the MXLinearDimConfig object
    config = MXLinearDimConfig(
        block_dim=None,
        block_size=int(row["block_size"]),
        elem_dtype=DTYPE_FP4,
        scale_type=clean_value(row["scale_type"]),
        fp_scale_factor=bool(row["fp_scale_factor"]),
        roundMode=get_round_mode(row["roundMode"]),
        use_approx=use_approx,
        dtype=get_torch_dtype(row["dtype"]),
    )

    # 3. Reconstruct the 'meta_config' dictionary
    meta_config = {
        "elem_dtype": DTYPE_FP4,
        "block_dim": clean_value(row["block_dim"]),
        "scale_type": clean_value(row["scale_type"]),
        "block_size": int(row["block_size"]),
        "loss_scaling": bool(row["loss_scaling"]),
        "smooth": clean_value(row["smooth"]),
        "alpha": float(row["alpha"]),
        "stepGradient": clean_value(row["stepGradient"]),
        "temperature": float(row["temperature"]),
        "fp_scale_factor": bool(row["fp_scale_factor"]),
        "roundMode": get_round_mode(row["roundMode"]),
        "k": float(row["k"]),
        "dtype": get_torch_dtype(row["dtype"]),
        "lb": float(row["lb"]),
        "ub": float(row["ub"]),
        "use_hadamard": clean_value(row["use_hadamard"]),
        "qGradient": clean_value(row["qGradient"]),
        "SR": clean_value(row["SR"]),
        "dataset": clean_value(row["dataset"]),
        "optimiser": clean_value(row["optimiser"]),
        "use_tensor_scaling": bool(row["use_tensor_scaling"]),
        "tensor_scaling_grad_est": clean_value(row["tensor_scaling_grad_est"]),
    }

    # 4. Assemble the final dictionary
    dict_save = {"meta_config": meta_config, "MXLinearDimConfig": config}
    return dict_save


def extract_and_save_best_config(
    data_df: pd.DataFrame, dataset_name: str, stat_dir: str, config_dir: str
):
    """
    Finds the best experiment and baseline configurations, saves their parameters
    to CSVs, and reconstructs and saves their runnable .pkl configurations.
    """
    print(f"\n--- Reconstructing configs for: {dataset_name} ---")
    os.makedirs(stat_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # --- Process Best Experiment Config ---
    print("Processing best experiment config...")
    if data_df.empty or "min_metric" not in data_df.columns:
        print(
            f"Warning: Experiment DataFrame for {dataset_name} is empty or missing 'min_metric'. Skipping."
        )
    else:
        best_row = data_df.loc[data_df["min_metric"].idxmin()]
        # Save summary CSV
        summary_csv_path = os.path.join(
            stat_dir, f"best_config_summary_{dataset_name}.csv"
        )
        best_row.to_frame().T.to_csv(summary_csv_path, index=False)
        print(f"✅ Best experiment config parameters saved to: {summary_csv_path}")
        # Reconstruct and save runnable PKL
        try:
            reconstructed_config = reconstruct_runnable_config(best_row)
            runnable_pkl_path = os.path.join(config_dir, f"config_{dataset_name}.pkl")
            with open(runnable_pkl_path, "wb") as f_out:
                pickle.dump(reconstructed_config, f_out)
            print(
                f"✅ Reconstructed runnable .pkl for best experiment saved to: {runnable_pkl_path}"
            )
        except Exception as e:
            print(f"❌ Failed to reconstruct/save runnable .pkl for best experiment: {e}")
            import traceback

            traceback.print_exc()

    # --- Process Best Baseline Config ---
    save_dataset_configs([dataset_name], config_dir, torch.bfloat16)


def filter_baseline_results(folder_path: str, dataset_value: str) -> pd.DataFrame:
    """
    Lists files in a folder, finds a CSV file matching the pattern
    'baseline_*_results.csv', reads it into a pandas DataFrame, and
    filters it based on a value in the 'dataset' column.
    """
    file_pattern = os.path.join(folder_path, "baseline_*_results.csv")
    matching_files = glob(file_pattern)

    if not matching_files:
        print(f"Error: No file matching the pattern '{file_pattern}' was found.")
        return pd.DataFrame()

    csv_file_path = matching_files[0]
    print(f"Found file: {csv_file_path}")

    try:
        df = pd.read_csv(csv_file_path, engine="python")
        column_to_filter = "dataset"
        if column_to_filter not in df.columns:
            if "datset" in df.columns:
                column_to_filter = "datset"
            else:
                print(
                    "Error: Neither 'dataset' nor 'datset' column found in the CSV file."
                )
                return pd.DataFrame()

        filtered_df = df[df[column_to_filter] == dataset_value].copy()
        if filtered_df.empty:
            print(f"Info: No rows found where '{column_to_filter}' is '{dataset_value}'.")
        return filtered_df
    except Exception as e:
        print(f"An error occurred while reading or processing the file: {e}")
        return pd.DataFrame()


def load_data(f):
    base = "analysis_outputs"
    data = pd.read_csv(f"{base}/{f}/{f}.csv", engine="python", on_bad_lines="warn")
    dataset_value = data["dataset"].unique().tolist()[0]
    baseline_row = filter_baseline_results(f"{base}/{f}", dataset_value)
    return {dataset_value: {"data": data, "baseline": baseline_row}}


def plot_training_metrics(all_data):
    """
    Generates and saves a grid of line plots comparing baseline and model training metrics.
    """
    num_datasets = len(all_data)
    if num_datasets == 0:
        print("No data to plot.")
        return

    # Determine grid size for subplots
    cols = int(np.ceil(np.sqrt(num_datasets)))
    rows = int(np.ceil(num_datasets / cols))

    _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()  # Flatten to make it easier to iterate

    for i, (dataset_name, dataset_info) in enumerate(all_data.items()):
        ax = axes[i]

        # --- Baseline Data ---
        baseline_df = dataset_info["baseline"]
        if not baseline_df.empty and "training_metrics" in baseline_df.columns:
            try:
                # Safely parse the string representation of the list
                baseline_metrics = ast.literal_eval(
                    baseline_df["training_metrics"].iloc[0]
                )
                ax.plot(baseline_metrics, label="Baseline", linestyle="--")
            except (ValueError, SyntaxError) as e:
                print(
                    f"Could not parse baseline training_metrics for {dataset_name}: {e}"
                )

        # --- Model Data ---
        data_df = dataset_info["data"]
        if (
            not data_df.empty
            and "min_metric" in data_df.columns
            and "training_metrics" in data_df.columns
        ):
            # Find the row with the smallest min_metric
            best_model_row = data_df.loc[data_df["min_metric"].idxmin()]
            try:
                # Safely parse the string representation of the list
                model_metrics = ast.literal_eval(best_model_row["training_metrics"])
                ax.plot(model_metrics, label="Best Config")
            except (ValueError, SyntaxError) as e:
                print(f"Could not parse model training_metrics for {dataset_name}: {e}")

        # --- Customize Subplot ---
        ax.set_title(dataset_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Metric")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("training_metrics_comparison.png")
    print("\nPlot saved as training_metrics_comparison.png")


def process_x_axis_names(dataset_name, src_data, metric):
    token_count = {
        "llama_9M": 9000000 * 100,
        "llama_60M": 60000000 * 100,
        "llama_350M": 350000000 * 42,
        "llama_1B": 1000000000 * 42,
    }
    dict_list = src_data.get(metric, [])

    if "llama" in dataset_name:
        total_epochs = dict_list[-1]["epoch"]
        token_per_epoch = token_count[dataset_name] / total_epochs
        x_axis = [token_per_epoch * el["epoch"] for el in dict_list]
        return x_axis, "tokens"

    return np.arange(len(dict_list)), "epoch"


def plot_validation_and_loss_curves(
    all_data, curve_sources, save_folder="individual_plots"
):
    """
    Plots validation metric and training loss curves for multiple datasets and multiple sources (e.g., best config, baselines).

    Parameters
    ----------
    all_data : dict
        {
            "dataset_name": {
                "source_name": {
                    "val_metrics": [...],
                    "training_metrics": [...]
                },
                ...
            },
            ...
        }
    curve_sources : list of dict
        Each dict has:
            {
                "key": str (key in dataset_info, e.g., "data" or "baseline"),
                "label_prefix": str (e.g., "Best Config", "Baseline A"),
                "style": str (matplotlib style, e.g., '-', '--', ':')
            }
    save_folder : str
        Folder where individual plots will be saved.
    """
    val_metric_key = "val_metrics"
    train_loss_key = "training_metrics"

    if not all_data:
        print("No data to plot.")
        return

    os.makedirs(save_folder, exist_ok=True)

    def extract_curve(curve, val_key, loss_key):
        """Extracts numerical lists from tensors or dicts."""
        if not curve:
            return []
        if hasattr(curve[0], "item"):
            curve = [v.item() for v in curve]
        if isinstance(curve[0], dict):
            key = val_key if val_key in curve[0] else loss_key
            curve = [v[key] for v in curve]
        return curve

    # --- Setup grid ---
    num_datasets = len(all_data)
    cols = int(np.ceil(np.sqrt(num_datasets)))
    rows = int(np.ceil(num_datasets / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (dataset_name, dataset_info) in enumerate(all_data.items()):
        # ax = axes[i]
        # ax2 = ax.twinx()
        # lines = []

        # # --- Loop through all sources ---
        # for source in curve_sources:
        #     src_data = dataset_info.get(source["key"], {})
        #     val_curve = extract_curve(src_data.get(val_metric_key, []), 'eval_loss', 'loss')
        #     loss_curve = extract_curve(src_data.get(train_loss_key, []), 'eval_loss', 'loss')

        #     if val_curve:
        #         lines.extend(ax.plot(val_curve, 'b' + source["style"],
        #                              label=f"{source['label_prefix']} Val Metric"))
        #     if loss_curve:
        #         lines.extend(ax2.plot(loss_curve, 'g' + source["style"],
        #                               label=f"{source['label_prefix']} Train Loss"))

        # # --- Style main grid plot ---
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Validation Metric', color='b')
        # ax2.set_ylabel('Training Loss', color='g')
        # ax.tick_params(axis='y', labelcolor='b')
        # ax2.tick_params(axis='y', labelcolor='g')
        # ax.set_title(dataset_name)
        # if lines:
        #     ax.legend(lines, [l.get_label() for l in lines], loc='best')
        # ax.grid(True, linestyle=':')

        # --- Individual plot ---
        fig_single, ax_single = plt.subplots(figsize=(10, 7))
        ax2_single = ax_single.twinx()
        lines_single = []

        for source in curve_sources:
            src_data = dataset_info.get(source["key"], {})
            val_curve = extract_curve(
                src_data.get(val_metric_key, []), "eval_loss", "loss"
            )
            loss_curve = extract_curve(
                src_data.get(train_loss_key, []), "eval_loss", "loss"
            )

            if val_curve:
                x_axis, x_name = process_x_axis_names(
                    dataset_name, src_data, val_metric_key
                )
                lines_single.extend(
                    ax_single.plot(
                        x_axis,
                        val_curve,
                        "b" + source["style"],
                        label=f"{source['label_prefix']} Val Metric",
                    )
                )
            if loss_curve:
                x_axis, x_name = process_x_axis_names(
                    dataset_name, src_data, train_loss_key
                )
                lines_single.extend(
                    ax2_single.plot(
                        x_axis,
                        loss_curve,
                        "g" + source["style"],
                        label=f"{source['label_prefix']} Train Loss",
                    )
                )

        ax_single.set_xlabel(x_name)
        ax_single.set_ylabel("Validation Metric", color="b")
        ax2_single.set_ylabel("Training Loss", color="g")
        ax_single.tick_params(axis="y", labelcolor="b")
        ax2_single.tick_params(axis="y", labelcolor="g")
        ax_single.set_title(f"Performance on {dataset_name}")
        if lines_single:
            ax_single.legend(
                lines_single, [line.get_label() for line in lines_single], loc="best"
            )
        ax_single.grid(True, linestyle=":")

        safe_filename = re.sub(r"[^\w\-_.]", "_", f"{dataset_name}_curves.png")
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(save_folder, safe_filename))
        plt.close(fig_single)
        print(f"Saved individual plot to {os.path.join(save_folder, safe_filename)}")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig("validation_and_loss_curves_grid.png")
    plt.close(fig)
    print("\nCombined grid plot saved as validation_and_loss_curves_grid.png")


if __name__ == "__main__":
    try:
        # Create directories for the final outputs
        final_stats_dir = "final_best_configs_stats"
        final_configs_dir = "final_best_configs_runnable"

        results_dirs = [
            d
            for d in os.listdir("analysis_outputs")
            if os.path.isdir(os.path.join("analysis_outputs", d))
        ]

        all_data = {}
        for results_dir in tqdm(results_dirs, desc="Loading and Processing Datasets"):
            try:
                dataset_results = load_data(results_dir)
                all_data.update(dataset_results)

                # Get the dataframes for this dataset
                dataset_name = list(dataset_results.keys())[0]
                data_df = dataset_results[dataset_name]["data"]
                baseline_df = dataset_results[dataset_name]["baseline"]

                # Call the updated function with the baseline_df
                extract_and_save_best_config(
                    data_df=data_df,
                    dataset_name=dataset_name,
                    stat_dir=final_stats_dir,
                    config_dir=final_configs_dir,
                )

            except Exception as e:
                print(f"\nFailed to process directory {results_dir}: {e}")

        if all_data:
            plot_training_metrics(all_data)

    except FileNotFoundError:
        print("Error: The 'analysis_outputs' directory was not found.")
    except Exception as e:
        print(f"An unexpected error occurred in the main block: {e}")

    # List of datasets
    datasets = [
        "gaussian_reg",
        "MNIST",
        "CIFAR10",
        "IMAGENET100",
        "small_diffusion",
        "big_diffusion",
        "llama_9M",
        "llama_60M",
        "llama_350M",
        "llama_1B",
    ]  # ['gaussian_reg', 'MNIST','CIFAR10','IMAGENET100','small_diffusion','big_diffusion','llama_9M','llama_60M','llama_350M','llama_1B','llama_7B']
    val_results_fold = "final_best_configs_runnable_results"

    # Load all data into a dictionary
    all_data = {}
    for dataset in datasets:
        try:
            with open(f"{val_results_fold}/results_{dataset}.pkl", "rb") as f:
                result = pickle.load(f)
            with open(f"{val_results_fold}/results_baseline_{dataset}.pkl", "rb") as f:
                baseline = pickle.load(f)
            all_data[dataset] = {"data": result, "baseline": baseline}
        except Exception as e:
            print(f"An unexpected error occurred while loading data for {dataset}: {e}")

    stable_spam_baselines = "single_gpu_baseline_stable_spam/_results"
    result_files = [
        f"{stable_spam_baselines}/{r}" for r in os.listdir(stable_spam_baselines)
    ] + [
        "baseline_stable_spam_llama_350M_results/results_0.pkl",
        "baseline_stable_spam_llama_1B_results/results_0.pkl",
    ]
    for r in result_files:
        try:
            with open(f"{r}", "rb") as f:
                baseline_spam = CPU_Unpickler(f).load()
                dataset = baseline_spam["dataset"]
                all_data[dataset]["baseline_spam"] = baseline_spam
        except Exception as e:
            print(f"An unexpected error occurred while loading data for {r}: {e}")

    source_list = [
        {"key": "data", "label_prefix": "Best Config", "style": "-"},
        {"key": "baseline", "label_prefix": "Adam Baseline", "style": "--"},
        {"key": "baseline_spam", "label_prefix": "StableSPAM Baseline", "style": ".-"},
    ]
    # Generate the plot
    plot_validation_and_loss_curves(all_data, source_list)
