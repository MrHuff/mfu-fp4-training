#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import os
import pickle
from glob import glob
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import unicodedata
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

from sklearn.linear_model import LassoCV, Lasso


def generate_latex_table(df, columns_to_export, caption, label, output_path):
    """
    Generates and saves a LaTeX table from a pandas DataFrame.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        columns_to_export (list): List of column names to include in the table.
        caption (str): The table's caption.
        label (str): The table's label for referencing (e.g., 'tab:my_results').
        output_path (str): The file path to save the .tex file.
    """
    # Make a copy to avoid modifying the original dataframe
    df_export = df[columns_to_export].copy()

    # Clean up column names for better display in the paper
    df_export.columns = [col.replace("_", " ").title() for col in df_export.columns]

    # Generate the LaTeX string using pandas' built-in functionality
    latex_string = df_export.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=True,  # This handles special characters like '_'
        float_format="%.4f",  # Format floating point numbers to 4 decimal places
        na_rep="N/A",  # Representation for missing values
    )

    # Write the string to a .tex file
    with open(output_path, "w") as f:
        f.write(latex_string)
    print(f"LaTeX table successfully saved to {output_path}")


def linearConfigExtract(linearConfig):
    final_dict = {key: value for key, value in linearConfig.__dict__.items()}

    if "use_approx" in final_dict and isinstance(final_dict["use_approx"], dict):
        nested_dict = final_dict.pop("use_approx")
        final_dict.update(nested_dict)
    return final_dict


def plot_categorical_summary_heatmap(
    df, cols, output_file="nan_categorical_heatmap.png", normalize=True
):
    df_nan = df[df["min_metric"].isna()]
    if df_nan.empty:
        print(
            f"No NaN values in 'min_metric' for heatmap generation. Skipping {output_file}"
        )
        return
    counts_dict = {}
    for col in cols:
        counts = df_nan[col].value_counts(dropna=False)
        counts_dict[col] = counts

    all_categories = sorted(
        set(str(cat) for counts in counts_dict.values() for cat in counts.index)
    )
    heatmap_df = pd.DataFrame(index=all_categories, columns=cols).fillna(0)

    for col in cols:
        for cat, count in counts_dict[col].items():
            heatmap_df.at[str(cat), col] = count

    if normalize:
        heatmap_df = heatmap_df.div(heatmap_df.sum(axis=0), axis=1)

    plt.figure(figsize=(len(cols) * 1.5, len(heatmap_df) * 0.3 + 2))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        cbar_kws={"label": "Frequency" if normalize else "Count"},
    )
    plt.title("Categorical Feature Distributions in NaN min_metric Group")
    plt.ylabel("Category")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Heatmap saved to {output_file}")


def sanitize_column_name(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def analyze_hparam_effects_lasso_subsample(
    fig_name, df, target, categorical_cols, n_subsamples=1000, frac=0.7, alpha=0.05
):
    df = df.copy()
    encoder = OneHotEncoder(drop=None, sparse_output=False)
    X_cat = encoder.fit_transform(df[categorical_cols])
    feature_names_raw = encoder.get_feature_names_out(categorical_cols)
    feature_names = [sanitize_column_name(name) for name in feature_names_raw]

    X = pd.DataFrame(X_cat, columns=feature_names)
    y = df[target].values

    base_model = LassoCV(cv=5).fit(X, y)
    best_alpha = base_model.alpha_

    final_model = Lasso(alpha=best_alpha).fit(X, y)
    coefs = pd.Series(final_model.coef_, index=feature_names)
    y_pred = final_model.predict(X)
    r2 = r2_score(y, y_pred)
    n_samples = int(len(df) * frac)
    subsample_coefs = []

    for _ in tqdm(range(n_subsamples), desc="Lasso Subsampling"):
        sample_idx = np.random.choice(len(df), size=n_samples, replace=False)
        X_sub = X.iloc[sample_idx]
        y_sub = y[sample_idx]
        sub_model = Lasso(alpha=best_alpha, max_iter=10000).fit(X_sub, y_sub)
        subsample_coefs.append(sub_model.coef_)

    subsample_coefs = np.array(subsample_coefs)
    ci_lower = np.percentile(subsample_coefs, 100 * (alpha / 2), axis=0)
    ci_upper = np.percentile(subsample_coefs, 100 * (1 - alpha / 2), axis=0)

    coef_df = pd.DataFrame(
        {"coef": coefs, "ci_lower": ci_lower, "ci_upper": ci_upper}, index=feature_names
    ).sort_values("coef")

    plt.figure(figsize=(10, max(6, len(feature_names) * 0.25)))
    ax = sns.barplot(
        x="coef",
        y=coef_df.index,
        data=coef_df,
        palette="viridis",
        orient="h",
        errorbar=None,
    )
    lower_error = abs(coef_df["coef"] - coef_df["ci_lower"])
    upper_error = abs(coef_df["ci_upper"] - coef_df["coef"])
    asymmetric_error = [lower_error, upper_error]
    ax.errorbar(
        x=coef_df["coef"],
        y=coef_df.index,
        xerr=asymmetric_error,
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    plt.axvline(0, color="gray", linestyle="--")
    plt.title(
        f"Lasso Effects on {target} with CI via Subsampling\nR² = {r2:.3f}, α = {best_alpha:.4f}"
    )
    plt.xlabel("Effect Size")
    plt.tight_layout()
    plt.savefig(f"{fig_name}.png")
    plt.close()
    return final_model, coef_df


def analyze_hparam_effects(fig_name, df, target, categorical_cols):
    df = df.copy()
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    X_cat = encoder.fit_transform(df[categorical_cols])
    feature_names_raw = encoder.get_feature_names_out(categorical_cols)
    feature_names = [sanitize_column_name(name) for name in feature_names_raw]
    X = pd.DataFrame(X_cat, columns=feature_names)
    y = df[target].values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    coef = model.params.drop("const", errors="ignore").sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef.values, y=coef.index, palette="viridis")
    plt.axvline(0, color="gray", linestyle="--")
    plt.title(f"Effect of Hyperparameters on {target} (One-hot regression)")
    plt.xlabel("Effect Size (vs. baseline)")
    plt.tight_layout()
    plt.savefig(f"{fig_name}.png")
    plt.close()
    return model


def load_all_results(results_dir, jobs_dir=None):
    result_files = sorted(glob(os.path.join(results_dir, "results_*.pkl")))
    all_data = []
    for file in tqdm(result_files, desc=f"Loading results from {results_dir}"):
        with open(file, "rb") as f:
            result = pickle.load(f)
        row = {}
        training_metrics = result.get("training_metrics", [])
        if training_metrics and isinstance(training_metrics[0], dict):
            training_metrics = [v["loss"] for v in training_metrics]
        row["min_metric"] = min(training_metrics) if training_metrics else None
        row["metrics_len"] = len(training_metrics)
        row["start_metric"] = training_metrics[0] if training_metrics else None
        row["training_metrics"] = training_metrics
        if training_metrics and len(training_metrics) > 1:
            start_loss = training_metrics[0]
            min_loss = row["min_metric"]
            if start_loss != 0:
                row["pct_improvement"] = 100 * (start_loss - min_loss) / start_loss
            else:
                row["pct_improvement"] = None
        else:
            row["pct_improvement"] = None
        meta = result.copy()
        meta.pop("training_metrics", None)
        for k, v in meta.items():
            if hasattr(v, "name"):
                row[k] = v.name
            else:
                row[k] = v
        if jobs_dir is not None:
            match = re.search(r"results_(\d+)\.pkl", file)
            if match:
                number_string = match.group(1)
                job_config_path = os.path.join(jobs_dir, f"config_{number_string}.pkl")
                if os.path.exists(job_config_path):
                    with open(job_config_path, "rb") as f:
                        job_config = pickle.load(f)
                    linearConfig = job_config.get("MXLinearDimConfig")
                    if linearConfig:
                        meta_from_job = linearConfigExtract(linearConfig)
                        for k, v in meta_from_job.items():
                            if hasattr(v, "name"):
                                row[k] = v.name
                            else:
                                row[k] = v
                else:
                    print(f"Warning: Config file not found for {file}: {job_config_path}")
            else:
                print(f"Warning: Could not extract config ID from filename: {file}")
        row["config_id"] = int(os.path.basename(file).split("_")[-1].split(".")[0])
        all_data.append(row)
    df = pd.DataFrame(all_data)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("N/A")
    return df


def filter_comparable_to_baseline(
    df, baseline_df, max_pct_drop=20, max_loss_increase=0.5
):
    # Drop rows with missing min_metric
    df_filtered_nan = df.dropna(subset=["min_metric"]).copy()
    baseline_candidates = baseline_df[baseline_df["pct_improvement"].notnull()]
    if baseline_candidates.empty:
        print("No valid baseline found. Falling back to top 20% of current dataframe.")
        num_to_keep = max(1, int(len(df_filtered_nan) * 0.20))
        return df_filtered_nan.nsmallest(num_to_keep, "min_metric").copy()

    best_baseline = baseline_candidates.loc[baseline_candidates["min_metric"].idxmin()]
    baseline_pct_improvement = best_baseline["pct_improvement"]
    baseline_min_metric = best_baseline["min_metric"]

    def is_comparable(row):
        if pd.isnull(row["pct_improvement"]) or pd.isnull(row["min_metric"]):
            return False
        pct_ok = row["pct_improvement"] >= baseline_pct_improvement * (
            1 - max_pct_drop / 100
        )
        loss_ok = row["min_metric"] <= baseline_min_metric * (1 + max_loss_increase)
        return pct_ok and loss_ok

    filtered_df = df_filtered_nan[df_filtered_nan.apply(is_comparable, axis=1)].copy()
    if filtered_df.empty or filtered_df.shape[0] < 50:
        print(
            f"Filtered dataframe has less than 50 entries ({filtered_df.shape[0]}). Falling back to top 20% of current dataframe."
        )
        num_to_keep = max(1, int(len(df_filtered_nan) * 0.20))
        return df_filtered_nan.nsmallest(num_to_keep, "min_metric").copy()
    return filtered_df


def group_by_summary(df, group_col):
    metric_cols = ["min_metric", "pct_improvement", "start_metric", "metrics_len"]
    existing_cols = [col for col in metric_cols if col in df.columns]
    if not existing_cols:
        raise ValueError("No relevant metric columns found in dataframe.")
    grouped = df.groupby(group_col)[existing_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] for col in grouped.columns.values
    ]
    return grouped


HYPARAM_COLS = [
    "scale_type",
    "block_size",
    "smooth",
    "alpha",
    "stepGradient",
    "temperature",
    "k",
    "use_hadamard",
    "qGradient",
    "SR",
    "optimiser",
    "loss_scaling",
    "fp_scale_factor",
    "roundMode",
    "use_tensor_scaling",
    "tensor_scaling_grad_est",
]


def getComplexityPoints(row):
    smooth = 0 if row["smooth"] == "STE" else 1
    stepGradient = 0 if row["stepGradient"] == "STE" else 1
    use_hadamard = 0 if row["use_hadamard"] in ["N/A", None, "", np.nan] else 1
    qGradient = 0 if row["qGradient"] == "STE" else 1
    SR = 0 if row["SR"] in ["N/A", None, "", np.nan] else 1
    use_tensor_scaling = 0 if not row["use_tensor_scaling"] else 1
    tensor_scaling_grad_est = 0 if row["tensor_scaling_grad_est"] == "ignore" else 1
    return sum(
        [
            smooth,
            stepGradient,
            use_hadamard,
            qGradient,
            SR,
            use_tensor_scaling,
            tensor_scaling_grad_est,
        ]
    )


def run_analysis_for_config(
    results_dir, dataset_name, jobs_dir=None, baseline_dir=None, cols=HYPARAM_COLS
):
    output_dir = os.path.join("analysis_outputs", results_dir)
    os.makedirs(output_dir, exist_ok=True)
    results_csv = f"{output_dir}/{results_dir}.csv"
    if not os.path.exists(results_csv):
        print(f"Loading results from {results_dir}...")
        df = load_all_results(results_dir, jobs_dir)
        df["complexity_points"] = df.apply(lambda row: getComplexityPoints(row), axis=1)
        df.to_csv(results_csv, index=False)
    else:
        print(f"Loading cached results from {results_csv}...")
        df = pd.read_csv(results_csv)
        df["complexity_points"] = df.apply(lambda row: getComplexityPoints(row), axis=1)

    median_metric = df["min_metric"].median()
    df["score"] = ((median_metric - df["min_metric"]) / median_metric) * (
        1 / df["complexity_points"]
    )

    print(f"Original DataFrame shape: {df.shape}")
    plot_categorical_summary_heatmap(
        df, cols, output_file=f"{output_dir}/nan_categorical_heatmap.png"
    )
    df = df[df["min_metric"].notna()].copy()
    print(f"DataFrame shape after dropping NaN min_metric: {df.shape}")

    df_baseline = None
    baseline_csv = f"{output_dir}/baseline_combined.csv"

    if not os.path.exists(baseline_csv):
        print(f"Loading baseline results from {baseline_dir}...")
        dfs = []
        # ensure baseline_dir is a list
        if isinstance(baseline_dir, str):
            baseline_dir = [baseline_dir]

        for dir_path in baseline_dir:
            df_tmp = load_all_results(dir_path)
            dfs.append(df_tmp)

        # concatenate all DataFrames
        df_baseline = pd.concat(dfs, ignore_index=True)
        df_baseline.to_csv(baseline_csv, index=False)
    else:
        print(f"Loading cached baseline results from {baseline_csv}...")
        df_baseline = pd.read_csv(baseline_csv)

    # filter by dataset
    df_baseline = df_baseline[df_baseline["dataset"] == dataset_name].copy()
    if df_baseline is None or df_baseline.empty:
        print(
            "Error: No valid baseline could be established. Skipping baseline-dependent analysis."
        )
    else:
        # Filter comparable configs and generate reports
        filtered_df = filter_comparable_to_baseline(df, df_baseline)
        filtered_df = filtered_df.sort_values("min_metric", ascending=True)
        print(f"[{results_dir}] Filtered comparable configs found: {len(filtered_df)}")
        df.sort_values("min_metric", ascending=True).head(5).to_csv(
            os.path.join(output_dir, "top_5_configs.csv"), index=False
        )

        # *** NEW: GENERATE LATEX TABLE ***
        if not filtered_df.empty:
            top_5_df = df.sort_values("min_metric", ascending=True).head(5).copy()
            top_5_df["Source"] = "Experiment"

            baseline_candidates = df_baseline[df_baseline["min_metric"].notnull()]
            if not baseline_candidates.empty:
                df_baseline = df_baseline[df_baseline["dataset"] == dataset_name].copy()
                df_baseline["Source"] = "Baseline"

                # Combine baseline and top 5 results
                comparison_df = pd.concat([df_baseline, top_5_df], ignore_index=True)
                comparison_df = comparison_df[
                    [
                        "Source",
                        "min_metric",
                        "scale_type",
                        "smooth",
                        "stepGradient",
                        "use_hadamard",
                        "qGradient",
                        "SR",
                        "optimiser",
                        "loss_scaling",
                        "roundMode",
                        "use_tensor_scaling",
                        "tensor_scaling_grad_est",
                    ]
                ]
                # Define columns to include in the LaTeX table
                table_cols = ["Source", "min_metric"] + [
                    col for col in HYPARAM_COLS if col in comparison_df.columns
                ]

                # Generate caption and label
                caption = f"Top 5 configurations for {dataset_name.replace('_', ' ')} compared to the best baseline run."
                label = f"tab:{dataset_name}_top5_comparison"
                output_file = os.path.join(output_dir, "top_5_comparison_table.tex")

                generate_latex_table(
                    comparison_df, table_cols, caption, label, output_file
                )

        # Run analysis on filtered data
        if not filtered_df.empty and filtered_df.shape[0] > 5:
            bang_for_the_buck_df = filtered_df.sort_values(["score"], ascending=False)
            bang_for_the_buck_df.to_csv(
                os.path.join(output_dir, "bang_for_the_buck.csv"), index=False
            )

            # TODO: bang for the buck metric is going to be based on median performance of the entire thing.

            _, coef_df = analyze_hparam_effects_lasso_subsample(
                os.path.join(output_dir, "lasso_filtered"),
                filtered_df,
                "min_metric",
                cols,
                n_subsamples=100,
                frac=0.7,
                alpha=1e-3,
            )
            coef_df.to_csv(os.path.join(output_dir, "lasso_filtered_param.csv"))

            ols_model = analyze_hparam_effects(
                os.path.join(output_dir, "OLS_filtered"), filtered_df, "min_metric", cols
            )
            with open(
                os.path.join(output_dir, "model_summary_OLS_filtered.txt"), "w"
            ) as f:
                f.write(ols_model.summary().as_text())


if __name__ == "__main__":
    configs = [
        # {
        #     "results_dir": "regression_jobs_tensor_scaling_results",
        #     "jobs_dir": "regression_jobs_tensor_scaling",
        #     "dataset_name": "gaussian_reg",
        #     "baseline_dir": ["baseline_jobs_small_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        #         {
        #     "results_dir": "CIFAR10_jobs_tensor_scaling_results",
        #     "jobs_dir": "CIFAR10_jobs_tensor_scaling",
        #     "dataset_name": "CIFAR10",
        #     "baseline_dir": ["baseline_jobs_small_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        # {
        #     "results_dir": "MNIST_jobs_tensor_scaling_results",
        #     "jobs_dir": "MNIST_jobs_tensor_scaling",
        #     "dataset_name": "MNIST",
        #     "baseline_dir": ["baseline_jobs_small_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        # {
        #     "results_dir": "IMAGENET100_jobs_tensor_scaling_results",
        #     "jobs_dir": "IMAGENET100_jobs_tensor_scaling",
        #     "dataset_name": "IMAGENET100",
        #     "baseline_dir": ["baseline_jobs_big_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        #         {
        #     "results_dir": "llama9M_jobs_tensor_scaling_results",
        #     "jobs_dir": "llama9M_jobs_tensor_scaling",
        #     "dataset_name": "llama_9M",
        #     "baseline_dir": ["baseline_llama_9M_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        #         {
        #     "results_dir": "llama_60M_jobs_tensor_scaling_results",
        #     "jobs_dir": "llama_60M_jobs_tensor_scaling",
        #     "dataset_name": "llama_60M",
        #     "baseline_dir": ["baseline_llama_60M_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        {
            "results_dir": "llama_350M_jobs_tensor_scaling_results",
            "jobs_dir": "llama_350M_jobs_tensor_scaling",
            "dataset_name": "llama_350M",
            "baseline_dir": [
                "baseline_llama_350M_results",
                "baseline_stable_spam_llama_350M_results",
            ],
        },
        #         {
        #     "results_dir": "small_diffusion_jobs_tensor_scaling_results",
        #     "jobs_dir": "small_diffusion_jobs_tensor_scaling",
        #     "dataset_name": "small_diffusion",
        #     "baseline_dir": ["baseline_jobs_small_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        # {
        #     "results_dir": "big_diffusion_jobs_tensor_scaling_results",
        #     "jobs_dir": "big_diffusion_jobs_tensor_scaling",
        #     "dataset_name": "big_diffusion",
        #     "baseline_dir": ["baseline_jobs_big_results", "single_gpu_baseline_stable_spam/_results"]
        # },
        {
            "results_dir": "llama_1B_jobs_tensor_scaling_results",
            "jobs_dir": "llama_1B_jobs_tensor_scaling",
            "dataset_name": "llama_1B",
            "baseline_dir": [
                "baseline_llama_1B_results",
                "baseline_stable_spam_llama_1B_results",
            ],
        },
    ]

    for config in configs:
        run_analysis_for_config(**config, cols=HYPARAM_COLS)
