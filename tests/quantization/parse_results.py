import os
import pickle
from glob import glob
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
import re
import unicodedata
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import ast

import re
import unicodedata
from sklearn.linear_model import LassoCV, Lasso
import torch
import io
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.lines import Line2D

def process_x_axis_names(dataset_name, src_data, metric):
    """Return x-axis values and label ('tokens' or 'epoch') for plotting curves."""
    token_count = {
        "llama_9M":  9000000 * 100,
        "llama_60M": 60000000 * 100,
        "llama_350M": 350000000 * 42,
        "llama_1B": 1000000000 * 42,
    }

    dict_list = src_data.get(metric, [])

    # Handle stringified list
    if isinstance(dict_list, str):
        try:
            dict_list = ast.literal_eval(dict_list)
        except Exception:
            return np.array([]), "epoch"
    if (dict_list is None) or (isinstance(dict_list, float) and math.isnan(dict_list)):
        return []
    if not dict_list:
        return []
    # If it's empty, bail gracefully
    # Handle llama datasets → use tokens as x-axis
    if "llama" in dataset_name:
        if isinstance(dict_list[0], dict) and "epoch" in dict_list[0]:
            total_epochs = dict_list[-1]["epoch"]
            token_per_epoch = token_count.get(dataset_name, 1) / total_epochs
            x_axis = [token_per_epoch * el["epoch"] for el in dict_list]
        else:
            token_per_step =  token_count.get(dataset_name, 1) /len(dict_list)
            x_axis = np.arange(1,len(dict_list)+1) * token_per_step
        return x_axis, "tokens"
    
    if "big_diffusion" in dataset_name:
        v =  np.arange(len(dict_list))
        scale = 199/v[-1]
        return v*scale, "steps"

    # Otherwise → just use sequential epochs
    return np.arange(len(dict_list)), "epoch"


def extract_curve(curve, val_key="eval_loss", loss_key="loss"):
    """Extracts numerical lists from tensors, dicts, or stringified lists/dicts.
    Returns [] if curve is empty, malformed, or contains NaN values.
    """
    if isinstance(curve, str):
        try:
            curve = ast.literal_eval(curve)
        except Exception:
            return []
    if not curve:
        return []
    if (curve is None) or (isinstance(curve, float) and math.isnan(curve)):
        return []
    # If curve is a string like "[...]" → turn into real list/dict

    # If values are tensors → extract scalars
    if hasattr(curve[0], "item"):
        curve = [v.item() for v in curve]

    # If values are dicts → pull out the right key
    if isinstance(curve[0], dict):
        key = val_key if val_key in curve[0] else loss_key
        curve = [v[key] for v in curve]

    # Final NaN check
    return curve

def plot_validation_and_loss_curves(all_data, curve_sources, save_folder='individual_plots'):
    """
    Plots validation metric and training loss curves for multiple datasets and multiple sources (e.g., best config, baselines).
    """
    val_metric_key = 'val_metrics'
    train_loss_key = 'training_metrics'

    if not all_data:
        print("No data to plot.")
        return

    os.makedirs(save_folder, exist_ok=True)

    for i, (dataset_name, dataset_info) in enumerate(all_data.items()):
        fig_single, ax_single = plt.subplots(figsize=(10, 7))
        ax2_single = ax_single.twinx()
        lines_single = []

        for source in curve_sources:
            src_data = dataset_info.get(source["key"], {})
            val_curve = extract_curve(src_data.get(val_metric_key, []), 'eval_loss', 'loss')
            loss_curve = extract_curve(src_data.get(train_loss_key, []), 'eval_loss', 'loss')


            if loss_curve:
                x_axis, x_name = process_x_axis_names(dataset_name,src_data,train_loss_key)
                lines_single.extend(ax2_single.plot(x_axis,loss_curve,source["color"] +  '-' , 
                                                    label=f"{source['label_prefix']} Train Loss"))
            if val_curve:
                x_axis, x_name = process_x_axis_names(dataset_name,src_data,val_metric_key)
                lines_single.extend(ax_single.plot(x_axis,val_curve, source["color"] + '--' , 
                                                   label=f"{source['label_prefix']} Val Metric"))

        ax_single.set_xlabel(x_name)
        ax_single.set_ylabel('Validation Metric')
        ax2_single.set_ylabel('Training Loss')
        ax_single.tick_params(axis='y')
        ax2_single.tick_params(axis='y')
        ax_single.set_title(f"Performance on {dataset_name}")
        if lines_single:
            ax_single.legend(lines_single, [l.get_label() for l in lines_single], loc='best')
        ax_single.grid(True, linestyle=':')

        safe_filename = re.sub(r'[^\w\-_.]', '_', f"{dataset_name}_curves.pdf")
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(save_folder, safe_filename))
        plt.close(fig_single)
        print(f"Saved individual plot to {os.path.join(save_folder, safe_filename)}")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Intercept tensor storage loading and force CPU mapping
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def latex_cols(col):
    col_name = {
        'scale_type':'Scale', 
        'block_size':r'\makecell{Block \\ size}', 
        'smooth':r'\makecell{Max \\ grad.}', 
        'stepGradient': r'\makecell{Quant.\\grad}',  
        'use_hadamard':'Hadamard', 
        'qGradient': r'\makecell{Scale\\grad}',
        'SR':'SR', 
        'optimiser':'Optimiser',
        'loss_scaling': r'\makecell{Loss\\scaling}',
        'roundMode':'Round mode',
        'use_tensor_scaling': r'\makecell{Tensor\\scaling}',
        'tensor_scaling_grad_est': r'\makecell{Tensor \\grad}',
        'complexity_points':r'\makecell{Complexity\\points}',
        'score':'Score',
        'nan_handling_mode': r'\makecell{NaN\\mode}',
        'val_metrics_min': r'\makecell{Val\\loss}',
        'training_metrics_min': r'\makecell{Train\\loss}'
    }


    if col in col_name.keys():
        return col_name[col]
        
    return col.replace('_', ' ').title()
def latex_safe(s):
    if isinstance(s, str):
        if 'Adam' in s:
            return 'Adam'
        elif 'StableSPAM' in s:
            return 'StableSPAM'

        return s.replace('_', r'\_')
    
    return s
def generate_latex_table(df, columns_to_export, caption, label, output_path):

    """
    Generates and saves a LaTeX table from a pandas DataFrame.
    """
    # Make a copy to avoid modifying the original dataframe
    df['block_size'] = df['block_size'].apply(
        lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (int, float)) else x
    )
    df_export = df[columns_to_export].copy()
    df_export = df_export.sort_values(['dataset','Source','scale_type'])

    # Clean up column names for better display in the paper
    df_export.columns = [latex_cols(col) for col in df_export.columns]
    df_export = df_export.applymap(latex_safe)

    # Generate the LaTeX string using pandas' built-in functionality
    latex_string = df_export.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=False,
        float_format="%.3f",
        na_rep='N/A'
    )

    # Write the string to a .tex file
    with open(output_path, 'w') as f:
        f.write(latex_string)
    print(f"LaTeX table successfully saved to {output_path}")


def linearConfigExtract(linearConfig):
    final_dict = {key: value for key, value in linearConfig.__dict__.items()}

    if 'use_approx' in final_dict and isinstance(final_dict['use_approx'], dict):
        nested_dict = final_dict.pop('use_approx')
        final_dict.update(nested_dict)
    return final_dict

def plot_categorical_summary_heatmap(df, cols,metric_key, output_file="nan_categorical_heatmap.png", normalize=True):
    df_nan = df[df[metric_key].isna()]
    if df_nan.empty:
        print(f"No NaN values in metric_key for heatmap generation. Skipping {output_file}")
        return
    counts_dict = {}
    for col in cols:
        counts = df_nan[col].value_counts(dropna=False)
        counts_dict[col] = counts

    all_categories = sorted(set(str(cat) for counts in counts_dict.values() for cat in counts.index))
    heatmap_df = pd.DataFrame(index=all_categories, columns=cols).fillna(0)

    for col in cols:
        for cat, count in counts_dict[col].items():
            heatmap_df.at[str(cat), col] = count

    if normalize:
        heatmap_df = heatmap_df.div(heatmap_df.sum(axis=0), axis=1)

    plt.figure(figsize=(len(cols) * 1.5, len(heatmap_df) * 0.3 + 2))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".2f" if normalize else "d",
        cmap="Blues", cbar_kws={'label': 'Frequency' if normalize else 'Count'}
    )
    plt.title("Categorical Feature Distributions in NaN min_metric Group")
    plt.ylabel("Category")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"Heatmap saved to {output_file}")


def sanitize_column_name(name):
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def analyze_hparam_effects_lasso_subsample(fig_name,df, target, categorical_cols, n_subsamples=1000, frac=0.7, alpha=0.05):
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

    coef_df = pd.DataFrame({
        'coef': coefs,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }, index=feature_names).sort_values('coef')

    plt.figure(figsize=(10, max(6, len(feature_names) * 0.25)))
    ax = sns.barplot(x='coef', y=coef_df.index, data=coef_df, palette='viridis', orient='h', errorbar=None)
    lower_error = abs(coef_df['coef'] - coef_df['ci_lower'])
    upper_error = abs(coef_df['ci_upper'] - coef_df['coef'])
    asymmetric_error = [lower_error, upper_error]
    ax.errorbar(x=coef_df['coef'], y=coef_df.index,
                xerr=asymmetric_error,
                fmt='none', ecolor='black', capsize=3)
    plt.axvline(0, color='gray', linestyle='--')
    plt.title(f"Lasso Effects on {target} with CI via Subsampling\nR² = {r2:.3f}, α = {best_alpha:.4f}")
    plt.xlabel("Effect Size")
    plt.tight_layout()
    plt.savefig(f"{fig_name}.png")
    plt.close()
    return final_model, coef_df

def analyze_hparam_effects(fig_name,df, target, categorical_cols):
    df = df.copy()
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_cat = encoder.fit_transform(df[categorical_cols])
    feature_names_raw = encoder.get_feature_names_out(categorical_cols)
    feature_names = [sanitize_column_name(name) for name in feature_names_raw]
    X = pd.DataFrame(X_cat, columns=feature_names)
    y = df[target].values
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    coef = model.params.drop("const", errors='ignore').sort_values()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coef.values, y=coef.index, palette="viridis")
    plt.axvline(0, color='gray', linestyle='--')
    plt.title(f"Effect of Hyperparameters on {target} (One-hot regression)")
    plt.xlabel("Effect Size (vs. baseline)")
    plt.tight_layout()
    plt.savefig(f'{fig_name}.png')
    plt.close()
    return model

def process_metrics(result: dict, key: str = "training_metrics") -> dict:
    """
    Extracts and summarizes metrics from a result dictionary.
    """
    row = {}
    metrics = result.get(key, [])

    # Convert list of dicts to list of losses if necessary
    if metrics and isinstance(metrics[0], dict):
        if key=='training_metrics':
            metrics = [v['loss'] for v in metrics]
        else:
            metrics = [v['eval_loss'] for v in metrics]
    if metrics and isinstance(metrics[0], torch.Tensor):
        metrics = [v.item() for v in metrics]
    metrics = [m for m in metrics if m is not None]
    row[key] = metrics
    row[f"{key}_min"] = min(metrics) if metrics else None
    row[f"{key}_len"] = len(metrics)
    row[f"{key}_start"] = metrics[0] if metrics else None
    # Compute percent improvement if applicable
    if metrics and len(metrics) > 1:
        start_value = metrics[0]
        min_value = row[f"{key}_min"]
        row[f"{key}_pct_improvement"] = (
            100 * (start_value - min_value) / start_value if start_value != 0 else None
        )
    else:
        row[f"{key}_pct_improvement"] = None

    return row

def _load_single_result(file, jobs_dir=None):
    
    with open(file, 'rb') as f:
        result = CPU_Unpickler(f).load()

    training_data = process_metrics(result, 'training_metrics')
    val_data = process_metrics(result, 'val_metrics')

    meta = result.copy()
    meta.pop("training_metrics", None)
    meta.pop("val_metrics", None)
    
    row = {k: getattr(v, "name", v) for k, v in meta.items()}
    row = {**row, **training_data, **val_data}

    if jobs_dir is not None:
        match = re.search(r'results_(\d+)\.pkl', file)
        if match:
            number_string = match.group(1)
            job_config_path = os.path.join(jobs_dir, f'config_{number_string}.pkl')
            if os.path.exists(job_config_path):
                with open(job_config_path, 'rb') as f:
                    job_config = pickle.load(f)
                linearConfig = job_config.get('MXLinearDimConfig')
                if linearConfig:
                    meta_from_job = linearConfigExtract(linearConfig)
                    for k, v in meta_from_job.items():
                        row[k] = getattr(v, "name", v)
            else:
                print(f"Warning: Config file not found for {file}: {job_config_path}")
        else:
            print(f"Warning: Could not extract config ID from filename: {file}")

    row["config_id"] = int(os.path.basename(file).split('_')[-1].split('.')[0])

    return row

def load_all_results(results_dir, jobs_dir=None, max_workers=None):
    result_files = sorted(glob(os.path.join(results_dir, 'results_*.pkl')))
    all_data = []

    if max_workers is None:
        max_workers = max(1, os.cpu_count() // 4)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_single_result, f, jobs_dir): f for f in result_files}
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Loading results from {results_dir}"):
            result = f.result()
            if result is not None:
                all_data.append(result)

    df = pd.DataFrame(all_data)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("N/A")
    return df

def filter_comparable_to_baseline(df, baseline_df,metric_key, max_pct_drop=20, max_loss_increase=0.5):
    # Drop rows with missing min_metric
    df_filtered_nan = df.dropna(subset=[metric_key]).copy()
    baseline_candidates = baseline_df
    if baseline_candidates.empty:
        print("No valid baseline found. Falling back to top 20% of current dataframe.")
        num_to_keep = max(1, int(len(df_filtered_nan) * 0.20))
        return df_filtered_nan.nsmallest(num_to_keep, metric_key).copy()

    best_baseline = baseline_candidates.loc[baseline_candidates[metric_key].idxmin()]
    baseline_min_metric = best_baseline[metric_key]

    def is_comparable(row):
        loss_ok = row[metric_key] <= baseline_min_metric * (1 + max_loss_increase)
        return loss_ok

    filtered_df = df_filtered_nan[df_filtered_nan.apply(is_comparable, axis=1)].copy()
    if filtered_df.empty or filtered_df.shape[0]<50:
        print(f"Filtered dataframe has less than 50 entries ({filtered_df.shape[0]}). Falling back to top 20% of current dataframe.")
        num_to_keep = max(1, int(len(df_filtered_nan) * 0.20))
        return df_filtered_nan.nsmallest(num_to_keep, metric_key).copy()
    return filtered_df


HYPARAM_COLS = [
    'scale_type', 'block_size', 'smooth', 'alpha', 'stepGradient',
    'temperature', 'k', 'use_hadamard', 'qGradient', 'SR', 'optimiser','loss_scaling','fp_scale_factor','roundMode','use_tensor_scaling','tensor_scaling_grad_est','complexity_points','score',
    'nan_handling_mode'

]

def score(row):
    score = row['diff'] * (1 / max(row["complexity_points"],1.0)) if row['diff']>=0 else row['diff'] * (max(row["complexity_points"],1.0))
    return score

def calculateScore(reference_metric, df,metric_key):
    df['diff'] = (reference_metric - df[metric_key]) / reference_metric
    df['score'] = df.apply(lambda row: score(row), axis=1)
    return df


def getComplexityPoints(row):
    smooth = 0 if row['smooth']=='STE' else 3.
    stepGradient = 0 if row['stepGradient']=='STE' else 2.0
    use_hadamard = 0 if pd.isna(row['use_hadamard']) or row['use_hadamard'] in {'N/A', '','None_exact'} else 1
    qGradient = 0 if row['qGradient']=='STE' else 1.5
    SR = 0 if row['SR']  in ['N/A',None,'', np.nan,'None_exact'] else 0.5
    use_tensor_scaling = 0 if  row['use_tensor_scaling'] in [False,'N/A',None] else 0.5
    loss_scaling = 0 if not row['loss_scaling'] else 0.5
    optimiser = 0.5 if 'SPAM' in row['optimiser'] else 0
    scale_stoc_round = 0.25 if row['roundMode'] == 'Stochastic' else 0 
    tensor_scaling_grad_est = 0 if row['tensor_scaling_grad_est'] in ['N/A',None,'', np.nan,'None_exact','ignore'] else 3.0


    return sum([smooth,stepGradient,use_hadamard,qGradient,SR,use_tensor_scaling,tensor_scaling_grad_est,loss_scaling,optimiser,scale_stoc_round])

def bang_for_the_buck_per_scale(df):
    """
    For each scale_type, returns the best positive and negative score configs.
    """
    result = {}
    for scale_type, group_df in df.groupby('scale_type'):
        positive_scores_df = group_df[group_df['score'] >= 0]
        negative_scores_df = group_df[group_df['score'] < 0]

        sorted_positive = positive_scores_df.sort_values(
            by=['score', 'complexity_points'], ascending=[False, True]
        ).head(1)

        sorted_negative = negative_scores_df.sort_values(
            by=['score', 'complexity_points'], ascending=[False, True] 
        ).head(1)

        result[scale_type] = {
            'positive_scores': sorted_positive,
            'negative_scores': sorted_negative
        }
    return result

def top_per_scale(df, metric_key, top_n=1, ascending=True):
    """
    Returns the top_n configurations per scale_type based on metric_key.
    """
    result = {}
    for scale_type, group_df in df.groupby('scale_type'):
        sorted_df = group_df.sort_values(
            by=metric_key, ascending=ascending
        ).head(top_n).copy()
        result[scale_type] = sorted_df
    return result

def plot_complexity_vs_score(df, df_pos, df_neg, df_loss, pure_fp4, dataset_name, output_dir):
    """
    Generates a scatter plot of complexity points vs. score with improved clarity.
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    # --- 1. Create a consistent color map for scale types ---
    unique_scales = sorted(df['scale_type'].unique())
    palette = sns.color_palette('viridis', n_colors=len(unique_scales))
    color_map = dict(zip(unique_scales, palette))

    # --- 2. Plot all data points ---
    sns.scatterplot(
        data=df,
        x='complexity_points',
        y='score',
        hue='scale_type',
        hue_order=unique_scales,
        palette=color_map,
        alpha=0.5,
        s=50,
        ax=ax
    )

    # --- 3. Plot and annotate special points ---
    point_types = {
        'Pos': (df_pos, '*', 15),
        'Neg': (df_neg, 'X', 10),
        'Loss': (df_loss, 'D', 10)
    }

    for label, (data, marker, size) in point_types.items():
        if not data.empty:
            for _, row in data.iterrows():
                scale_color = color_map.get(row['scale_type'])
                ax.plot(row['complexity_points'], row['score'],
                        marker=marker,
                        color=scale_color,
                        markersize=size,
                        markeredgecolor='black',
                        linestyle='None',
                        label='_nolegend_') # Hide from legend
                ax.annotate(
                    f"{label} ({row['scale_type']})",
                    (row['complexity_points'], row['score']),
                    textcoords="offset points",
                    xytext=(0, -20 if marker != '*' else 15),
                    ha='center',
                    fontsize=9,
                    color=scale_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=scale_color, lw=0.5, alpha=0.8))

    # --- 4. Plot and annotate reference lines ---
    # BFLOAT16 Baseline
    ax.axhline(0, color='grey', linestyle='-', linewidth=1.5, zorder=0)
    ax.annotate(
        'BFLOAT16 Baseline', xy=(1, 0), xycoords=('axes fraction', 'data'),
        xytext=(5, 0), textcoords='offset points', ha='left', va='center',
        fontsize=10, color='dimgray')

    # Pure FP4 References
    if not pure_fp4.empty:
        for _, row in pure_fp4.iterrows():
            ax.axhline(y=row['score'], color='black', linestyle=':', linewidth=1.5, zorder=0)
            ax.annotate(
                f"Pure FP4 ({row['scale_type']})", xy=(1, row['score']), xycoords=('axes fraction', 'data'),
                xytext=(5, 0), textcoords='offset points', ha='left', va='center',
                fontsize=10, color='black')

    # --- 5. Final plot styling and legend ---
    plt.title(f'Complexity vs. Score for {dataset_name}')
    plt.xlabel('Complexity Points')
    plt.ylabel('Score (Higher is Better)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create a new, clean legend below the plot
    handles, labels = ax.get_legend_handles_labels()
    
    proxy_markers = [
        Line2D([0], [0], marker='*', color='grey', label='Best Score (Pos)', linestyle='None', markersize=10),
        Line2D([0], [0], marker='X', color='grey', label='Best Score (Neg)', linestyle='None', markersize=8),
        Line2D([0], [0], marker='D', color='grey', label='Best Loss', linestyle='None', markersize=8),
        Line2D([0], [0], color='black', linestyle=':', label='Pure FP4 Score')
    ]
    handles.extend(proxy_markers)
    labels.extend([h.get_label() for h in proxy_markers])

    # Remove the original legend from inside the plot
    ax.get_legend().remove()
    
    # Add the combined legend below the plot
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=4, title="Legend")
    
    fig.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust for bottom legend

    safe_filename = re.sub(r'[^\w\-_.]', '_', f"{dataset_name}_complexity_vs_score.pdf")
    output_path = os.path.join(output_dir, safe_filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved complexity vs. score plot to {output_path}")

    
def run_analysis_for_config(results_dir, dataset_name, jobs_dir=None, baseline_dir=None, cols=HYPARAM_COLS):
    fold_name = 'analysis_outputs_ue5m3'
    output_dir = os.path.join(fold_name, dataset_name)

    source_list = [
        {
            "key": 'data_loss',
            "label_prefix": "Best Loss",
            "color": 'b'
        },
                {
            "key": 'data_score',
            "label_prefix": "Best score",
            "color": 'g'
        },
        {
            "key": 'baseline',
            "label_prefix": "BF16",
            "color": 'k'
        },
        {
            "key": 'fp4_e4m3_scale',
            "label_prefix": "Pure NVFP4",
            "color": 'r'
        },
        {
            "key": 'fp4_e8m0_scale',
            "label_prefix": "Pure MXFP4",
            "color": 'y'
        },
            {
            "key": 'fp4_e5m3_scale',
            "label_prefix": "Pure FP4(E5M3)",
            "color": 'y'
        },
    ]
            
    analysis_cols =  [
    'scale_type', 'block_size', 'smooth', 'alpha', 'stepGradient',
    'temperature', 'k', 'use_hadamard', 'qGradient', 'SR', 'optimiser','loss_scaling','fp_scale_factor','roundMode','use_tensor_scaling','tensor_scaling_grad_est','nan_handling_mode'
]
    os.makedirs(output_dir, exist_ok=True)
    results_csv = f"{output_dir}/raw_data.csv"
    for metric_key in ['val_metrics_min']:

        if not os.path.exists(results_csv):
            print(f"Loading results from directories and creating cache...")
            
            all_dfs = [load_all_results(res_dir, j_dir) for res_dir,j_dir in zip(results_dir,jobs_dir)]
            df = pd.concat(all_dfs, ignore_index=True)
            df.sort_values('config_id').to_csv(results_csv, index=False)
        else:
            print(f"Loading cached results from {results_csv}...")
            df = pd.read_csv(results_csv)
        df = df[df['dataset'] == dataset_name]

        if 'nan_handling_mode' not in df.columns:
            df['nan_handling_mode'] = 'nearest_subnormal'
        df['nan_handling_mode'] = df['nan_handling_mode'].fillna('nearest_subnormal')
                
        df['complexity_points'] = df.apply(lambda row: getComplexityPoints(row), axis=1)

        df_baseline = None
        baseline_csv = f"{output_dir}/baseline_combined.csv"

        if not os.path.exists(baseline_csv):
            print(f"Loading baseline results from {baseline_dir}...")
            dfs = []
            if isinstance(baseline_dir, str):
                baseline_dir = [baseline_dir]
            
            for dir_path in baseline_dir:
                df_tmp = load_all_results(dir_path)
                dfs.append(df_tmp)
            
            df_baseline = pd.concat(dfs, ignore_index=True)
            df_baseline.to_csv(baseline_csv, index=False)
        else:
            print(f"Loading cached baseline results from {baseline_csv}...")
            df_baseline = pd.read_csv(baseline_csv)

        os.makedirs(f'{output_dir}/{metric_key}',exist_ok=True)
        print(f"Original DataFrame shape: {df.shape}")
        plot_categorical_summary_heatmap(df,analysis_cols,metric_key, output_file=f"{output_dir}/{metric_key}/nan_categorical_heatmap.png")
        df = df[df[metric_key].notna()].copy()
        print(f"DataFrame shape after dropping NaN min_metric: {df.shape}")

        if '_SSWIG' in dataset_name:
            df_baseline = df_baseline[df_baseline['dataset'] == dataset_name.strip('_SSWIG')].copy()
        else:
            df_baseline = df_baseline[df_baseline['dataset'] == dataset_name].copy()
        df_baseline[metric_key] = pd.to_numeric(df_baseline[metric_key], errors='coerce')
        reference_metric = df_baseline[metric_key].min(skipna=True)

        df = calculateScore(reference_metric,df,metric_key)
        df=df[~((df['block_size'] == 16) & (df['scale_type'] == 'E5M3'))]
        pure_fp4 = df.loc[df[df['complexity_points']==0].groupby('scale_type')[metric_key].idxmin()]
        dfs_by_scale = bang_for_the_buck_per_scale(df)
        top_dfs_by_scale = top_per_scale(df, metric_key)
        
        # --- Generate Complexity vs Score Plot ---
        all_pos_scores = pd.concat([
            dfs_by_scale.get(scale, {}).get('positive_scores', pd.DataFrame())
            for scale in df['scale_type'].unique()
        ])
        all_neg_scores = pd.concat([
            dfs_by_scale.get(scale, {}).get('negative_scores', pd.DataFrame())
            for scale in df['scale_type'].unique()
        ])
        all_best_loss = pd.concat([
            top_dfs_by_scale.get(scale, pd.DataFrame())
            for scale in df['scale_type'].unique()
        ])
        
        plot_complexity_vs_score(
            df=df,
            df_pos=all_pos_scores,
            df_neg=all_neg_scores,
            df_loss=all_best_loss,
            pure_fp4=pure_fp4,
            dataset_name=dataset_name,
            output_dir= f'{fold_name}_plots'
        )

        top_bang = df.sort_values( by=['score', 'complexity_points'],ascending=[False, True]).head(1)

        if df_baseline is None or df_baseline.empty:
            print("Error: No valid baseline could be established. Skipping baseline-dependent analysis.")
        else:
            filtered_df = filter_comparable_to_baseline(df, df_baseline,metric_key)
            filtered_df = filtered_df.sort_values(metric_key, ascending=True)
            print(f"[{results_dir[0]}] Filtered comparable configs found: {len(filtered_df)}")
            df.sort_values(metric_key, ascending=True).head(5).to_csv(os.path.join(f"{output_dir}/{metric_key}", "top_5_configs.csv"), index=False)
            
            comparison_df_rows = []
            
            # Add baselines
            df_baseline['Source'] = 'Baseline'
            comparison_df_rows.append(df_baseline)
            
            # Add pure FP4
            pure_fp4_copy = pure_fp4.copy()
            pure_fp4_copy['Source'] = 'Pure FP4'
            comparison_df_rows.append(pure_fp4_copy)
            
            # Add best scores and losses
            all_pos_scores['Source'] = r'Best Score (Pos)'
            all_neg_scores['Source'] = r'Best Score (Neg)'
            all_best_loss['Source'] = 'Best loss'
            
            comparison_df_rows.extend([all_pos_scores, all_neg_scores, all_best_loss])
            
            comparison_df = pd.concat(comparison_df_rows, ignore_index=True)
            
            # Rename sources for clarity in the table
            source_rename_map = {
                'Best loss': lambda r: f"Best loss {r['scale_type']}",
            }
            for source, func in source_rename_map.items():
                mask = comparison_df['Source'] == source
                comparison_df.loc[mask, 'Source'] = comparison_df[mask].apply(func, axis=1)

            comparison_df = comparison_df[['dataset','Source','val_metrics_min','training_metrics_min',
'scale_type', 'block_size' ,'smooth',  'stepGradient', 'use_hadamard', 'qGradient', 'SR', 'optimiser','loss_scaling','roundMode','use_tensor_scaling','tensor_scaling_grad_est','nan_handling_mode','complexity_points','score'
]]
            table_cols = ['dataset','Source', 'val_metrics_min','training_metrics_min'] + [col for col in HYPARAM_COLS if col in comparison_df.columns]
            
            caption = f"Top configurations for {dataset_name.replace('_', ' ')} compared to the baseline."
            label = f"tab:{dataset_name}_top_comparison"
            output_file = os.path.join(f"{output_dir}/{metric_key}", "top_comparison_table.tex")
            
            generate_latex_table(comparison_df, table_cols, caption, label, output_file)

            best_e4m3_dict = pure_fp4[pure_fp4['scale_type']=='E4M3'].sort_values(metric_key, ascending=True).head(1).squeeze().to_dict()
            best_e8m0_dict = pure_fp4[pure_fp4['scale_type']=='E8M0'].sort_values(metric_key, ascending=True).head(1).squeeze().to_dict()
            best_e5m3_dict = pure_fp4[pure_fp4['scale_type']=='E5M3'].sort_values(metric_key, ascending=True).head(1).squeeze().to_dict()
            best_e8m3_dict = pure_fp4[pure_fp4['scale_type']=='E8M3'].sort_values(metric_key, ascending=True).head(1).squeeze().to_dict()

            top_bang_dict = top_bang.squeeze().to_dict()

            all_data = {dataset_name: {'data_loss': df.sort_values(metric_key, ascending=True).head(1).squeeze().to_dict(),
            'data_score':top_bang_dict,
            'baseline': df_baseline.sort_values(metric_key, ascending=True).head(1).squeeze().to_dict(), 
            'fp4_e4m3_scale': best_e4m3_dict,
            'fp4_e8m0_scale':best_e8m0_dict,
            'fp4_e5m3_scale': best_e5m3_dict,
            'fp4_e8m3_scale': best_e8m3_dict,
            
            }}
            plot_validation_and_loss_curves(all_data,source_list,save_folder = f'{fold_name}_plots')

            if not filtered_df.empty and filtered_df.shape[0]>5:    
                model, coef_df = analyze_hparam_effects_lasso_subsample(os.path.join(f"{output_dir}/{metric_key}", "lasso_filtered"), filtered_df, metric_key, analysis_cols,n_subsamples=100, frac=0.7, alpha=1e-3)
                coef_df.to_csv(os.path.join(f"{output_dir}/{metric_key}", "lasso_filtered_param.csv"))
                
                ols_model = analyze_hparam_effects(os.path.join(f"{output_dir}/{metric_key}", "OLS_filtered"), filtered_df, metric_key, analysis_cols)
                with open(os.path.join(f"{output_dir}/{metric_key}", "model_summary_OLS_filtered.txt"), "w") as f:
                    f.write(ols_model.summary().as_text())
    # Return the last generated dataframe for potential multi-config summary tables
    return comparison_df, table_cols

if __name__ == "__main__":
    # configs = [
    #             {
    #         "results_dir": ["24_single_parallelisation_exact_fixed/_results","24_stack_extra_results"],
    #         "jobs_dir": ["24_single_parallelisation_exact_fixed","24_stack_extra"],
    #         "dataset_name": "CIFAR10",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },
    #     {
    #         "results_dir": ["24_single_parallelisation_exact_fixed/_results","24_stack_extra_results"],
    #         "jobs_dir": ["24_single_parallelisation_exact_fixed","24_stack_extra"],
    #         "dataset_name": "MNIST",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },
    #             {
    #         "results_dir": ["small_diffusion_fixed_results"],
    #         "jobs_dir": ["small_diffusion_fixed"],
    #         "dataset_name": "small_diffusion",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },
    #   {
    #         "results_dir": ['llama_9M_extra_results'],
    #         "jobs_dir": ['llama_9M_extra'],
    #         "dataset_name": "llama_9M",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },


    # ]

    # main_dfs = []
    # for config in tqdm(configs, desc="Processing All Datasets"):
    #     comp_df, table_cols= run_analysis_for_config(**config , cols=HYPARAM_COLS)
    #     main_dfs.append(comp_df)

    # main_concat = pd.concat(main_dfs)
    # caption = "Additional experimental results"
    # label = f"tab:addtional_results"
    # generate_latex_table(main_concat, table_cols, caption, label, "addtional_results.tex")
   

    # configs = [
    #     {
    #         "results_dir": ["llama_60M_exact_fixed_results","llama_60M_E4M3_SR_scale_results","llama_60M_E8M0_SR_scale_results","llama_60M_extra_results"],
    #         "jobs_dir": ["llama_60M_exact_fixed","llama_60M_E4M3_SR_scale","llama_60M_E8M0_SR_scale","llama_60M_extra"],
    #         "dataset_name": "llama_60M",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },
    #     {
    #         "results_dir": ["llama_350M_E8M0_results","llama_350M_E8M0_SR_scale_results","llama_350M_E8M0_extra_results","llama_350M_E4M3_results",'llama_350M_pure_results'],
    #         "jobs_dir": ["llama_350M_E8M0","llama_350M_E8M0_SR_scale","llama_350M_E8M0_extra","llama_350M_E4M3",'llama_350M_pure'],
    #         "dataset_name": "llama_350M",
    #         "baseline_dir": ["baseline_llama_350M_results", "baseline_stable_spam_llama_350M_results"]
    #     },

    #             {
    #         "results_dir": ["llama_1B_E8M0_results","llama_1B_E4M3_results","llama_1B_pure_results"],
    #         "jobs_dir": ["llama_1B_E8M0","llama_1B_E4M3","llama_1B_pure"],
    #         "dataset_name": "llama_1B",
    #         "baseline_dir": ["baseline_llama_1B_results","baseline_stable_spam_llama_1B_results"]
    #     },
    # ]

    # main_dfs = []
    # for config in tqdm(configs, desc="Processing LLM Datasets"):
    #     comp_df, table_cols= run_analysis_for_config(**config , cols=HYPARAM_COLS)
    #     main_dfs.append(comp_df)

    # main_concat = pd.concat(main_dfs)
    # caption = "LLM results"
    # label = f"tab:llm_results"
    # generate_latex_table(main_concat, table_cols, caption, label, "llm_latex_table.tex")



    # configs = [
       

    #         {   
    #         "results_dir": ["24_single_parallelisation_exact_fixed/_results","24_stack_extra_results"],
    #         "jobs_dir": ["24_single_parallelisation_exact_fixed","24_stack_extra"],
    #         "dataset_name": "gaussian_reg",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },
    #     {
    #         "results_dir": ["big_diffusion_fixed_results"],
    #         "jobs_dir": ["big_diffusion_fixed"],
    #         "dataset_name": "big_diffusion",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },
    #             {
    #         "results_dir": ["IMAGENET100_exact_fixed_results","IMAGENET100_extra_results"],
    #         "jobs_dir": ["IMAGENET100_exact_fixed","IMAGENET100_extra"],
    #         "dataset_name": "IMAGENET100",
    #         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    #     },
    # ]

    # main_dfs = []
    # for config in tqdm(configs, desc="Processing Vision Datasets"):
    #     comp_df, table_cols= run_analysis_for_config(**config , cols=HYPARAM_COLS)
    #     main_dfs.append(comp_df)

    # main_concat = pd.concat(main_dfs)
    # caption = "Experimental Results"
    # label = f"tab:diffusion_results"
    # generate_latex_table(main_concat, table_cols, caption, label, "diffusion_latex_table.tex")


    configs = [
    {
        "results_dir": ["24_stack_e5m3_results","e5m3_pure_single_gpu_results"],
        "jobs_dir": ["24_stack_e5m3","e5m3_pure_single_gpu"],
        "dataset_name": "CIFAR10",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ["24_stack_e5m3_results","e5m3_pure_single_gpu_results"],
        "jobs_dir": ["24_stack_e5m3","e5m3_pure_single_gpu"],
        "dataset_name": "MNIST",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ["24_stack_e5m3_results","e5m3_pure_single_gpu_results"],
        "jobs_dir": ["24_stack_e5m3","e5m3_pure_single_gpu"],
        "dataset_name": "gaussian_reg",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ["IMAGENET100_e5m3_results","e5m3_pure_single_gpu_results"],
        "jobs_dir": ["IMAGENET100_e5m3","e5m3_pure_single_gpu"],
        "dataset_name": "IMAGENET100",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ["e5m3_diffusion_results","e5m3_pure_single_gpu_results"],
        "jobs_dir": ["e5m3_diffusion","e5m3_pure_single_gpu"],
        "dataset_name": "small_diffusion",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ["e5m3_diffusion_results","e5m3_pure_single_gpu_results"],
        "jobs_dir": ["e5m3_diffusion","e5m3_pure_single_gpu"],
        "dataset_name": "big_diffusion",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ['llama_9M_e5m3_results',"e5m3_pure_single_gpu_results"],
        "jobs_dir": ['llama_9M_e5m3',"e5m3_pure_single_gpu"],
        "dataset_name": "llama_9M",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ['llama_60M_e5m3_results',"e5m3_pure_single_gpu_results"],
        "jobs_dir": ['llama_60M_e5m3',"e5m3_pure_single_gpu"],
        "dataset_name": "llama_60M",
        "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
    },
    {
        "results_dir": ['llama_350M_e5m3_results',"e5m3_pure_llama_350M_results"],
        "jobs_dir": ['llama_350M_e5m3',"e5m3_pure_llama_350M"],
        "dataset_name": "llama_350M",
        "baseline_dir": ["baseline_llama_350M_results", "baseline_stable_spam_llama_350M_results"]
    },
    {
        "results_dir": ['llama_1B_e5m3_results','e5m3_pure_llama_1B_results'],
        "jobs_dir": ['llama_1B_e5m3','e5m3_pure_llama_1B'],
        "dataset_name": "llama_1B",
        "baseline_dir": ["baseline_llama_1B_results","baseline_stable_spam_llama_1B_results"]
    },
]

    dfs = []
    for config in configs:
        comp_df, table_cols = run_analysis_for_config(**config, cols=HYPARAM_COLS)
        dfs.append(comp_df)

    concat = pd.concat(dfs)
    caption = "UE5M3 results"
    label = f"tab:ablation"
    generate_latex_table(concat, table_cols, caption, label, "ue5m3_latex_table.tex")


# configs = [
#     {
#         "results_dir": ['llama_9M_SSWIG_results'],
#         "jobs_dir": ['llama_9M_SSWIG'],
#         "dataset_name": "llama_9M_SSWIG",
#         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
#     },
#     {
#         "results_dir": ['llama_60M_SSWIG_results'],
#         "jobs_dir": ['llama_60M_SSWIG'],
#         "dataset_name": "llama_60M_SSWIG",
#         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
#     },
#     {
#         "results_dir": ['llama_350M_SSWIG_results'],
#         "jobs_dir": ['llama_350M_SSWIG'],
#         "dataset_name": "llama_350M_SSWIG",
#         "baseline_dir": ["baseline_llama_350M_results", "baseline_stable_spam_llama_350M_results"]
#     },
#     {
#         "results_dir": ['llama_1B_SSWIG_results'],
#         "jobs_dir": ['llama_1B_SSWIG'],
#         "dataset_name": "llama_1B_SSWIG",
#         "baseline_dir": ["baseline_llama_1B_results","baseline_stable_spam_llama_1B_results"]
#     },
# ]

# dfs = []
# for config in configs:
#     comp_df, table_cols = run_analysis_for_config(**config, cols=HYPARAM_COLS)
#     dfs.append(comp_df)

# concat = pd.concat(dfs)
# caption = "SWIG results"
# label = f"tab:ablation_swig"
# generate_latex_table(concat, table_cols, caption, label, "swig_latex_table.tex")


# configs = [
#     {
#         "results_dir": ['e8m3_ablations_results'],
#         "jobs_dir": ['e8m3_ablations'],
#         "dataset_name": "llama_9M",
#         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
#     },
#     {
#         "results_dir": ['e8m3_ablations_results'],
#         "jobs_dir": ['e8m3_ablations'],
#         "dataset_name": "llama_60M",
#         "baseline_dir": ["single_gpu_baseline_results", "single_gpu_baseline_stable_spam/_results"]
#     }
# ]

# dfs = []
# for config in configs:
#     comp_df, table_cols = run_analysis_for_config(**config, cols=HYPARAM_COLS)
#     dfs.append(comp_df)

# concat = pd.concat(dfs)
# caption = "E8M3 ablation results"
# label = f"tab:ablation_e8m3"
# generate_latex_table(concat, table_cols, caption, label, "e8m3_latex_table.tex")
