import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchao.prototype.mx_formats.config import MXLinearConfig,MXGemmKernelChoice
from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_linear, mx_mm
from low_bits_training.quantization.dimensionQuantisationClass import (
    MXLinearDimConfig,
    MXFPscalingModule,
    new_to_mx
)
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP4,
)
import itertools
import json 
import seaborn as sns
import os
import numpy as np
import random
class ErrorMetrics():
    def __init__(self, out_ref, out):
        epsilon = 1e-6  # Small value to prevent division by zero
        self.relative_error = ((out - out_ref).abs() / (out_ref.abs() + epsilon)).to(torch.float32)
        self.absolute_error = (out - out_ref).abs().to(torch.float32)
        self.squared_error = ((out - out_ref) ** 2).to(torch.float32)
        
        # Relative error metrics
        self.mean_rel = self.relative_error.mean().item()
        self.median_rel = self.relative_error.median().item()
        self.max_rel = self.relative_error.max().item()
        self.std_rel = self.relative_error.std().item()
        self.p99_rel = self.relative_error.quantile(0.99).item()
        self.p999_rel = self.relative_error.quantile(0.999).item()
        self.iqr_rel = (self.relative_error.quantile(0.75) - self.relative_error.quantile(0.25)).item()
        
        # Absolute error metrics (MAE)
        self.mean_abs = self.absolute_error.mean().item()
        self.median_abs = self.absolute_error.median().item()
        self.max_abs = self.absolute_error.max().item()
        self.std_abs = self.absolute_error.std().item()
        self.p99_abs = self.absolute_error.quantile(0.99).item()
        self.p999_abs = self.absolute_error.quantile(0.999).item()
        self.iqr_abs = (self.absolute_error.quantile(0.75) - self.absolute_error.quantile(0.25)).item()
        
        # RMS Error
        self.rms_error = torch.sqrt(self.squared_error.mean()).item()
        
        # Identify top 10 largest relative errors
        top_rel_indices = torch.topk(self.relative_error.flatten(), 10).indices
        self.worst_offenders_rel = {
            "out": out.flatten()[top_rel_indices].tolist(),
            "out_ref": out_ref.flatten()[top_rel_indices].tolist(),
            "errors": self.relative_error.flatten()[top_rel_indices].tolist()
        }
        
        # Identify top 10 largest absolute errors
        top_abs_indices = torch.topk(self.absolute_error.flatten(), 10).indices
        self.worst_offenders_abs = {
            "out": out.flatten()[top_abs_indices].tolist(),
            "out_ref": out_ref.flatten()[top_abs_indices].tolist(),
            "errors": self.absolute_error.flatten()[top_abs_indices].tolist()
        }

    def get_stats(self):
        return [
            self.mean_rel, self.median_rel, self.max_rel, self.std_rel, self.p99_rel, self.p999_rel, self.iqr_rel,
            self.mean_abs, self.median_abs, self.max_abs, self.std_abs, self.p99_abs, self.p999_abs, self.iqr_abs,
            self.rms_error, json.dumps(self.worst_offenders_rel), json.dumps(self.worst_offenders_abs)
        ]


df_results = pd.DataFrame(columns=[
    "block_size", "exmy","max_type",
    "mean_rel_error", "median_rel_error", "max_rel_error", "std_rel_error", "p99_rel_error", "p999_rel_error", "iqr_rel_error",
    "mean_abs_error", "median_abs_error", "max_abs_error", "std_abs_error", "p99_abs_error", "p999_abs_error", "iqr_abs_error",
    "rms_error", "worst_offenders_rel", "worst_offenders_abs"
])


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.linear = torch.nn.Linear(1024, 512, dtype=torch.bfloat16)
        self.linear = torch.nn.Linear(1024, 512, dtype=torch.bfloat16)
        torch.nn.init.constant_(self.linear.weight, 1)

        # Set biases to zero (optional)
        torch.nn.init.constant_(self.linear.bias, 1)

    def forward(self, x):
        return self.linear(x)

def plot_metrics(df,folder='.'):
    os.makedirs(folder,exist_ok=True)
    metrics = [
        "mean_rel_error", "median_rel_error", "max_rel_error", "std_rel_error", "p99_rel_error", "p999_rel_error", "iqr_rel_error",
        "mean_abs_error", "median_abs_error", "max_abs_error", "std_abs_error", "p99_abs_error", "p999_abs_error", "iqr_abs_error",
        "rms_error"
    ]
    x_name = df.columns[0]
    exmy_values = df["exmy"].unique()
    color_palette = sns.color_palette("tab10", len(exmy_values))  # Use Set2 or any other color palette of your choice
    exmy_color_map = dict(zip(exmy_values, color_palette))
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=x_name, y=metric, hue="exmy", marker="o")
        
        # Highlight special case BLOCK_DIM = 1
        special_case = df[df[x_name] == 1]
        if not special_case.empty:
            for exmy, color in exmy_color_map.items():
                # Get the values for the current exmy group
                special_case_metric_values = special_case[special_case["exmy"] == exmy][metric].values
                for val in special_case_metric_values:
                    plt.axhline(y=val, linestyle='dashed', color=color, label=f"BLOCK_DIM=1 ({metric}, {exmy})")
        plt.xscale("log", base=2)
        plt.title(f"{metric} vs {x_name}")
        plt.xlabel(x_name)
        plt.ylabel(metric)
        plt.legend(title="exmy")
        plt.grid()
        plt.legend(title="exmy", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout()  # Adjust the plot to prevent clipping
        plt.savefig(f"{folder}/{metric}_vs_block_size.png")

DTYPE_BASE = torch.bfloat16
BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
SCALE_TYPES = ['E5M2', 'E4M3','E8M7','E8M0']
MAX_APPROX = ['STE','softsoftmax']#softsoftmax "hardsoftmax", 'STE','absmax'
STEP_GRADIENTS = ['STE'] #[None, 'baseline', 'spline', 'sigmoid'] #This apparently sucks ass, figure out why and fix it! Cut the bullshit!


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    w  = torch.randn((1024, 1024), dtype=torch.bfloat16).to(device) 
    input_data = torch.randn((2000,1024)).to(torch.bfloat16).to(device) 
    output_ref = input_data@w
    for block_size, scale_type, ma, step_gradient in itertools.product(BLOCK_SIZES, SCALE_TYPES, MAX_APPROX, STEP_GRADIENTS):
        
        config = MXLinearDimConfig(
                            block_dim =block_size if block_size==1 else None, 
                            block_size=block_size if block_size!=1 else None, 
                            elem_dtype=DTYPE_FP4, 
                            scale_type=scale_type,
                            use_approx={
                                            'smooth': ma,
                                                'alpha': 1.0 , 
                                                'stepGradient': step_gradient, 
                                                'temperature': 0.025, 
                                                'k':5, 
                                                'dtype':DTYPE_BASE,
                                                'lb':1,
                                                'ub':1.5,
                                                },
                            dtype = DTYPE_BASE
                            )
        scalingModule = MXFPscalingModule(
            elem_dtype=config.elem_dtype,
            block_size=config.block_size,
            block_dim=config.block_dim,
            scale_type=config.scale_type,
            use_approx=config.use_approx
            )

        MX_w,_,_ = new_to_mx(tensor = w,
                        scalingModule= scalingModule
                        )
        MX_input,_,_ = new_to_mx(tensor = input_data,
                        scalingModule= scalingModule
                        )
        output_2 = MX_input@MX_w
        err_metrics = ErrorMetrics(output_ref, output_2)
        df_results.loc[len(df_results)] = [block_size, scale_type,ma] + err_metrics.get_stats()
    df_results.to_csv("matmul_res.csv", index=False)
    plot_metrics(df_results[df_results['max_type']=='STE'],'matmul_res_STE')
    plot_metrics(df_results[df_results['max_type']=='softsoftmax'],'matmul_res_soft')

    BLOCK_SIZES = [16, 32, 64]

    for block_size, scale_type, ma, step_gradient in itertools.product(BLOCK_SIZES, SCALE_TYPES, MAX_APPROX, STEP_GRADIENTS):
        for a in [1e-4,0.001,0.01,0.05,0.1,0.25,0.5,1,1.5,2,4.0,5.0,10,25]:
            config = MXLinearDimConfig(
                                block_dim =block_size if block_size==1 else None, 
                                block_size=block_size if block_size!=1 else None, 
                                elem_dtype=DTYPE_FP4, 
                                scale_type=scale_type,
                                use_approx={
                                                'smooth': ma,
                                                    'alpha': 1.0 , 
                                                    'stepGradient': step_gradient, 
                                                    'temperature': 0.025, 
                                                    'k':5, 
                                                    'dtype':DTYPE_BASE,
                                                    'lb':1,
                                                    'ub':1.5,
                                                    },
                                dtype = DTYPE_BASE
                                )
            scalingModule = MXFPscalingModule(
                elem_dtype=config.elem_dtype,
                block_size=config.block_size,
                block_dim=config.block_dim,
                scale_type=config.scale_type,
                use_approx=config.use_approx
                )
            w  = torch.randn((1024, 1024), dtype=torch.bfloat16).to(device) 
            input_data = torch.randn((2000,1024)).to(torch.bfloat16).to(device) * a
            output_ref = input_data@w

            MX_w,_,_ = new_to_mx(tensor = w,
                            scalingModule= scalingModule
                            )
            MX_input,_,_ = new_to_mx(tensor = input_data,
                            scalingModule= scalingModule
                            )
            output_2 = MX_input@MX_w
            err_metrics = ErrorMetrics(output_ref, output_2)
            df_results.loc[len(df_results)] = [a, scale_type,ma] + err_metrics.get_stats()
        df_results.to_csv(f"matmul_res_bs={block_size}.csv", index=False)
        plot_metrics(df_results[df_results['max_type']=='STE'],f'matmul_res_STE_bs={block_size}')
        plot_metrics(df_results[df_results['max_type']=='softsoftmax'],f'matmul_res_soft_bs={block_size}')


