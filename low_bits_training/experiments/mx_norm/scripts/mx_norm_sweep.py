# Copyright (c) 2025 Graphcore Ltd. All rights reserved.

"""
Use this file as a running log to keep track of what we have run
"""

import inspect
import os
import subprocess
import sys

from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict

HOME = Path(__file__).parents[4]

CommandList = List[Tuple[str, Union[str, int, float, bool]]]


def run_command(
    command: str,
    command_list: CommandList,
    env_vars: Optional[Dict[str, str]] = None,
    print_only: bool = False,
) -> None:
    """
    Execute the command with the given parameters and environment variables.

    Args:
        command: Base command to execute
        command_list: List of parameter-value tuples to append to command
        env_vars: Dictionary of environment variables to prepend to command
        print_only: If True, only print the command without executing it

    Raises:
        subprocess.CalledProcessError: If command execution fails
    """
    cmd_parts = []

    # Add environment variables if provided - for display purposes
    env_display = ""
    if env_vars:
        env_display = " ".join(
            [f"{var_name}={var_value}" for var_name, var_value in env_vars.items()]
        )
        cmd_parts.append(env_display)

    # Add the base command
    cmd_parts.append(command)

    # Add parameters
    for param, value in command_list:
        if isinstance(value, bool):
            if value:
                cmd_parts.append(f"--{param}")
        else:
            if isinstance(value, str) and " " in value:
                cmd_parts.extend([f"--{param}", f'"{value}"'])
            else:
                cmd_parts.extend([f"--{param}", str(value)])

    cmd_str = " ".join(cmd_parts)

    # Skip execution if print_only is True
    if print_only:
        print(f"{cmd_str}")
        return

    try:
        # Build the command without env vars for subprocess
        cmd_without_env = cmd_str
        if env_display:
            cmd_without_env = cmd_str.replace(env_display + " ", "")

        # Get current environment and update with our custom variables
        current_env = os.environ.copy()
        if env_vars:
            current_env.update(env_vars)

        subprocess.run(cmd_without_env, shell=True, check=True, env=current_env)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {cmd_str}")
        print(f"Error: {e}")


def get_func_name():
    """Get the name of the function that called this function."""
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    return caller_frame.f_code.co_name


def submit_job(
    options,
    priority=None,
    num_nodes=1,
    cluster=False,
    dry_run=False,
    name=None,
    config_file=None,
):
    if cluster:
        env_vars = dict(
            NUM_NODES=str(num_nodes), CPUS_PER_GPU="20", CONFIG_FILE=config_file
        )
    else:
        assert num_nodes == 1
        env_vars = dict(SBATCH_ARGS="--nodes=1 --cpus-per-gpu=13")
    if priority:
        env_vars["PRIORITY"] = f"{priority}-priority"
    if name:
        env_vars["WANDB_NAME"] = name
    run_command(
        f"{HOME}/submit.sh",
        [(k, v) for k, v in options.items()],
        env_vars=env_vars,
        print_only=dry_run,
    )


def bf16_baseline_250m_lr_sweep(dry_run):
    # LR sweep for depth=4, width=2048 (250M params), RMSNorm, BF16 matmuls, 5B tokens

    # CM: should be 2048**2 * 4 layers * 12 weight matrices = 200M params
    options = {
        "model.n_layers": 4,
        "model.dim": 2048,
        # "training.batch_size": 4, # default batch size in base config is 4
        "training.steps": 40000,  # 40000 steps * 8 GPUs * 4 samples per GPU * 4096 tokens per sample ~= 5B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "training.dataset": "slimpajama-6b",
    }
    for lr in [2**k for k in range(-14, -6)]:  # lr = [2**-14, ..., 2**-7]
        options["optimizer.lr"] = lr
        submit_job(options, dry_run=dry_run)


def bf16_baseline_250m_lr_sweep_bs8(dry_run):
    # LR sweep for depth=4, width=2048 (250M params), RMSNorm, BF16 matmuls, 5B tokens
    # increased local batch size to 8 as only around 25% memory used

    # CM: should be 2048**2 * 4 layers * 12 weight matrices = 200M params
    options = {
        "model.n_layers": 4,
        "model.dim": 2048,
        "training.batch_size": 8,
        "training.steps": 20000,  # 20000 steps * 8 GPUs * 8 samples per GPU * 4096 tokens per sample ~= 5B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "training.dataset": "slimpajama-6b",
    }
    for lr in [2**k for k in range(-14, -6)]:  # lr = [2**-14, ..., 2**-7]
        options["optimizer.lr"] = lr
        submit_job(options, dry_run=dry_run)


def bf16mm_mxnorm_250m_lr_sweep(dry_run):
    # LR sweep for depth=4, width=2048 (250M params), MXnorm, BF16 matmuls, 5B tokens
    # CM: should be 2048**2 * 4 layers * 12 weight matrices = 200M params

    # MXnorm variants tested: fixed iteration, lookup table, mean absmax only, scaled mean absmax
    options = {
        "model.n_layers": 4,
        "model.dim": 2048,
        # "training.batch_size": 4,
        "training.steps": 40000,  # 40000 steps * 8 GPUs * 4 samples per GPU * 4096 tokens per sample ~= 5B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "training.dataset": "slimpajama-6b",
        "model.converters": "mx_rmsnorm",
    }
    for method in [
        "fixed_point_iter_with_lut",
        "lut_and_lerp",
        "linear_scale",
        "mean_absmax",
    ]:
        for lr in [2**k for k in range(-12, -2)]:  # lr = [2**-12, ..., 2**-3]
            options["mx_rmsnorm.sigma_absmax_mapping_fn"] = method
            options["optimizer.lr"] = lr
            if method == "lut_and_lerp":
                options["mx_rmsnorm.n_lut_entries"] = 256
            submit_job(options, dry_run=dry_run)


def bf16_baseline_250m_lr_sweep_higher_lrs(dry_run):
    # LR sweep for depth=4, width=2048 (250M params), RMSNorm, BF16 matmuls, 5B tokens
    # CM: should be 2048**2 * 4 layers * 12 weight matrices = 200M params
    options = {
        "model.n_layers": 4,
        "model.dim": 2048,
        # "training.batch_size": 4, # default batch size in base config is 4
        "training.steps": 40000,  # 40000 steps * 8 GPUs * 4 samples per GPU * 4096 tokens per sample ~= 5B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "training.dataset": "slimpajama-6b",
    }
    for lr in [2**k for k in range(-6, -2)]:  # lr = [2**-6 ..., 2**-3]
        options["optimizer.lr"] = lr
        submit_job(options, dry_run=dry_run)


def bf16_baseline_1b_lr_sweep(dry_run):
    # LR sweep for depth=16, width=2048 (1B params), RMSNorm, BF16 matmuls, 5B tokens
    # CM: should be 2048**2 * 16 layers * 12 weight matrices = 800M params
    common_options = {
        # "model.n_layers": 16,
        "model.dim": 2048,
        "training.batch_size": 8,  # improves MFU by 5%
        "training.steps": 20000,  # 20000 steps * 32 GPUs * 8 samples per GPU * 4096 tokens per sample ~= 21B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
    }
    for lr in [2**k for k in range(-14, -2)]:  # lr = [2**-14, ..., 2**-3]
        options = {}
        options["optimizer.lr"] = lr
        submit_job(common_options | options, num_nodes=4, cluster=True, dry_run=dry_run)


def bf16mm_mxnorm_1b_lr_sweep(dry_run):
    # LR sweep for depth=16, width=2048 (1B params), MXnorm, BF16 matmuls, 5B tokens
    # CM: should be 2048**2 * 16 layers * 12 weight matrices = 200M params

    # MXnorm variants tested: fixed iteration, lookup table, mean absmax only, scaled mean absmax
    common_options = {
        "model.n_layers": 16,
        "model.dim": 2048,
        "training.batch_size": 8,  # improves MFU by 5%
        "training.steps": 20000,  # 20000 steps * 32 GPUs * 8 samples per GPU * 4096 tokens per sample ~= 21B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "model.converters": "mx_rmsnorm",
    }
    for method in [
        "fixed_point_iter_with_lut",
        "lut_and_lerp",
        "linear_scale",
        "mean_absmax",
    ]:
        options = {}
        for lr in [2**k for k in range(-14, -2)]:  # lr = [2**-14, ..., 2**-3]
            options["mx_rmsnorm.sigma_absmax_mapping_fn"] = method
            if method in ["fixed_point_iter_with_lut", "lut_and_lerp"]:
                options["mx_rmsnorm.n_lut_entries"] = 256
            options["optimizer.lr"] = lr
            submit_job(
                common_options | options, num_nodes=4, cluster=True, dry_run=dry_run
            )


def bf16_baseline_250m_lr_sweep_upcast_norm(dry_run):
    # LR sweep for depth=4, width=2048 (250M params), RMSNorm, BF16 matmuls, 5B tokens
    # Uses RMSNorm with upcasted inputs
    options = {
        "model.n_layers": 4,
        "model.dim": 2048,
        "training.batch_size": 4,  # default batch size in base config is 4
        "training.steps": 40000,  # 40000 steps * 8 GPUs * 4 samples per GPU * 4096 tokens per sample ~= 5B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "training.dataset": "slimpajama-6b",
        "model.converters": "mx_rmsnorm",
        "mx_rmsnorm.sigma_absmax_mapping_fn": "rms",
    }
    for lr in [2**k for k in range(-14, -2)]:  # lr = [2**-14 ..., 2**-3]
        options["optimizer.lr"] = lr
        submit_job(options, dry_run=dry_run)


def bf16_baseline_1b_lr_sweep_upcast_norm(dry_run):
    # LR sweep for depth=16, width=2048 (~1B params), RMSNorm, BF16 matmuls, 20B tokens
    # Uses RMSNorm with upcasted inputs
    common_options = {
        "model.n_layers": 16,
        "model.dim": 2048,
        "training.batch_size": 8,  # default batch size in base config is 4
        "training.steps": 20000,  # 20000 steps * 8 GPUs * 8 samples per GPU * 4096 tokens per sample ~= 5B tokens
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "model.converters": "mx_rmsnorm",
        "mx_rmsnorm.sigma_absmax_mapping_fn": "rms",
    }
    for lr in [2**k for k in range(-14, -2)]:  # lr = [2**-14 ..., 2**-3]
        options = {}
        options["optimizer.lr"] = lr
        submit_job(common_options | options, num_nodes=4, cluster=True, dry_run=dry_run)


def mxmm_lr_sweep_1b_250m(dry_run):
    common_options = {
        "model.dim": 2048,
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "model.converters": "mx_rmsnorm,mxfp",
        "mx_rmsnorm.scale_rounding_fn": "rceil",
        "mxfp.scale_rounding_fn": "rceil",
    }
    lrs = [2**k for k in range(-14, -2)]
    mxnorm_methods = ["lut_and_lerp", "mean_absmax"]

    # paired options across model scales
    depths = [4, 16]
    batch_sizes = [4, 8]
    num_nodes = [1, 4]
    training_steps = [40000, 20000]
    model_scale_paired_options = zip(depths, batch_sizes, num_nodes, training_steps)

    for L, B, N, T in model_scale_paired_options:
        for method in mxnorm_methods:
            for lr in lrs:
                options = {
                    "model.n_layers": L,
                    "training.batch_size": B,
                    "training.steps": T,
                    "mx_rmsnorm.sigma_absmax_mapping_fn": method,
                    "optimizer.lr": lr,
                }
                submit_job(
                    common_options | options, num_nodes=N, cluster=True, dry_run=dry_run
                )


def mxmm_v2_lr_sweep_1b_250m(dry_run):
    common_options = {
        "model.dim": 2048,
        "wandb.project": "low-bits-training-mxnorm",
        "job.config_file": "train_configs/llama3_1b.toml",
        "model.converters": "mx_norm_linear",
        "mxfp.scale_rounding_fn": "cublas_ceil",
        "mxfp.block_size": 32,
        "mxfp.activation_dtype": "e4m3",
        "mxfp.weight_dtype": "e4m3",
        "mxfp.gradient_dtype": "e5m2",
    }
    lrs = [2**k for k in range(-14, -2)]
    mxnorm_methods = ["pre", "post"]

    # paired options across model scales
    depths = [4, 16]
    batch_sizes = [4, 8]
    num_nodes = [1, 4]
    training_steps = [40000, 20000]
    model_scale_paired_options = zip(depths, batch_sizes, num_nodes, training_steps)

    for L, B, N, T in model_scale_paired_options:
        for method in mxnorm_methods:
            for lr in lrs:
                options = {
                    "model.n_layers": L,
                    "training.batch_size": B,
                    "training.steps": T,
                    "mx_norm_linear.norm_mode": method,
                    "optimizer.lr": lr,
                }
                if method == "post":
                    options["mx_norm_linear.n_lut_entries"] = 256
                if L == 16:
                    submit_job(
                        common_options | options,
                        priority="low",
                        num_nodes=N,
                        cluster=True,
                        dry_run=dry_run,
                    )


def unstable_250m(dry_run):
    common_options = {
        "model.dim": 2048,
        "model.n_layers": 4,
        "training.local_batch_size": 16,
        "training.steps": 10000,
        "mxfp.scale_rounding_fn": "cublas_ceil",
        "mxfp.block_size": 32,
        "mxfp.activation_dtype": "e4m3",
        "mxfp.weight_dtype": "e4m3",
        "mxfp.gradient_dtype": "e5m2",
        "model.hf_assets_path": "./torchtitan_submodule/assets/hf/Llama-3.1-8B",
        "checkpoint.interval": 2000,
        "wandb.project": "low-bits-training-mxnorm-debug",
        "activation_checkpoint.mode": "full",
        "optimizer.beta2": 0.99,
    }

    mxnorm_options = {
        "model.converters": "mx_norm_linear",
        "job.experimental_modules": "mx_norm",
        "mx_norm_linear.norm_mode": "pre",
    }

    rmsnorm_options = {
        "model.converters": "mxfp",
        "job.experimental_modules": "mxfp8",
    }

    norm_options_dict = {"rmsnorm": rmsnorm_options, "mxnorm": mxnorm_options}

    config_file = "train_configs/llama3_1b.toml"
    for norm_key, norm_options in norm_options_dict.items():
        for k in range(-9, -3):
            options = {"optimizer.lr": 2**k}
            submit_job(
                common_options | norm_options | options,
                dry_run=dry_run,
                name=f"{norm_key}_{get_func_name()}_k{-k}",
                num_nodes=1,
                cluster=True,
                config_file=config_file,
            )


def unstable_slim250m(dry_run):
    common_options = {
        "model.dim": 1024,
        "model.n_heads": 16,
        "model.n_kv_heads": 4,
        "model.ffn_dim_multiplier": 4096 / 4 / 1024 * 3 / 2,
        "model.n_layers": 16,
        "mxfp.scale_rounding_fn": "cublas_ceil",
        "mxfp.block_size": 16,
        "mxfp.activation_dtype": "e4m3",
        "mxfp.weight_dtype": "e4m3",
        "mxfp.gradient_dtype": "e5m2",
        "model.hf_assets_path": "./torchtitan_submodule/assets/hf/Llama-3.1-8B",
        "checkpoint.interval": 2000,
        "wandb.project": "low-bits-training-mxnorm-debug",
        "activation_checkpoint.mode": "full",
    }

    mxnorm_options = {
        "model.converters": "mx_norm_linear",
        "job.experimental_modules": "mx_norm",
        "mx_norm_linear.norm_mode": "pre",
    }

    rmsnorm_options = {
        "model.converters": "mxfp",
        "job.experimental_modules": "mxfp8",
    }

    norm_options_dict = {"rmsnorm": rmsnorm_options, "mxnorm": mxnorm_options}

    config_file = "train_configs/llama3_1b.toml"
    for norm_key, norm_options in norm_options_dict.items():
        for bs in [4, 8, 16]:
            for beta2 in [0.95, 0.98, 0.99]:
                for k in range(-9, -6):
                    options = {
                        "optimizer.lr": 2**k,
                        "optimizer.beta2": beta2,
                        "training.local_batch_size": bs,
                        "training.steps": 160000 // bs,
                    }
                    submit_job(
                        common_options | norm_options | options,
                        dry_run=dry_run,
                        name=f"{norm_key}_{get_func_name()}_k{-k}_b{bs}_b2{int(beta2*100)}",
                        num_nodes=1,
                        cluster=True,
                        config_file=config_file,
                    )


def slim250m_003(dry_run):
    common_options = {
        "model.dim": 1024,
        "model.n_heads": 16,
        "model.n_kv_heads": 4,
        "model.ffn_dim_multiplier": 4096 / 4 / 1024 * 3 / 2,
        "model.n_layers": 16,
        "mxfp.scale_rounding_fn": "cublas_ceil",
        "mxfp.block_size": 16,
        "mxfp.activation_dtype": "e4m3",
        "mxfp.weight_dtype": "e4m3",
        "mxfp.gradient_dtype": "e5m2",
        "model.hf_assets_path": "./torchtitan_submodule/assets/hf/Llama-3.1-8B",
        "checkpoint.interval": 2000,
        "checkpoint.keep_latest_k": 2,
        "wandb.project": "low-bits-training-mxnorm-debug",
        "activation_checkpoint.mode": "full",
    }

    mxnorm_options = {
        "model.converters": "mx_norm_linear",
        "job.experimental_modules": "mx_norm",
        "mx_norm_linear.norm_mode": "pre",
    }

    rmsnorm_options = {
        "model.converters": "mxfp",
        "job.experimental_modules": "mxfp8",
    }

    norm_options_dict = {"rmsnorm": rmsnorm_options, "mxnorm": mxnorm_options}

    config_file = "train_configs/llama3_1b.toml"
    for norm_key, norm_options in norm_options_dict.items():
        for bs in [8, 16]:
            for beta2 in [0.95, 0.98]:
                for k in range(-10, -7):
                    for warmup in [500, 1000]:
                        options = {
                            "optimizer.lr": 2**k,
                            "optimizer.beta2": beta2,
                            "training.local_batch_size": bs,
                            "training.steps": 160000 // bs,
                            "lr_scheduler.warmup_steps": warmup,
                        }
                        submit_job(
                            common_options | norm_options | options,
                            dry_run=dry_run,
                            name=f"{norm_key}_{get_func_name()}_k{-k}_b{bs}_b2{int(beta2*100)}_wu{warmup//100}h",
                            num_nodes=1,
                            cluster=True,
                            config_file=config_file,
                        )


if __name__ == "__main__":
    dry_run = "--dry_run" in sys.argv[1:]
    # 16/06/2025
    # bf16_baseline_250m_lr_sweep(dry_run)
    # bf16_baseline_250m_lr_sweep_bs8(dry_run)

    # 17/06/2025
    # Fixed issues with MXRMSNormConverter code path
    # Fixed issues with `lut_and_lerp`, `linear_scale`, `mean_absmax`

    # 20/06/2025
    # Reran lut_and_kerp with `mx_rmsnorm.sigma_absmax_mapping_k=256`
    # Change lr range to [2**-12, ..., 2**-3]
    # bf16mm_mxnorm_250m_lr_sweep(dry_run)
    # bf16_baseline_250m_lr_sweep_higher_lrs(dry_run)

    # 23/06/2025
    # Rerun bf16mm_mxnorm_250m_lr_sweep after bug fixes
    # bf16mm_mxnorm_250m_lr_sweep(dry_run)
    # bf16_baseline_250m_lr_sweep_higher_lrs(dry_run)

    # 24/05/2025
    # bf16_baseline_1b_lr_sweep(dry_run)
    # bf16mm_mxnorm_1b_lr_sweep(dry_run)

    # 27/06/2025
    # Reran bf16 baseline with upcasted rmsnorm
    # bf16_baseline_250m_lr_sweep_upcast_norm(dry_run)
    # bf16_baseline_1b_lr_sweep_upcast_norm(dry_run)

    # 09/07/2025
    # Run lr sweeps with mxnorm and mxfp matmul
    # mxmm_lr_sweep_1b_250m(dry_run)

    # 28/07/2025
    # Run lr sweeps with normalised to_mx and mxfp matmul
    # mxmm_v2_lr_sweep_1b_250m(dry_run)

    # 08/10/2025
    # Find unstable settings for small mxnorm models
    # Tried increasing batch size and beta2
    # Compare with rmsnorm baseline
    # unstable_250m(dry_run)
    # unstable_slim250m(dry_run)
    slim250m_003(dry_run)
