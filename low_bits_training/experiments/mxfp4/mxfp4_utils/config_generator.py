#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import os
import torch
import itertools
from quantization.MXFPconfig import MXLinearDimConfig, DTYPE_FP4
import pickle
import gfloat


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
        out_path = os.path.join(job_name, f"config_{idx}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(dict_save, f)


def save_dataset_configs_stable_spam(datasets, job_name, dtype):
    os.makedirs(job_name, exist_ok=True)
    for idx, ds in enumerate(datasets):
        dict_save = {
            "meta_config": {
                "dataset": ds,
                "dtype": dtype,
                "optimiser": "StableSPAM(self.model.parameters(), lr=lr, adaclip =True)",
                "loss_scaling": False,
            },
            "MXLinearDimConfig": None,
        }
        out_path = os.path.join(job_name, f"config_{idx}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(dict_save, f)


# high learning rates!!!! lr=0.01, justify with Stable-SPAM observation and looking at other people's code.

# Search space
SCALE_TYPES = ["E4M3"]
MAX_APPROX = ["STE"]  # ['STE','softsoftmax','hardsoftmax','absmax']
STEP_GRADIENTS = ["STE"]  # ['STE', 'baseline','spline','sigmoid']
A_VALUES = [80]  # example alpha values
SCALING_QUANT_GRADIENTS = ["STE"]  # ['STE', 'baseline','spline','sigmoid']
SR = [None, "all", "IntelFP4", "NewIntelFP4"]  # [None,'all','forward','backward']
HADAMARD = [None]  # [None,'all','forward','backward']
OPTIMISER_STATEMENT = [
    "optim.Adam(self.model.parameters(), lr=lr)"
]  #'optim.Adam(self.model.parameters(), lr=lr)', 'StableSPAM(self.model.parameters(), lr=lr, adaclip =True)'
SCALE_ROUNDING = [
    gfloat.RoundMode.TiesToEven
]  # gfloat.RoundMode.TiesToEven,gfloat.RoundMode.TowardPositive
FP_SCALING_FACTOR = [False]  # True, False
LOSS_SCALING = [True]  # True,False
USE_TENSOR_SCALING = [False, True]

DATASETS = [
    "gaussian_reg",
    "MNIST",
    "CIFAR10",
    "IMAGENET100",
    "small_diffusion",
    "big_diffusion",
    "llama_9M",
    "llama_60M",
]  # ['gaussian_reg', 'MNIST','CIFAR10','IMAGENET100','small_diffusion','big_diffusion','llama_9M','llama_60M','llama_350M','llama_1B','llama_7B']

# BLOCK_SIZES = [32]
# SCALE_TYPES = ['E8M0']
# MAX_APPROX = ['STE']
# STEP_GRADIENTS = ['STE']
# A_VALUES = [0.25, 0.5, 1.0,5,10,20]  # example alpha values
# SCALING_QUANT_GRADIENTS = ['STE'] #[None, 'baseline', 'spline', 'sigmoid'] #
# SR = [None]
# HADAMARD= [None]
# OPTIMISER_STATEMENT = ['optim.Adam(self.model.parameters(), lr=lr)' ]
# DATASETS = ['gaussian_reg', 'MNIST','CIFAR10','IMAGENET100','small_diffusion','big_diffusion','llama_9M']

if __name__ == "__main__":
    DTYPE = torch.bfloat16

    JOB_NAME = "EXAMPLE"
    os.makedirs(JOB_NAME, exist_ok=True)
    file_counter = 0
    for (
        scale_type,
        ma,
        step_gradient,
        sq,
        sr_mode,
        hada_mode,
        optimiser,
        ds,
        scale_round_mode,
        fp_scale_factor,
        tensor_scaling,
        loss_scaling,
    ) in itertools.product(
        SCALE_TYPES,
        MAX_APPROX,
        STEP_GRADIENTS,
        SCALING_QUANT_GRADIENTS,
        SR,
        HADAMARD,
        OPTIMISER_STATEMENT,
        DATASETS,
        SCALE_ROUNDING,
        FP_SCALING_FACTOR,
        USE_TENSOR_SCALING,
        LOSS_SCALING,
    ):
        block_size = 32 if scale_type == "E8M0" else 16

        a_list = [1.0]
        if ma in ["softsoftmax", "hardsoftmax"]:
            a_list = A_VALUES
        k_list = [5.0]
        if step_gradient == "baseline":
            k_list = [5]
        TENSOR_SCALING_GRAD_list = [None]
        if tensor_scaling:
            if ma in ["STE", "absmax"]:
                TENSOR_SCALING_GRAD_list = ["ignore"]  # ['ignore','absmax']
            if ma in ["softsoftmax", "hardsoftmax"]:
                TENSOR_SCALING_GRAD_list = ["ignore"]

        for a, k, tsg in itertools.product(a_list, k_list, TENSOR_SCALING_GRAD_list):
            config = MXLinearDimConfig(
                block_dim=None,
                block_size=block_size,
                elem_dtype=DTYPE_FP4,
                scale_type=scale_type,
                fp_scale_factor=fp_scale_factor,
                roundMode=scale_round_mode,
                use_approx={
                    "smooth": ma,
                    "alpha": a,
                    "stepGradient": step_gradient,
                    "temperature": 0.5,
                    "k": k,
                    "dtype": DTYPE,
                    "lb": 0.5,
                    "ub": 10000,
                    "use_hadamard": hada_mode,
                    "qGradient": sq,
                    "SR": sr_mode,
                    "use_tensor_scaling": tensor_scaling,
                    "tensor_scaling_grad_est": tsg,
                },
                dtype=DTYPE,
            )
            config_json_ref = {
                "elem_dtype": DTYPE_FP4,
                "block_dim": None,
                "scale_type": scale_type,
                "block_size": block_size,
                "loss_scaling": loss_scaling,
                "smooth": ma,
                "alpha": a,
                "stepGradient": step_gradient,
                "temperature": 0.5,
                "fp_scale_factor": fp_scale_factor,
                "roundMode": scale_round_mode,
                "k": k,
                "dtype": DTYPE,
                "lb": 0.5,
                "ub": 10000.0,
                "use_hadamard": hada_mode,
                "qGradient": sq,
                "SR": sr_mode,
                "dataset": ds,
                "optimiser": optimiser,
                "use_tensor_scaling": tensor_scaling,
                "tensor_scaling_grad_est": tsg,
            }

            dict_save = {"meta_config": config_json_ref, "MXLinearDimConfig": config}
            out_path = os.path.join(JOB_NAME, f"config_{file_counter}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(dict_save, f)

            file_counter += 1

    DTYPE = torch.bfloat16  # or whatever your global dtype is
    save_dataset_configs(
        ["gaussian_reg", "MNIST", "CIFAR10", "llama_9M", "small_diffusion"],
        "baseline_jobs_small",
        DTYPE,
    )
    save_dataset_configs_stable_spam(
        [
            "gaussian_reg",
            "MNIST",
            "CIFAR10",
            "llama_9M",
            "small_diffusion",
            "IMAGENET100",
            "big_diffusion",
            "llama_60M",
        ],
        "single_gpu_baseline_stable_spam",
        DTYPE,
    )
    for llama in ["llama_9M", "llama_60M", "llama_350M", "llama_1B", "llama_7B"]:
        save_dataset_configs([llama], f"baseline_{llama}", DTYPE)

    for llama in ["llama_350M", "llama_1B"]:
        save_dataset_configs_stable_spam([llama], f"baseline_stable_spam_{llama}", DTYPE)
