#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import os
import torch
import itertools
from quantization.dimensionQuantisationClass import MXLinearDimConfig
from torchao.prototype.mx_formats.constants import DTYPE_FP4


import pickle


def jsonToLinearConfig(jsonBlob: dict):
    config = MXLinearDimConfig(
        block_dim=None,
        block_size=jsonBlob["block_size"],
        elem_dtype=jsonBlob["elem_dtype"],
        scale_type=jsonBlob["scale_type"],
        use_approx={
            "smooth": jsonBlob["smooth"],
            "alpha": jsonBlob["alpha"],
            "stepGradient": jsonBlob["stepGradient"],
            "temperature": jsonBlob["temperature"],
            "k": jsonBlob["k"],
            "dtype": jsonBlob["dtype"],
            "lb": jsonBlob["lb"],
            "ub": jsonBlob["ub"],
            "use_hadamard": jsonBlob["use_hadamard"],
            "qGradient": jsonBlob["qGradient"],
            "SR": jsonBlob["SR"],
        },
        dtype=jsonBlob["dtype"],
    )
    return config


# high learning rates!!!! lr=0.01, justify with Stable-SPAM observation and looking at other people's code.

# Search space
# BLOCK_SIZES = [16,32]
# SCALE_TYPES = ['E8M0','E4M3']
# MAX_APPROX = ['STE','softsoftmax','hardsoftmax','absmax']
# STEP_GRADIENTS = ['STE', 'baseline','spline','sigmoid']
# A_VALUES = [0.25, 0.5, 1.0,5,10,20]  # example alpha values
# SCALING_QUANT_GRADIENTS = ['STE', 'baseline','spline','sigmoid'] #[None, 'baseline', 'spline', 'sigmoid'] #
# SR = [None,'all','forward','backward']
# HADAMARD= [None,'all','forward','backward']
# OPTIMISER_STATEMENT = ['optim.Adam(self.model.parameters(), lr=lr)','optim.SGD(self.model.parameters(), lr=lr)', 'StableSPAM(self.model.parameters(), lr=lr, adaclip =True)','StableSPAM(self.model.parameters(), lr=lr,adaclip=False)' ]
# DATASETS = ['gaussian_reg']#['gaussian_reg', 'MNIST','CIFAR10','IMAGENET100','small_diffusion','big_diffusion','llama_9M','llama_60M','llama_350M','llama_1B','llama_7B']

BLOCK_SIZES = [16]
SCALE_TYPES = ["E4M3"]
MAX_APPROX = ["absmax"]
STEP_GRADIENTS = ["baseline"]
A_VALUES = [40, 80]  # example alpha values
SCALING_QUANT_GRADIENTS = ["baseline"]  # [None, 'baseline', 'spline', 'sigmoid'] #
SR = ["all"]
HADAMARD = ["all"]
OPTIMISER_STATEMENT = ["optim.Adam(self.model.parameters(), lr=lr)"]
DATASETS = ["llama_9M"]

if __name__ == "__main__":
    DTYPE = torch.bfloat16

    JOB_NAME = "EXAMPLE"
    os.makedirs(JOB_NAME, exist_ok=True)
    file_counter = 0
    for (
        block_size,
        scale_type,
        ma,
        step_gradient,
        sq,
        sr_mode,
        hada_mode,
        optimiser,
        ds,
    ) in itertools.product(
        BLOCK_SIZES,
        SCALE_TYPES,
        MAX_APPROX,
        STEP_GRADIENTS,
        SCALING_QUANT_GRADIENTS,
        SR,
        HADAMARD,
        OPTIMISER_STATEMENT,
        DATASETS,
    ):
        a_list = [1.0]
        if ma in ["softsoftmax", "hardsoftmax"]:
            a_list = A_VALUES
        k_list = [5.0]
        if step_gradient == "baseline":
            k_list = [1, 5]

        for a, k in itertools.product(a_list, k_list):
            config = MXLinearDimConfig(
                block_dim=None,
                block_size=block_size,
                elem_dtype=DTYPE_FP4,
                scale_type=scale_type,
                use_approx={
                    "smooth": ma,
                    "alpha": a,
                    "stepGradient": step_gradient,
                    "temperature": 0.025,
                    "k": k,
                    "dtype": DTYPE,
                    "lb": 1,
                    "ub": 1.5,
                    "use_hadamard": hada_mode,
                    "qGradient": sq,
                    "SR": sr_mode,
                },
                dtype=DTYPE,
            )
            config_json_ref = {
                "elem_dtype": DTYPE_FP4,
                "block_dim": None,
                "scale_type": scale_type,
                "block_size": block_size,
                "smooth": ma,
                "alpha": a,
                "stepGradient": step_gradient,
                "temperature": 0.025,
                "k": k,
                "dtype": DTYPE,
                "lb": 1,
                "ub": 1.5,
                "use_hadamard": hada_mode,
                "qGradient": sq,
                "SR": sr_mode,
                "dataset": ds,
                "optimiser": optimiser,
            }

            dict_save = {"meta_config": config_json_ref, "MXLinearDimConfig": config}
            out_path = os.path.join(JOB_NAME, f"config_{file_counter}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(dict_save, f)

            file_counter += 1
