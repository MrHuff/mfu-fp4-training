#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from quantization.MXFPconfig import MXLinearDimConfig
from quantization.fusedMXFPMatmul import swap_linear_with_mx_linear_fused
from torchao.prototype.mx_formats.constants import DTYPE_FP4
import itertools
import gfloat
from tqdm import tqdm
from test_mxfp_training_all_experiments import BF16LossScaler
# from torchviz import make_dot

torch.autograd.set_detect_anomaly(True)


# TODO, keep playing around and discuss what operations can be included in backprop, try to replicate the total STE and understand why including absmax normalisation etc sucks
def worker_init_fn(worker_id):
    seed = 42
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class RegressionTrainer:
    def __init__(
        self, model, optimizer_statement, loss, lr=1e-2, dtype=torch.bfloat16, epochs=20
    ):
        self.model = model

        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []
        self.optimizer = eval(optimizer_statement, globals(), locals())
        self.dtype = dtype
        self.criterion = loss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        num_samples = 100000
        input_dim = 1024
        X = torch.randn(num_samples, input_dim, dtype=dtype)
        true_weights = torch.randn(input_dim, 1, dtype=dtype)
        y = X @ true_weights
        split = int(0.8 * num_samples)

        train_dataset = TensorDataset(X[:split], y[:split])
        val_dataset = TensorDataset(X[split:], y[split:])

        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(
            train_dataset,
            batch_size=4096,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4096,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            generator=g,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = BF16LossScaler()  # This is only for float16

    def train(self):
        self.model = self.model.to(self.dtype).to(self.device)
        for epoch in range(self.epochs):
            self.model.train()
            train_acc = 0
            for X_batch, y_batch in tqdm(self.train_loader, desc="Training"):
                X_batch, y_batch = (
                    X_batch.to(self.dtype).to(self.device),
                    y_batch.to(self.device),
                )
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                scaled_loss = self.scaler.scale_loss(loss)
                scaled_loss.backward()
                self.scaler.unscale_grads(model)
                self.scaler.step(self.optimizer)
                train_acc += loss.item()

            self.train_losses.append(train_acc / len(self.train_loader))

            # Validation
            self.model.eval()
            val_acc = 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = (
                        X_batch.to(self.dtype).to(self.device),
                        y_batch.to(self.device),
                    )
                    y_pred = self.model(X_batch)
                    val_loss = self.criterion(y_pred, y_batch)
                    val_acc += val_loss
            self.val_losses.append(val_acc / len(self.val_loader))
            print(
                f"Epoch {epoch+1}/{self.epochs}: Train Acc = {self.train_losses[-1]:.4f}, Val Acc = {self.val_losses[-1]:.4f}"
            )


class TinyModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1, dtype=dtype)

    def forward(self, x):
        return self.linear(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Slower, but reproducible
    torch.backends.cudnn.benchmark = False


SCALE_TYPES = ["E4M3"]
MAX_APPROX = ["STE"]
STEP_GRADIENTS = ["STE"]
A_VALUES = [40]  # example alpha values
SCALING_QUANT_GRADIENTS = ["STE"]  # ['STE', 'baseline', 'spline', 'sigmoid'] #
SR = ["all"]
HADAMARD = [None]
OPTIMISER_STATEMENT = ["optim.Adam(self.model.parameters(),lr=lr)"]
SCALE_ROUNDING = [gfloat.RoundMode.TiesToEven]
FP_SCALING_FACTOR = [False]
USE_TENSOR_SCALING = [True]
DATASETS = [
    "gaussian_reg"
]  # ['gaussian_reg', 'MNIST','CIFAR10','IMAGENET100','small_diffusion','big_diffusion','llama_9M','llama_60M','llama_350M','llama_1B','llama_7B']
# TENSOR_SCALING_GRAD = ['STE','ignore','absmax']


# Bruv you've got vanishing gradients, beware of clipping functions and keep using graphviz to inspect the backprop flow
# Also do note that somehow the scaler quantisation function records the entire backpropagation of the scaling quantisation, not sure if this actually is desired!


if __name__ == "__main__":
    # set_seed(42)
    # model = TinyModel()
    # model = model.cuda()
    # base_trainer = RegressionTrainer(model)
    # base_trainer.train()
    DTYPE = torch.bfloat16
    for a in [0.5]:
        all_results = []
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
        ):
            block_size = 32 if scale_type == "E8M0" else 16
            a_list = [1.0]
            if ma in ["softsoftmax", "hardsoftmax"]:
                a_list = A_VALUES
            k_list = [5.0]
            if step_gradient == "baseline":
                k_list = [1, 5]
            if ma in ["STE", "absmax"]:
                TENSOR_SCALING_GRAD_list = ["STE", "ignore", "absmax"]
            if ma in ["softsoftmax", "hardsoftmax"]:
                TENSOR_SCALING_GRAD_list = ["STE", "ignore"]

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
                        "lb": 1.0,
                        "ub": 10000,
                        "use_hadamard": hada_mode,
                        "qGradient": sq,
                        "SR": sr_mode,
                        "use_tensor_scaling": tensor_scaling,
                        "tensor_scaling_grad_est": tsg,
                    },
                    dtype=DTYPE,
                )
                set_seed(42)
                model = TinyModel(DTYPE)
                model = model.cuda()
                swap_linear_with_mx_linear_fused(model, config=config)
                trainer = RegressionTrainer(
                    model, optimizer_statement=optimiser, loss=torch.nn.MSELoss()
                )
                trainer.train()

    # for k,v in trainer.first_epoch_grad[0].items():
    #     v_ = base_trainer.first_epoch_grads[0][k]
    #     print(k, torch.mean((v_-v)**2))
    #     print(v_)
    #     print(v)
