#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import itertools
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from quantization.MXFPconfig import MXLinearDimConfig
from quantization.fusedMXFPMatmul import swap_linear_with_mx_linear_fused
from torchao.prototype.mx_formats.constants import DTYPE_FP4
import random
import gfloat
from test_mxfp_training_all_experiments import BF16LossScaler

# torch.autograd.set_detect_anomaly(True)
# TODO ok big issue here is that all gradients are 0 for some dumb reason for E4M3, E8M0 only gets tiny gradients through... why?


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Slower, but reproducible
    torch.backends.cudnn.benchmark = False


class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample 2x
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNNModelCIFAR(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Downsample 2x
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.LayerNorm(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8),
            nn.Linear(64 * 7 * 7 if input_channels == 1 else 64 * 8 * 8, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ImageTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        lr=1e-4,
        epochs=20,
        dtype=torch.bfloat16,
        loss_scaler=False,
    ):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.train_accuracies = []
        self.val_accuracies = []
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.dtype = dtype
        self.scaler = BF16LossScaler()  # This is only for float16
        self.loss_scaler = loss_scaler

    def _accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        return (preds == labels).float().mean().item()

    def train(self):
        print(model)
        self.model.train()
        self.model.to(self.device)
        for epoch in range(self.epochs):
            train_acc = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = (
                    X_batch.to(self.dtype).to(self.device),
                    y_batch.to(self.device),
                )
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)

                if self.loss_scaler:
                    scaled_loss = self.scaler.scale_loss(loss)
                    scaled_loss.backward()
                    self.scaler.unscale_grads(model)
                    self.scaler.step(self.optimizer)
                else:
                    loss.backward()
                    self.optimizer.step()

                train_acc += self._accuracy(y_pred, y_batch)

            self.train_accuracies.append(train_acc / len(self.train_loader))

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
                    val_acc += self._accuracy(y_pred, y_batch)

            self.val_accuracies.append(val_acc / len(self.val_loader))
            print(
                f"Epoch {epoch+1}/{self.epochs}: Train Acc = {self.train_accuracies[-1]:.4f}, Val Acc = {self.val_accuracies[-1]:.4f}"
            )


def load_dataset(name="MNIST", batch_size=128):
    if name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        val_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )
        input_channels, num_classes = 1, 10
    elif name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )
        input_channels, num_classes = 3, 10
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, input_channels, num_classes


# Search space
SCALE_TYPES = ["E4M3"]
MAX_APPROX = ["STE"]
STEP_GRADIENTS = ["STE"]
A_VALUES = [40]  # example alpha values
SCALING_QUANT_GRADIENTS = ["STE"]  # [None, 'baseline', 'spline', 'sigmoid'] #
SR = ["all"]
HADAMARD = [None]
OPTIMISER_STATEMENT = ["optim.Adam(self.model.parameters(), lr=lr)"]
SCALE_ROUNDING = [gfloat.RoundMode.TiesToEven]
FP_SCALING_FACTOR = [False]
DATASETS = ["MNIST"]
LOSS_SCALING = [True]
USE_TENSOR_SCALING = [True]
# Tend to diverge towards the end with spiky training losses - that's consistent with exploding optimizers!
# Seems like outliers does fuck shit up - try hadamard trick or SR
# Scale quantisation error! Adjust for that as well! MNIST not playing well
# Most interesting - E8M0 only converges with Adam, E4M3 does a descent job with Adam as well as SGD!
# Welp, let's revisit the range of the reconstructions error and try understand the trade-offs!

# Paper - take, an accurate MXFP4 simulator!!!
# Understand why softmax approximation doesn't work!

# Find maximally "safe" learning rate for high precision stuff.
# Implement loss scaling for E4M3, gradient subnormal range weak - tends to benefits from a little "boost".
if __name__ == "__main__":
    DTYPE = torch.bfloat16
    batch_size = 512
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
        loss_scaling,
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
        LOSS_SCALING,
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
                    "lb": 1,
                    "ub": 100,
                    "use_hadamard": hada_mode,
                    "qGradient": sq,
                    "SR": sr_mode,
                    "use_tensor_scaling": tensor_scaling,
                    "tensor_scaling_grad_est": "ignore",
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
                "loss_scaling": loss_scaling,
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
        train_loader, val_loader, in_ch, out_ch = load_dataset(ds, batch_size=batch_size)
        model = CNNModel(in_ch, out_ch) if ds == "MNIST" else CNNModelCIFAR(in_ch, out_ch)
        model = model.to(DTYPE)
        swap_linear_with_mx_linear_fused(model, config=config)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"Weights of {name}: {module.weight.data}")

        trainer = ImageTrainer(
            model, train_loader, val_loader, dtype=DTYPE, loss_scaler=loss_scaling
        )
        trainer.train()
