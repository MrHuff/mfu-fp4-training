#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#
import os
import torch
import numpy as np
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from quantization.fusedMXFPMatmul import swap_linear_with_mx_linear_fused
from tqdm import tqdm
import random
import pickle
from benchmark_models.classifiers import ImageNet100, CNNModel, CNNModelCIFAR, LinearModel
from benchmark_models.smallDiffusion import (
    SmallDiffusionTrainer,
    SmallGaussianDiffusion,
    UNet,
)
from benchmark_models.bigDiffusion import (
    BigDiffusionTrainer,
    AdaptedUNet,
    FFHQDataset,
    BigGaussianDiffusion,
)
from benchmark_models.bigger_llama import LlamaTrainerWrapperBigger

import multiprocessing
from benchmark_models.utils import BF16LossScaler


def worker_init_fn(worker_id):
    seed = 42
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def load_config(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["MXLinearDimConfig"], data["meta_config"]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Slower, but reproducible
    torch.backends.cudnn.benchmark = False


def swap_layers(model, metaConfig, swap_bool=True):
    if swap_bool:
        model = model.to(metaConfig["dtype"])
        swap_linear_with_mx_linear_fused(model, config=linearConfig)


class HuggingfaceImageNet100(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item["image"].convert("RGB")
        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


class PredictionTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer_statement,
        loss,
        lr=1e-2,
        dtype=torch.bfloat16,
        epochs=10,
        loss_scaling=False,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []
        self.optimizer = eval(optimizer_statement, globals(), locals())
        self.dtype = dtype
        self.criterion = loss
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = BF16LossScaler()
        self.loss_scaling = loss_scaling
        self.model = self.model.to(self.dtype)
        # self.model = torch.compile(self.model)

    def train(self):
        self.model = self.model.to(self.device)
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

                if self.loss_scaling:
                    scaled_loss = self.scaler.scale_loss(loss)
                    scaled_loss.backward()
                    self.scaler.unscale_grads(model)
                    self.scaler.step(self.optimizer)
                else:
                    loss.backward()
                    self.optimizer.step()
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


def get_dataset_and_model(name="MNIST", checkpoint_folder="checkpoint"):
    train_config = {}
    set_seed(42)

    if name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.MNIST(
            root="./data", train=True, transform=transform, download=True
        )
        val_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )
        input_channels, num_classes = 1, 10
        model = CNNModel(input_channels, num_classes)
        batch_size = 512
        train_config["lr"] = 1e-3
        train_config["epochs"] = 20
        num_workers = 2

    elif name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )
        input_channels, num_classes = 3, 10
        model = CNNModelCIFAR(input_channels, num_classes)
        batch_size = 512
        train_config["lr"] = 1e-3
        train_config["epochs"] = 20
        num_workers = 2

    elif name == "IMAGENET100":
        from datasets import load_dataset

        # Load dataset from HuggingFace
        ds = load_dataset("clane9/imagenet-100")

        # Define transform
        transform = transforms.Compose(
            [
                transforms.Resize(224),  # Resize to ImageNet default size
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

        train_dataset = HuggingfaceImageNet100(ds["train"], transform=transform)
        val_dataset = HuggingfaceImageNet100(ds["validation"], transform=transform)
        input_channels, num_classes = 3, 100
        model = ImageNet100(input_channels, num_classes)
        batch_size = 512
        train_config["lr"] = 1e-3
        train_config["epochs"] = 20
        num_workers = 8

    elif name == "gaussian_reg":
        num_samples = 100000
        input_dim = 1024
        X = torch.randn(num_samples, input_dim, dtype=metaConfig["dtype"])
        true_weights = torch.randn(input_dim, 1, dtype=metaConfig["dtype"])
        y = X @ true_weights
        split = int(0.8 * num_samples)
        train_dataset = TensorDataset(X[:split], y[:split])
        val_dataset = TensorDataset(X[split:], y[split:])
        model = LinearModel(input_dim=input_dim)
        batch_size = 4096
        train_config["lr"] = 1e-2
        train_config["epochs"] = 20
        num_workers = 1

    elif "llama" in name:
        model = LlamaTrainerWrapperBigger(
            model_type=name,
            optimizer_eval_string=metaConfig["optimiser"],
            loss_scaling=metaConfig["loss_scaling"],
            checkpoint_folder=checkpoint_folder,
        )
        train_dataset = datasets.MNIST(root="./data", train=True, download=True)
        val_dataset = datasets.MNIST(root="./data", train=False, download=True)
        batch_size = 1
        num_workers = 1

    elif name == "small_diffusion":
        model = UNet()
        model = model.to(metaConfig["dtype"])
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, transform=transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            root="./data", train=False, transform=transform, download=True
        )
        batch_size = 512
        train_config["lr"] = 1e-3
        train_config["epochs"] = 20
        num_workers = 4

    elif name == "big_diffusion":
        image_resolution = 128
        base_channels = (
            64  # Good starting point for 128x128, adjust based on VRAM/performance
        )
        time_emb_dim = 256  # Time embedding dimension
        num_down_blocks = 2  # Number of down/up stages for 128x128
        # Attention at 64x64 and 32x32 resolutions (for 128x128 input)
        attn_resolutions = [64, 32]
        model = AdaptedUNet(
            in_channels=3,
            out_channels=3,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
            num_down_blocks=num_down_blocks,
            attn_resolutions=attn_resolutions,
        )
        model = model.to(metaConfig["dtype"])
        model = torch.compile(model)
        train_dataset = FFHQDataset(resolution=image_resolution, mode="train")
        val_dataset = FFHQDataset(resolution=image_resolution, mode="val")
        batch_size = 20
        train_config["lr"] = 1e-4
        train_config["epochs"] = 3
        num_workers = 12

    else:
        raise ValueError("Unsupported dataset")
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=g,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        generator=g,
        persistent_workers=True,
    )
    return train_loader, val_loader, model, train_config


def orchestrateTraining(
    linearConfig,
    metaConfig,
    train_loader,
    val_loader,
    model,
    train_configs,
    checkpoint_folder="checkpoint_folder",
):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    swap_layers_bool = linearConfig is not None

    if metaConfig["dataset"] in ["gaussian_reg", "MNIST", "CIFAR10", "IMAGENET100"]:
        swap_layers(model, metaConfig=metaConfig, swap_bool=swap_layers_bool)
        loss = (
            torch.nn.MSELoss()
            if metaConfig["dataset"] == "gaussian_reg"
            else torch.nn.CrossEntropyLoss()
        )
        trainer = PredictionTrainer(
            model,
            train_loader,
            val_loader,
            metaConfig["optimiser"],
            lr=train_configs["lr"],
            loss=loss,
            dtype=metaConfig["dtype"],
            epochs=train_configs["epochs"],
            loss_scaling=metaConfig["loss_scaling"],
        )
        trainer.train()
        losses = trainer.train_losses
        val_losses = trainer.val_losses
    elif "llama" in metaConfig["dataset"]:
        losses, val_losses = model.run(config=linearConfig)

    elif metaConfig["dataset"] == "small_diffusion":
        swap_layers(model, metaConfig=metaConfig, swap_bool=swap_layers_bool)
        diffusion = SmallGaussianDiffusion(model, dtype=metaConfig["dtype"])
        trainer = SmallDiffusionTrainer(
            model=model,
            diffusion=diffusion,
            dataloader=train_loader,
            optimizer_statement=metaConfig["optimiser"],
            lr=train_configs["lr"],
            sample_dir=checkpoint_folder,
            epochs=train_configs["epochs"],
            loss_scaling=metaConfig["loss_scaling"],
            val_loader=val_loader,
        )
        trainer.train()
        losses = trainer.losses
        val_losses = trainer.val_losses

    elif metaConfig["dataset"] == "big_diffusion":
        swap_layers(model, metaConfig=metaConfig, swap_bool=swap_layers_bool)
        diffusion = BigGaussianDiffusion(model, dtype=metaConfig["dtype"], timesteps=1000)
        trainer = BigDiffusionTrainer(
            model=model,
            diffusion=diffusion,
            dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer_statement=metaConfig["optimiser"],
            sample_dir=checkpoint_folder,  # Save samples to a different directory
            lr=train_configs["lr"],
            epochs=train_configs[
                "epochs"
            ],  # Or more, training diffusion models takes time
            image_resolution=128,
            loss_scaling=metaConfig["loss_scaling"],
        )
        trainer.train()
        losses = trainer.losses
        val_losses = trainer.val_losses

    return losses, val_losses


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_config", type=str, required=True)
    parser.add_argument("--config_id", type=str, required=True)
    args = parser.parse_args()

    file_path = args.file_config
    config_id = args.config_id

    file = f"{file_path}/config_{config_id}.pkl"
    results_fold = f"{file_path}_results"
    results_path = f"{results_fold}/results_{config_id}.pkl"
    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0
    checkpoint_folder = f"{file_path}_config_{config_id}"
    if is_main_process and os.path.exists(results_path):
        print(f"Results already exist at {results_path}, skipping...")
        exit(0)

    linearConfig, metaConfig = load_config(filepath=file)
    if is_main_process:
        print(linearConfig, metaConfig)

    train_loader, val_loader, model, train_configs = get_dataset_and_model(
        metaConfig["dataset"], checkpoint_folder
    )
    training_metrics, val_metrics = orchestrateTraining(
        linearConfig,
        metaConfig,
        train_loader,
        val_loader,
        model,
        train_configs,
        checkpoint_folder,
    )

    # Only the main process saves the final result.
    if is_main_process:
        jobOutput = {
            "training_metrics": training_metrics,
            "val_metrics": val_metrics,
            **metaConfig,
        }
        os.makedirs(results_fold, exist_ok=True)
        with open(results_path, "wb") as f:
            pickle.dump(jobOutput, f)
        print(f"🎉 Results saved to {results_path}")
