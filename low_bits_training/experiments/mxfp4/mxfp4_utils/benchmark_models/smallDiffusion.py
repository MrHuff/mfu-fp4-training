#
# Copyright (c) 2025 Graphcore Ltd. All rights reserved.
#

import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm
import os
from benchmark_models.utils import BF16LossScaler
import math
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings).to(
            time.dtype
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        # print(f"ResBlock __init__: in_ch={in_ch}, out_ch={out_ch}")
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),  # This is the line where the error originates
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch), nn.SiLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        # print(f"ResBlock forward - Input x shape: {x.shape}")
        h = self.block1(x)
        # print(f"ResBlock forward - After block1 h shape: {h.shape}")
        h = h + self.time_proj(t)[:, :, None, None]
        h = self.block2(h)
        # print(f"ResBlock forward - After block2 h shape: {h.shape}")
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, base=128, time_dim=256):
        super().__init__()
        # print(f"UNet __init__: base={base}, time_dim={time_dim}")
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )
        # self.time_mlp = nn.Sequential(
        #     nn.Linear(1, time_dim),
        #     nn.SiLU(),
        #     nn.Linear(time_dim, time_dim),
        # )
        self.init = nn.Conv2d(3, base, 3, padding=1)
        self.down1 = ResBlock(base, base * 2, time_dim)
        self.down2 = ResBlock(base * 2, base * 4, time_dim)
        self.mid = ResBlock(base * 4, base * 4, time_dim)
        # Check these lines very carefully
        self.up2 = ResBlock(base * 6, base * 2, time_dim)  # This is the likely culprit
        self.up1 = ResBlock(base * 3, base, time_dim)  # from 512 → 384

        self.final = nn.Sequential(
            nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1)
        )
        self.pool = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.init(x)
        x2 = self.down1(self.pool(x1), t_emb)
        x3 = self.down2(self.pool(x2), t_emb)
        x4 = self.mid(x3, t_emb)
        # The input to up2 is what needs careful scrutiny
        up2_input = torch.cat([self.up(x4), x2], 1)
        # print(f"UNet forward - Input to up2 (concatenated) shape: {up2_input.shape}")
        x = self.up2(up2_input, t_emb)  # Error occurs here
        # print(f"UNet forward - After up2 x shape: {x.shape}")
        up1_input = torch.cat([self.up(x), x1], 1)
        # print(f"UNet forward - Input to up1 (concatenated) shape: {up1_input.shape}")
        x = self.up1(up1_input, t_emb)
        return self.final(x)


class SmallGaussianDiffusion(nn.Module):
    def __init__(
        self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, dtype=torch.bfloat16
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.dtype = dtype
        self.register_schedule(beta_start, beta_end)

    def register_schedule(self, beta_start, beta_end):
        betas = torch.linspace(beta_start, beta_end, self.timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas.to(self.dtype))
        self.register_buffer("alphas", alphas.to(self.dtype))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(self.dtype))
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(self.dtype)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).to(self.dtype),
        )

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        # Use .gather() for safe indexing
        sqrt_alpha = self.sqrt_alphas_cumprod.gather(0, t).reshape(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod.gather(0, t).reshape(
            -1, 1, 1, 1
        )

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def p_losses(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)

        # --- FIX 1 & 2 ---
        # Pass integer 't' directly, not the normalized float
        predicted_noise = self.model(x_noisy, t.to(x_noisy.dtype))
        # Use the correct functional module 'F'
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    @torch.no_grad()
    def sample(self, shape, device):
        x = torch.randn(shape, device=device, dtype=self.dtype)

        for t_int in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            time_tensor = torch.full((shape[0],), t_int, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self.model(x, time_tensor.to(self.dtype))

            # Get alpha and beta values
            alpha_bar = self.alphas_cumprod[t_int]
            alpha_bar_prev = (
                self.alphas_cumprod[t_int - 1]
                if t_int > 0
                else torch.tensor(1.0, device=device)
            )

            # 1. Predict the original image (x0)
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(
                alpha_bar
            )
            pred_x0 = pred_x0.clamp(-1.0, 1.0)  # Clip to valid range

            # 2. Compute direction pointing to x_t
            pred_dir = torch.sqrt(1 - alpha_bar_prev) * predicted_noise

            # 3. Compute x_{t-1}
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + pred_dir

        return x


class SmallDiffusionTrainer:
    # --- FIX 3 ---
    def __init__(
        self,
        model,
        diffusion,
        dataloader,
        optimizer_statement,
        sample_dir="samples",
        lr=1e-3,
        epochs=100,
        loss_scaling=False,
        val_loader=None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        self.epochs = epochs
        self.sample_dir = sample_dir
        os.makedirs(self.sample_dir, exist_ok=True)
        self.loader = dataloader
        self.val_loader = val_loader
        self.loss_scaling = loss_scaling

        self.scaler = BF16LossScaler()

        # Safer optimizer creation
        self.optimizer = eval(optimizer_statement, globals(), locals())
        # Add a scheduler for faster convergence
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=lr, steps_per_epoch=len(self.loader), epochs=epochs
        )
        self.losses = []
        self.val_losses = []

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            loss_val = 0.0
            for step, (images, _) in enumerate(
                tqdm(self.loader, desc=f"Epoch {epoch} [Train]")
            ):
                images = images.to(self.diffusion.dtype).to(self.device)
                t = torch.randint(
                    0, self.diffusion.timesteps, (images.size(0),), device=self.device
                )

                self.optimizer.zero_grad()
                loss = self.diffusion.p_losses(images, t)
                if self.loss_scaling:
                    scaled_loss = self.scaler.scale_loss(loss)

                    scaled_loss.backward()

                    self.scaler.unscale_grads(self.model)

                    self.scaler.step(self.optimizer)

                else:
                    loss.backward()

                    self.optimizer.step()

                self.scheduler.step()  # Step the scheduler each iteration
                loss_val += loss.item()
                if step % 100 == 0:
                    print(
                        f"\n[Epoch {epoch} | Step {step}] Loss: {loss.item():.4f}, LR: {self.scheduler.get_last_lr()[0]:.6f}"
                    )
            self.losses.append(loss_val / len(self.loader))

            # --- VALIDATION STEP ---
            if self.val_loader:
                self.validate(epoch)

            # Save generated samples
            self.save_samples(epoch)

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()  # Set model to evaluation mode
        val_loss_val = 0.0
        for images, _ in tqdm(
            self.val_loader, desc=f"Epoch {epoch} [Validate]", leave=False
        ):
            images = images.to(self.diffusion.dtype).to(self.device)
            t = torch.randint(
                0, self.diffusion.timesteps, (images.size(0),), device=self.device
            )
            loss = self.diffusion.p_losses(images, t)
            val_loss_val += loss.item()

        avg_val_loss = val_loss_val / len(self.val_loader)
        self.val_losses.append(avg_val_loss)

    @torch.no_grad()
    def save_samples(self, epoch, n_samples=32):
        self.model.eval()
        samples = self.diffusion.sample((n_samples, 3, 32, 32), self.device)
        samples = (samples.clamp(-1, 1) + 1) / 2  # Rescale from [-1, 1] to [0, 1]
        vutils.save_image(samples, f"{self.sample_dir}/epoch_{epoch}.png", nrow=8)
