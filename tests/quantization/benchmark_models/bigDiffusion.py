import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from tqdm import tqdm
import os
from datasets import load_dataset
from low_bits_training.quantization.stable_spam import (
    StableSPAM
)
import torch.optim as optim

from PIL import Image
import math
from benchmark_models.utils import BF16LossScaler

class FFHQDataset(Dataset):
    def __init__(self, root_dir="./data", resolution=128, mode = 'train'):
        super().__init__()
        self.resolution = resolution
        self.local_save_path = os.path.join(root_dir, 'ffhq_128_hf_local')

        print(f"Loading FFHQ 128x128 dataset from Hugging Face Hub or local cache...")
        # Load the dataset. 'load_dataset' will first check local cache
        # then download if not found.
        ffhq_hf_dataset = load_dataset("nuwandaa/ffhq128")

        # Save the dataset to the specified local path if it's not already there
        # This ensures a persistent local copy in the specified directory.
        if not os.path.exists(self.local_save_path):
            os.makedirs(self.local_save_path, exist_ok=True)
            ffhq_hf_dataset.save_to_disk(self.local_save_path)
            print(f"Dataset saved locally to: {self.local_save_path}")
        else:
            print(f"Dataset already exists locally at: {self.local_save_path}")

        # We'll use the 'train' split of the dataset
        full = ffhq_hf_dataset['train']
        split_dataset = full.train_test_split(test_size=0.05, seed=42)
        train_ds = split_dataset['train']
        val_ds = split_dataset['test']

        if mode == 'train':
            self.ds = train_ds
        elif mode == 'val':
            self.ds = val_ds
        print(f"Dataset loaded with {len(self.ds)} samples from Hugging Face.")

        transform_list = [
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        if resolution != 128:
             # Resize should come before ToTensor for better quality and performance
             transform_list.insert(0, transforms.Resize(resolution, antialias=True))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # Access the image from the Hugging Face dataset.
        # The 'image' column in the 'nuwandaa/ffhq128' dataset typically returns a PIL Image.
        image = self.ds[idx]['image']

        # The dataset typically yields PIL Images directly, so conversion from numpy array is not always needed here.
        # However, if 'image' could be a numpy array, the following check is still useful:
        if not isinstance(image, Image.Image):
             # If it's a numpy array (e.g., uint8), convert it to PIL Image
             if hasattr(image, 'dtype') and image.dtype != 'uint8':
                 image = image.astype('uint8')
             image = Image.fromarray(image)

        return self.transform(image)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fourier_freqs = nn.Parameter(torch.randn(dim // 2), requires_grad=False)

    def forward(self, t):
        args = t.unsqueeze(-1) * self.fourier_freqs
        emb = torch.cat((args.sin(), args.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        # Ensure num_groups (8) divides in_channels
        # Added these assertions for debugging clarity, they are good practice.
        assert in_channels % 8 == 0, f"ResBlock: in_channels ({in_channels}) must be divisible by 8 for GroupNorm."
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Ensure num_groups (8) divides out_channels
        assert out_channels % 8 == 0, f"ResBlock: out_channels ({out_channels}) must be divisible by 8 for GroupNorm."
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1) 
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)
class AttentionBlock(nn.Module):
    """
    An attention block that is optimized for speed with flash attention
    and channels_last memory format.
    """
    def __init__(self, channels):
        super().__init__()
        assert channels % 8 == 0, f"AttentionBlock: channels ({channels}) must be divisible by 8 for GroupNorm."
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        
        # Use nn.Linear for QKV projection. It will operate on the last dimension (channels).
        self.qkv = nn.Linear(
            in_features=channels, 
            out_features=channels * 3
        )
        self.proj = nn.Linear(
            in_features=channels, 
            out_features=channels
        )

    def forward(self, x):
        # Input 'x' is channels-first: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Sanity check for debugging
        x_in = x
        
        x = self.norm(x)
        
        # Reshape for Linear layer: (B, C, H, W) -> (B, H * W, C)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        
        # Project to Q, K, V
        qkv = self.qkv(x).chunk(3, dim=-1)
        
        # Apply scaled dot-product attention (Flash Attention)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        
        # Final projection
        x = self.proj(x)
        
        # Reshape back to channels-first: (B, H * W, C) -> (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        return x + x_in

class AdaptedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, time_emb_dim=512, num_down_blocks=2,
                 attn_resolutions=[64, 32],
                 image_resolution=128):
        super().__init__()
        self.image_resolution = image_resolution
        self.num_down_blocks = num_down_blocks
        self.base_channels = base_channels

        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.initial_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.mid_blocks = nn.ModuleList()

        current_channels = base_channels

        # Downsampling path
        for i in range(num_down_blocks):
            multiplier = 2 ** (i + 1)
            next_channels = base_channels * multiplier
            s_down = self.image_resolution // (2**(i + 1))
            is_attn_down = (s_down in attn_resolutions)

            self.down_blocks.append(nn.ModuleList([
                ResBlock(current_channels, next_channels, time_emb_dim),
                AttentionBlock(next_channels) if is_attn_down else nn.Identity(),
                ResBlock(next_channels, next_channels, time_emb_dim),
                nn.Conv2d(next_channels, next_channels, 4, stride=2, padding=1)
            ]))
            current_channels = next_channels

        # Middle blocks
        self.mid_blocks.append(ResBlock(current_channels, current_channels, time_emb_dim))
        self.mid_blocks.append(AttentionBlock(current_channels))
        self.mid_blocks.append(ResBlock(current_channels, current_channels, time_emb_dim))

        # Upsampling path
        channels_for_upsample_input = current_channels

        for i in range(num_down_blocks):
             out_channels_transposed = base_channels * (2**(num_down_blocks - i - 1))
             s_up = (self.image_resolution // (2**num_down_blocks)) * (2**(i+1))
             is_attn_up = (s_up in attn_resolutions)
             skip_connection_channels = base_channels * (2**(self.num_down_blocks - i))
             in_channels_resblock = out_channels_transposed + skip_connection_channels

             self.up_blocks.append(nn.ModuleList([
                 ResBlock(in_channels_resblock, out_channels_transposed, time_emb_dim),
                 AttentionBlock(out_channels_transposed) if is_attn_up else nn.Identity(),
                 ResBlock(out_channels_transposed, out_channels_transposed, time_emb_dim),
                 nn.ConvTranspose2d(channels_for_upsample_input, out_channels_transposed, 4, stride=2, padding=1)
             ]))
             channels_for_upsample_input = out_channels_transposed

        # Final output layers
        self.final_conv = nn.Conv2d(channels_for_upsample_input + base_channels, out_channels, 3, padding=1)
        self.final_act = nn.SiLU()
        self.output_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.final_output = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    # ====================================================================
    # THIS IS THE METHOD WITH DEBUG PRINTS
    # ====================================================================
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        skips = []

        # print("\n--- U-NET FORWARD PASS ---")
        # print(f"[Input] x shape: {x.shape}")

        x = self.initial_conv(x)
        # print(f"[Initial Conv] -> {x.shape}")
        skips.append(x)

        # --- Downsampling Path ---
        # print("\n--- Downsampling Path ---")
        for i, block_list in enumerate(self.down_blocks):
            block1, attn, block2, downsample_conv = block_list
            x = block1(x, t_emb)
            x = attn(x)
            x = block2(x, t_emb)
            skips.append(x)
            x = downsample_conv(x)

        # --- Middle blocks ---
        resblock_mid1, attn_mid, resblock_mid2 = self.mid_blocks
        x = resblock_mid1(x, t_emb)
        x = attn_mid(x)
        x = resblock_mid2(x, t_emb)

        skips.reverse()

        # --- Upsampling Path ---
        # print("\n--- Upsampling Path ---")
        for i, block_list in enumerate(self.up_blocks):
            block1, attn, block2, upsample_conv = block_list
            # print(f"\n[Up Block {i}] input: {x.shape}")
            x = upsample_conv(x)
            skip = skips[i]
            # print(f"  > Upsampled x: {x.shape} | Skip connection: {skip.shape}")
            x = torch.cat([x, skip], dim=1)
            # print(f"  > After cat: {x.shape}")
            x = block1(x, t_emb)
            # print(f"  > After ResBlock1: {x.shape}")
            # print(f"  > Entering AttentionBlock...")
            x = attn(x)
            # print(f"  < Exited AttentionBlock. Shape: {x.shape}")
            x = block2(x, t_emb)
            # print(f"  > After ResBlock2: {x.shape}")

        # print("\n--- Final Layers ---")
        # print(f"[Final Concat] input x: {x.shape} | Skip connection: {skips[-1].shape}")
        x = torch.cat([x, skips[-1]], dim=1)
        # print(f"  > After final cat: {x.shape}")

        x = self.final_conv(x)
        x = self.final_act(x)
        x = self.output_conv(x)
        x = self.final_output(x)
        # print(f"[Output] final x shape: {x.shape}")
        return x
# --- Modified DiffusionTrainer (no changes needed beyond existing ones) ---
class BigDiffusionTrainer:
    def __init__(self, model, diffusion, dataloader,val_dataloader, optimizer_statement, sample_dir="samples", lr=1e-4, epochs=100, device=None,
                 image_resolution=128, loss_scaling = False):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        self.epochs = epochs
        self.sample_dir = sample_dir
        os.makedirs(self.sample_dir, exist_ok=True)
        self.image_resolution = image_resolution

        self.loader = dataloader
        self.val_loader = val_dataloader

        self.optimizer = eval(optimizer_statement,globals(),locals())
        self.losses = []
        self.val_losses = []
        self.scaler = BF16LossScaler()
        self.loss_scaling  = loss_scaling
        self.logging_steps = 100
        self.global_step = 0
        self.logged_losses = {} # Use a dict to map step to loss

    def train(self):
        """
        Trains the diffusion model with step-based loss logging.
        """
        # Accumulator for the loss over the logging interval
        accumulated_loss = 0.0
        
        # Set the model to training mode
        self.model.train()

        print("🚀 Starting training...")
        for epoch in range(self.epochs):
            # Use tqdm for a progress bar, showing the current epoch
            pbar = tqdm(self.loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for images in pbar:
                # Move data to the correct device and dtype
                images = images.to(self.diffusion.dtype).to(self.device) 
                
                # Sample timesteps
                t = torch.randint(0, self.diffusion.timesteps, (images.size(0),), device=self.device)

                # Calculate loss
                loss = self.diffusion.p_losses(images, t)

                # Backpropagation
                self.optimizer.zero_grad()
                if self.loss_scaling:
                    # Mixed precision training
                    scaled_loss = self.scaler.scale_loss(loss)
                    scaled_loss.backward()
                    self.scaler.unscale_grads(self.model)
                    self.scaler.step(self.optimizer)
                else:
                    # Standard training
                    loss.backward()
                    self.optimizer.step()

                # --- LLM-Style Loss Tracking ---
                accumulated_loss += loss.item()
                self.global_step += 1

                # Check if it's time to log the average loss
                if self.global_step % self.logging_steps == 0:
                    # Calculate the average loss over the last `logging_steps`
                    avg_loss = accumulated_loss / self.logging_steps
                    
                    # Log to console and update the progress bar
                    print(f"[Step {self.global_step}] Average Loss: {avg_loss:.4f}")
                    pbar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})
                    
                    # Store the loss for later plotting or analysis
                    self.losses.append(avg_loss)

                    
                    # Reset the accumulator
                    accumulated_loss = 0.0
                if self.global_step % 1000 == 0:
                    val_loss = self.validate(self.global_step)
                    self.val_losses.append(val_loss)
            # --- End-of-Epoch Actions ---
            # You can still perform actions like saving samples at the end of each epoch
            print(f"Epoch {epoch+1} finished. Saving samples...")
            self.save_samples(epoch, n_samples=16)

    print("✅ Training complete.")
    @torch.no_grad()
    def save_samples(self, epoch, n_samples=64):
        samples = self.diffusion.sample((n_samples, 3, self.image_resolution, self.image_resolution), self.device)
        samples = (samples.clamp(-1, 1) + 1) / 2
        vutils.save_image(samples, f"{self.sample_dir}/epoch_{epoch}.png", nrow=8)

    @torch.no_grad()
    def validate(self, step):
        self.model.eval() # Set model to evaluation mode
        val_loss_val = 0.0
        for images in tqdm(self.val_loader, desc=f"Step {step} [Validate]", leave=False):
            images = images.to(self.diffusion.dtype).to(self.device)
            t = torch.randint(0, self.diffusion.timesteps, (images.size(0),), device=self.device)
            loss = self.diffusion.p_losses(images, t)
            val_loss_val += loss.item()
        
        avg_val_loss = val_loss_val / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
# --- Original GaussianDiffusion (with minor adjustment for model input) ---
class BigGaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02,dtype=torch.bfloat16):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.dtype = dtype
        self.register_schedule(beta_start, beta_end)

    def register_schedule(self, beta_start, beta_end):
        betas = torch.linspace(beta_start, beta_end, self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        self.register_buffer("betas", betas.to(self.dtype))
        self.register_buffer("alphas", alphas.to(self.dtype))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(self.dtype))
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod.to(self.dtype))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod.to(self.dtype))
        # Added for sampling (needed in the reverse process calculation)
        self.posterior_variance = self.betas # Simplified variance

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        # Use .to(x0.device) to ensure tensors are on the same device
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None].to(x0.device)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None].to(x0.device)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def p_losses(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        # Pass the raw timestep t to the adapted model's time_mlp
        predicted_noise = self.model(x_noisy, t)
        return torch.functional.F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, shape, device):
        x = torch.randn(shape, device=device)
        x = x.to(self.dtype)
        # Use a tqdm loop for the sampling process
        for t in tqdm(reversed(range(self.timesteps)), desc="Sampling"):
            time_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x, time_tensor) # Pass raw time_tensor

            # Use the noise prediction to estimate x_{t-1}
            # Calculations based on the DDPM paper (reparameterization trick)
            alpha_t = self.alphas[t]
            # alpha_bar_t = self.alphas_cumprod[t]
            # sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
            beta_t = self.betas[t]

            # Mean of the posterior distribution
            mean = (1 / torch.sqrt(alpha_t)) * (x - beta_t / sqrt_one_minus_alpha_bar_t * predicted_noise)

            if t > 0:
                # Add noise for the next step
                # Variance of the posterior
                posterior_variance_t = self.posterior_variance[t]
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = mean # No noise at the last step (t=0)

        return x
