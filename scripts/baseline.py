import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from PIL import Image

class FlatImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        self.image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(root_dir)
            for f in files
            if os.path.splitext(f.lower())[1] in valid_exts
        ]
        if not self.image_paths:
            raise FileNotFoundError(f"No valid image files found in '{root_dir}'")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

# ------------------------------------------------------------------------------
# 1. Sinusoidal Time Embedding & Building Blocks
# ------------------------------------------------------------------------------

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time_step: torch.Tensor) -> torch.Tensor:
        device = time_step.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time_step[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm, GELU, and Time-Embedding injection."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.act = nn.GELU()

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = self.residual_conv(x)
        h = self.act(self.norm1(self.conv1(x)))
        # Project and broadcast time conditioning into spatial dims [B, C, 1, 1]
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + residual

# ------------------------------------------------------------------------------
# 2. Lightweight U-Net Architecture
# ------------------------------------------------------------------------------

class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_dim: int = 64):
        super().__init__()
        time_dim = base_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder (Downsampling)
        self.down1 = ConvBlock(in_channels, base_dim, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base_dim, base_dim * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1 = ConvBlock(base_dim * 2, base_dim * 4, time_dim)
        self.mid2 = ConvBlock(base_dim * 4, base_dim * 2, time_dim)

        # Decoder (Upsampling)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upconv2 = ConvBlock(base_dim * 4, base_dim, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upconv1 = ConvBlock(base_dim * 2, base_dim, time_dim)

        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)

        # Down
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)

        # Mid
        m = self.mid1(self.pool2(x2), t_emb)
        m = self.mid2(m, t_emb)

        # Up with skip-connections
        u2 = self.up2(m)
        u2 = self.upconv2(torch.cat([u2, x2], dim=1), t_emb)
        u1 = self.up1(u2)
        u1 = self.upconv1(torch.cat([u1, x1], dim=1), t_emb)

        return self.out_conv(u1)

# ------------------------------------------------------------------------------
# 3. Diffusion Forward & Reverse Process Container
# ------------------------------------------------------------------------------

class Diffusion:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cuda"):
        self.timesteps = timesteps
        self.device = device

        # Precompute diffusion schedule constants
        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Forward diffusion: adds noise to initial image x_0 at step t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, num_samples: int, image_size: int, channels: int = 3):
        """Reverse diffusion: generates samples from pure Gaussian noise."""
        model.eval()
        x = torch.randn((num_samples, channels, image_size, image_size), device=self.device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t)

            alpha = self.alpha[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.beta[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # DDPM mean formulation: mu_theta(x_t, t)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # Rescale [-1, 1] to [0, 1] for saving
        x = (x.clamp(-1, 1) + 1) / 2
        return x

# ------------------------------------------------------------------------------
# 4. Verbose Training Loop
# ------------------------------------------------------------------------------

def train():
    # Hyperparameters
    IMG_SIZE = 128
    BATCH_SIZE = 8
    START_EPOCH = 490
    EPOCHS = 1000
    OUTPUT_DIR = "dataset/diffusion_outputs"
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoints", f"model_epoch_{START_EPOCH}.pt")
    LR = 2e-4
    TIMESTEPS = 1000
    DATASET_PATH = "dataset/JSRT_images"
    OUTPUT_DIR = "dataset/diffusion_outputs"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)

    # Data Pipeline (normalized to [-1, 1])
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Maps [0, 1] -> [-1, 1]: (x - 0.5) / 0.5 = 2x - 1
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"Loading dataset from: {DATASET_PATH}...")
    dataset = FlatImageDataset(root_dir=DATASET_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
    total_batches = len(dataloader)

    print(f"Device: {DEVICE}")
    print(f"Total Images: {len(dataset)} | Batches/Epoch: {total_batches} | Image Resolution: {IMG_SIZE}x{IMG_SIZE}")

    # Initialize Model, Optimizer, and Diffusion Engine
    model = SimpleUNet(in_channels=3, out_channels=3, base_dim=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    diffusion = Diffusion(timesteps=TIMESTEPS, device=DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {num_params:,}\n" + "=" * 95)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[INFO] Found checkpoint: {CHECKPOINT_PATH}. Loading state...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        last_epoch = checkpoint["epoch"]
        START_EPOCH = last_epoch + 1
        global_step = last_epoch * total_batches
        
        print(f"[INFO] Successfully resumed from Epoch {last_epoch}. "
              f"Continuing from Epoch {START_EPOCH} (Step: {global_step:06d}) to {EPOCHS}.\n")
    else:
        print(f"[INFO] No checkpoint found at {CHECKPOINT_PATH}. Starting fresh from Epoch 1.\n")
        global_step = 0

    print("=" * 95)

    for epoch in range(START_EPOCH, EPOCHS + 1):
        epoch_start_time = time.time()
        running_loss = 0.0
        running_grad_norm = 0.0

        for batch_idx, (images, _) in enumerate(dataloader):
            batch_start = time.time()
            images = images.to(DEVICE)

            # Sample random timesteps uniformly
            t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=DEVICE).long()
            
            # Forward process
            x_noisy, noise = diffusion.q_sample(images, t)
            
            # Predict added noise
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norm for convergence tracking
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_grad_norm += grad_norm.item()
            global_step += 1
            batch_dur = time.time() - batch_start

            # Quantitative logging every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                avg_b_loss = running_loss / (batch_idx + 1)
                avg_b_grad = running_grad_norm / (batch_idx + 1)
                t_mean = t.float().mean().item()
                t_min = t.min().item()
                t_max = t.max().item()

                print(
                    f"Epoch [{epoch:03d}/{EPOCHS:03d}] | "
                    f"Batch [{batch_idx + 1:04d}/{total_batches:04d}] | "
                    f"Step: {global_step:06d} | "
                    f"MSE Loss: {loss.item():.5f} (Avg: {avg_b_loss:.5f}) | "
                    f"|Grad|: {grad_norm.item():.4f} | "
                    f"t-stats [min: {t_min:3d}, avg: {t_mean:5.1f}, max: {t_max:3d}] | "
                    f"Speed: {images.size(0) / batch_dur:.1f} img/s"
                )

        epoch_dur = time.time() - epoch_start_time
        epoch_loss = running_loss / total_batches
        epoch_grad = running_grad_norm / total_batches

        print("-" * 95)
        print(
            f"EPOCH {epoch:03d} SUMMARY: "
            f"Mean Loss: {epoch_loss:.6f} | "
            f"Mean Grad Norm: {epoch_grad:.4f} | "
            f"Duration: {epoch_dur:.2f}s"
        )
        print("-" * 95)

        # Generate sample images after each 5 epoch
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Sampling validation grid at Epoch {epoch}...")
            sample_start = time.time()
            samples = diffusion.p_sample_loop(model, num_samples=4, image_size=IMG_SIZE)
            sample_path = os.path.join(OUTPUT_DIR, "samples", f"sample_epoch_{epoch:03d}.png")
            utils.save_image(samples, sample_path, nrow=4)
            print(f"Grid saved to {sample_path} (Generation took {time.time() - sample_start:.2f}s)\n")

        # Save checkpoint periodically
        if epoch % 5 == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"model_epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}\n")

if __name__ == "__main__":
    train()