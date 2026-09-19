import os
import math
import time
import argparse
import torch
import torch.nn as nn
from torchvision import utils

# ==============================================================================
# 1. Model Architecture (Must match your training script exactly)
# ==============================================================================

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
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ConvBlock(nn.Module):
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
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + residual


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

        self.down1 = ConvBlock(in_channels, base_dim, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base_dim, base_dim * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.mid1 = ConvBlock(base_dim * 2, base_dim * 4, time_dim)
        self.mid2 = ConvBlock(base_dim * 4, base_dim * 2, time_dim)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upconv2 = ConvBlock(base_dim * 4, base_dim, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upconv1 = ConvBlock(base_dim * 2, base_dim, time_dim)

        self.out_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        m = self.mid1(self.pool2(x2), t_emb)
        m = self.mid2(m, t_emb)
        u2 = self.up2(m)
        u2 = self.upconv2(torch.cat([u2, x2], dim=1), t_emb)
        u1 = self.up1(u2)
        u1 = self.upconv1(torch.cat([u1, x1], dim=1), t_emb)
        return self.out_conv(u1)

# ==============================================================================
# 2. Diffusion Sampler (DDPM + Fast DDIM)
# ==============================================================================

class DiffusionSampler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device: str = "cuda"):
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    @torch.no_grad()
    def sample_ddpm(self, model: nn.Module, num_samples: int, image_size: int, channels: int = 3, return_intermediates: bool = False):
        """Full 1,000-step standard DDPM reverse diffusion."""
        model.eval()
        x = torch.randn((num_samples, channels, image_size, image_size), device=self.device)
        intermediates = []

        print(f"Sampling {num_samples} images with DDPM (1,000 steps)...")
        start_time = time.time()

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t)

            alpha = self.alpha[i]
            alpha_hat = self.alpha_hat[i]
            beta = self.beta[i]

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alpha)) * (x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

            if return_intermediates and i % 100 == 0:
                intermediates.append(((x.clamp(-1, 1) + 1) / 2).cpu())

            if (1000 - i) % 200 == 0:
                print(f"Step [{1000 - i:04d}/1000] completed...")

        duration = time.time() - start_time
        print(f"DDPM generation complete in {duration:.2f}s ({num_samples / duration:.2f} img/s)")

        x = (x.clamp(-1, 1) + 1) / 2
        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def sample_ddim(self, model: nn.Module, num_samples: int, image_size: int, ddim_steps: int = 50, channels: int = 3):
        """Accelerated deterministic DDIM sampling (20-50 steps)."""
        model.eval()
        times = torch.linspace(0, self.timesteps - 1, steps=ddim_steps).long().to(self.device)
        times = list(reversed(times.tolist()))

        x = torch.randn((num_samples, channels, image_size, image_size), device=self.device)

        print(f"Sampling {num_samples} images with fast DDIM ({ddim_steps} steps)...")
        start_time = time.time()

        for idx, t_val in enumerate(times):
            t = torch.full((num_samples,), t_val, device=self.device, dtype=torch.long)
            predicted_noise = model(x, t)

            alpha_hat = self.alpha_hat[t_val]

            # Reconstruct estimated x_0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            # Trajectory step
            if idx < len(times) - 1:
                prev_t_val = times[idx + 1]
                alpha_hat_prev = self.alpha_hat[prev_t_val]
                direction = torch.sqrt(1.0 - alpha_hat_prev) * predicted_noise
                x = torch.sqrt(alpha_hat_prev) * pred_x0 + direction
            else:
                x = pred_x0

        duration = time.time() - start_time
        print(f"DDIM generation complete in {duration:.2f}s ({num_samples / duration:.2f} img/s)")

        return (x.clamp(-1, 1) + 1) / 2

# ==============================================================================
# 3. Main CLI & Execution Logic
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate images from a trained Diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="dataset/diffusion_outputs/checkpoints/model_epoch_050.pt",
                        help="Path to trained checkpoint file (.pt)")
    parser.add_argument("--out_dir", type=str, default="./generated_results",
                        help="Directory where output images will be saved")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Total number of images to generate")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Resolution of generated images (must match training resolution)")
    parser.add_argument("--method", type=str, choices=["ddpm", "ddim"], default="ddim",
                        help="Sampling algorithm: 'ddpm' (1000 steps) or 'ddim' (fast)")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="Steps to run if using DDIM")
    parser.add_argument("--save_steps_strip", action="store_true",
                        help="Save intermediate steps showing the denoising progression (DDPM mode)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: '{args.checkpoint}'")

    print(f"[Device] {device.upper()}")
    print(f"[Loading] Checkpoint: {args.checkpoint}")

    # Initialize model and load weights
    model = SimpleUNet(in_channels=3, out_channels=3, base_dim=64).to(device)
    ckpt_data = torch.load(args.checkpoint, map_location=device)
    
    if "model_state_dict" in ckpt_data:
        model.load_state_dict(ckpt_data["model_state_dict"])
        epoch_trained = ckpt_data.get("epoch", "Unknown")
        print(f"[Success] Loaded model weights trained up to Epoch: {epoch_trained}")
    else:
        model.load_state_dict(ckpt_data)
        print("[Success] Loaded raw model weights directly.")

    sampler = DiffusionSampler(timesteps=1000, device=device)

    # Run Sampling
    if args.method == "ddpm":
        if args.save_steps_strip:
            samples, intermediates = sampler.sample_ddpm(
                model, num_samples=args.num_images, image_size=args.image_size, return_intermediates=True
            )
            # Intermediate progression strip for the first generated sample
            strip = torch.stack([step[0] for step in intermediates])
            strip_path = os.path.join(args.out_dir, "denoising_progression.png")
            utils.save_image(strip, strip_path, nrow=len(intermediates))
            print(f"[Saved] Denoising sequence strip: {strip_path}")
        else:
            samples = sampler.sample_ddpm(model, num_samples=args.num_images, image_size=args.image_size)
    else:
        samples = sampler.sample_ddim(
            model, num_samples=args.num_images, image_size=args.image_size, ddim_steps=args.ddim_steps
        )

    # Save Grid
    grid_path = os.path.join(args.out_dir, f"generated_grid_{args.method}.png")
    grid_nrow = int(math.sqrt(args.num_images))
    utils.save_image(samples, grid_path, nrow=grid_nrow)
    print(f"[Saved] Combined Grid: {grid_path}")

    # Save Individual Images
    individual_dir = os.path.join(args.out_dir, "individual_samples")
    os.makedirs(individual_dir, exist_ok=True)
    for i in range(args.num_images):
        single_path = os.path.join(individual_dir, f"sample_{i+1:03d}.png")
        utils.save_image(samples[i], single_path)
    
    print(f"[Saved] {args.num_images} individual images in: {individual_dir}")

if __name__ == "__main__":
    main()