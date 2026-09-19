import os
import glob
import torch
import matplotlib.pyplot as plt

checkpoints_dir = "dataset/diffusion_outputs/checkpoints"

# Find all checkpoint files matching the naming pattern
ckpt_files = glob.glob(os.path.join(checkpoints_dir, "model_epoch_*.pt"))

if not ckpt_files:
    raise FileNotFoundError(f"No checkpoint files found in '{checkpoints_dir}'")

epochs = []
losses = []

for filepath in ckpt_files:
    # Load metadata on CPU without loading heavy weights into memory
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu")
    
    epoch = checkpoint.get("epoch", None)
    loss = checkpoint.get("loss", None)

    if epoch is not None and loss is not None:
        epochs.append(epoch)
        losses.append(loss)

# Sort strictly by epoch order
epochs, losses = zip(*sorted(zip(epochs, losses)))

loss = []
# Display extracted numeric data
print(f"Extracted {len(epochs)} checkpoints:")
for ep, ls in zip(epochs, losses):
    print(f"  Epoch {ep:03d}: Loss = {ls:.6f}")
    loss.append(ls)

# Plotting on logarithmic scale
plt.figure(figsize=(9, 5), dpi=120)
plt.plot(epochs, losses, marker="o", linestyle="-", linewidth=2, label="Mean Loss")

# plt.yscale("log")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (Log Scale)", fontsize=12)
plt.title("Diffusion Training Loss Across Checkpoints", fontsize=14, fontweight="bold")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()