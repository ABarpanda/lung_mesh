
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

@torch.no_grad()
def evaluate_candidates(image, thetas, rhos, chunk_size=16,
                        min_overlap=0.5):
    """
    Evaluate corresponding (theta, rho) candidate pairs.
    image: (1, H, W), CUDA float32
    thetas and rhos: equally sized 1D tensors.
    """

    _, H, W = image.shape
    device = image.device
    C = 1

    yc = (H - 1) / 2
    xc = (W - 1) / 2

    ys = torch.arange(H, device=device, dtype=torch.float32) - yc
    xs = torch.arange(W, device=device, dtype=torch.float32) - xc

    Y, X = torch.meshgrid(ys, xs, indexing="ij")

    cos_all = torch.cos(thetas)
    sin_all = torch.sin(thetas)

    results = []

    for start in range(0, len(thetas), chunk_size):

        end = min(start + chunk_size, len(thetas))

        ct = cos_all[start:end, None, None]
        st = sin_all[start:end, None, None]
        ro = rhos[start:end, None, None]

        # Distance from each pixel to each line
        dist = X * ct + Y * st - ro

        # Reflection
        xr = xc + X - 2 * dist * ct
        yr = yc + Y - 2 * dist * st

        valid = (
            (xr >= 0) & (xr <= W - 1) &
            (yr >= 0) & (yr <= H - 1)
        )

        grid = torch.stack((
            2 * xr / (W - 1) - 1,
            2 * yr / (H - 1) - 1
        ), dim=-1)

        B = end - start

        reflected = F.grid_sample(
            image.unsqueeze(0).expand(B, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        diff = (reflected - image.unsqueeze(0)).square()
        diff.mul_(valid.unsqueeze(1))

        count = valid.sum(dim=(1, 2)).float()

        mse = diff.sum(dim=(1, 2, 3)) / count.clamp_min(1)

        mse[count < min_overlap * H * W] = float("inf")

        results.append(mse)

    return torch.cat(results)


@torch.no_grad()
def fast_symmetry_search(
    image,
    top_k=5,
    coarse_size=128,
    fine_size=256,
    coarse_angles=90,
    chunk_size=16,
):
    """
    image: (1, H, W), CUDA float32

    Returns:
        line_angle_deg
        y_intercept
        minimum_error
        rho_original
    """

    assert image.is_cuda
    assert image.ndim == 3 and image.shape[0] == 1

    _, H, W = image.shape
    device = image.device

    # Preserve aspect ratio
    scale_coarse = coarse_size / max(H, W)
    scale_fine = fine_size / max(H, W)

    def resize_image(scale):
        nh = max(2, round(H * scale))
        nw = max(2, round(W * scale))

        resized = F.interpolate(
            image.unsqueeze(0),
            size=(nh, nw),
            mode="bilinear",
            align_corners=False
        )

        return resized.squeeze(0)

    small = resize_image(scale_coarse)
    medium = resize_image(scale_fine)

    # ---------------------------------
    # STAGE 1: Coarse search
    # ---------------------------------

    _, h1, w1 = small.shape

    max_rho = math.hypot((w1 - 1) / 2, (h1 - 1) / 2)

    theta1 = torch.arange(
        coarse_angles, device=device, dtype=torch.float32
    ) * (math.pi / coarse_angles)

    rho1 = torch.arange(
        -max_rho, max_rho + 0.01, 2.0,
        device=device
    )

    T1, R1 = torch.meshgrid(theta1, rho1, indexing="ij")

    t1 = T1.flatten()
    r1 = R1.flatten()

    error1 = evaluate_candidates(
        small, t1, r1, chunk_size
    )

    k = min(top_k, len(error1))

    best_values, best_indices = torch.topk(
        error1, k=k, largest=False
    )

    # ---------------------------------
    # STAGE 2: Fine search
    # ---------------------------------

    theta_candidates = []
    rho_candidates = []

    angle_offsets = torch.arange(
        -8, 9, device=device, dtype=torch.float32
    ) * (math.pi / 720)  # 0.25 degrees

    rho_offsets = torch.arange(
        -6, 7, device=device, dtype=torch.float32
    )

    scale_ratio = scale_fine / scale_coarse

    for idx in best_indices:

        base_theta = t1[idx]
        base_rho = r1[idx] * scale_ratio

        dt, dr = torch.meshgrid(
            angle_offsets, rho_offsets, indexing="ij"
        )

        theta_candidates.append(
            torch.remainder(base_theta + dt.flatten(), math.pi)
        )

        rho_candidates.append(
            base_rho + dr.flatten()
        )

    t2 = torch.cat(theta_candidates)
    r2 = torch.cat(rho_candidates)

    error2 = evaluate_candidates(
        medium, t2, r2, chunk_size
    )

    best_idx = torch.argmin(error2)

    best_theta = t2[best_idx]
    best_rho = r2[best_idx]

    min_error = error2[best_idx].item()

    # Convert rho from resized to original image coordinates
    rho_original = best_rho.item() / scale_fine

    theta = best_theta.item()

    # Line direction
    line_angle = (math.degrees(theta) + 90) % 180

    # Original image coordinate system
    xc = (W - 1) / 2
    yc = (H - 1) / 2

    if abs(math.sin(theta)) < 1e-6:
        y_intercept = float("inf")
    else:
        y_intercept = (
            yc +
            (rho_original + xc * math.cos(theta))
            / math.sin(theta)
        )

    return line_angle, y_intercept, min_error, rho_original


def display_symmetry_overlay(image, angle_deg, y_intercept):
    """
    Display the detected symmetry axis over the original image.

    image: Original grayscale image as a NumPy array or CPU/GPU tensor.
    angle_deg: Detected line direction in degrees.
    y_intercept: Detected y-intercept in image coordinates.
    """

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze().numpy()

    image = np.asarray(image)

    H, W = image.shape

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(image, cmap="gray")

    # Convert line angle to slope
    if abs(np.sin(np.radians(angle_deg))) < 1e-6:
        # Horizontal line
        x = np.array([0, W - 1])
        y = np.full(2, y_intercept)

    elif abs(np.cos(np.radians(angle_deg))) < 1e-6:
        # Vertical line
        x = np.full(2, W / 2)
        y = np.array([0, H - 1])

    else:
        slope = np.tan(np.radians(angle_deg))

        x = np.array([0, W - 1])
        y = y_intercept + slope * x

        # Clip the line to image boundaries
        valid = (y >= 0) & (y < H)

        if not np.all(valid):
            points = []

            for xx, yy in zip(x, y):
                if 0 <= yy < H:
                    points.append((xx, yy))

            for yy in [0, H - 1]:
                xx = (yy - y_intercept) / slope
                if 0 <= xx < W:
                    points.append((xx, yy))

            if len(points) >= 2:
                points = np.array(points)
                x = points[:2, 0]
                y = points[:2, 1]

    ax.plot(
        x, y,
        color="red",
        linewidth=2,
        label=f"Detected axis: {angle_deg:.2f}°"
    )

    ax.set_title(
        f"Detected Symmetry Axis\n"
        f"Angle: {angle_deg:.2f}° | "
        f"Y-intercept: {y_intercept:.2f}"
    )

    ax.legend(loc="upper right")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = r"D:\Projects\lung_mesh\dataset\RSNA_jpeg\0a2f6cf6-1f45-44c8-bcf0-98a3b466b597.jpeg"

    img = Image.open(path).convert("L")

    tensor = (
        torch.from_numpy(np.array(img, copy=True))
        .float()
        .div(255)
        .unsqueeze(0)
        .cuda()
    )

    # Warm up CUDA before timing
    _ = torch.cuda.synchronize()

    import time
    start = time.perf_counter()

    angle, intercept, error, rho = fast_symmetry_search(
        tensor,
        top_k=5,
        coarse_size=128,
        fine_size=256,
        chunk_size=16
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"Angle: {angle:.3f}")
    print(f"Y intercept: {intercept:.3f}")
    print(f"Rho: {rho:.3f}")
    print(f"Error: {error:.7f}")
    print(f"Time: {elapsed:.3f} seconds")

    # Display the symmetry overlay
    display_symmetry_overlay(img, angle, intercept)