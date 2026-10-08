"""
Lung segmentation inference (PyTorch version of the Kaggle U-Net notebook).
"""
import os
import cv2
import numpy as np
import torch

MODEL_PATH = "models/unet_lung_seg.pt"
INPUT_SIZE = 512
THRESHOLD = 0.8

# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def load_model(model_path: str = MODEL_PATH, device: str = None) -> torch.nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False is required: the file contains the whole onnx2torch graph module
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model

# ----------------------------------------------------------------------
# Pre-processing (same as the notebook's test_load_image)
# ----------------------------------------------------------------------
def read_gray(image_path: str) -> np.ndarray:
    # np.fromfile + imdecode also works with non-ASCII Windows paths
    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img

def preprocess(img_gray: np.ndarray) -> torch.Tensor:
    x = img_gray / 255.0                              # scale to [0, 1]
    x = cv2.resize(x, (INPUT_SIZE, INPUT_SIZE))       # 512x512
    x = x.astype(np.float32)[None, :, :, None]        # (1, H, W, 1)  -> NHWC like Keras
    return torch.from_numpy(x)

# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_probability(model: torch.nn.Module, img_gray: np.ndarray) -> np.ndarray:
    """Returns the 512x512 float32 probability map in [0, 1]."""
    device = next(model.parameters()).device
    x = preprocess(img_gray).to(device)

    out = model(x).cpu().numpy()

    # The converted model keeps Keras' NHWC layout: (1, 512, 512, 1).
    # Handle an NCHW output (1, 1, 512, 512) too, just in case.
    if out.ndim == 4 and out.shape[-1] != 1 and out.shape[1] == 1:
        out = np.transpose(out, (0, 2, 3, 1))
    return out[0, :, :, 0]

def segment_lungs(image_path: str, model: torch.nn.Module = None, threshold: float = THRESHOLD):
    """
    Returns:
        prob      : (512, 512) float32 probability map
        mask      : (H, W) uint8 binary mask (0/255) at the original image size
        image     : (H, W) uint8 original grayscale image
    """
    model = model or load_model()
    image = read_gray(image_path)
    prob = predict_probability(model, image)

    h, w = image.shape
    prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob_full > threshold).astype(np.uint8) * 255
    return prob, mask, image

def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(bgr)
    color[mask > 0] = (0, 0, 255)  # red (BGR)
    return cv2.addWeighted(bgr, 0.7, color, 0.3, 0)

def imwrite(path: str, img: np.ndarray):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(path)  # unicode-safe on Windows

def get_outputs(image_path, model=None, model_path=MODEL_PATH, threshold=THRESHOLD, out_dir="results"):
    """Saves the mask, overlay and lung-only image. Returns their paths."""
    model = model or load_model(model_path)          # load once, reuse for many images
    prob, mask, image = segment_lungs(image_path, model, threshold)

    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]

    # 1) binary mask (0/255), original size
    mask_path = os.path.join(out_dir, f"{name}_mask.png")
    imwrite(mask_path, mask)

    # 2) overlay: X-ray with the mask in red
    overlay = make_overlay(image, mask)
    overlay_path = os.path.join(out_dir, f"{name}_overlay.png")
    imwrite(overlay_path, overlay)

    # 3) lungs only: everything outside the mask becomes black
    lungs = cv2.bitwise_and(image, image, mask=mask)
    lungs_path = os.path.join(out_dir, f"{name}_lungs.png")
    imwrite(lungs_path, lungs)

    return mask_path, overlay_path, lungs_path

def main():
    get_outputs(r"dataset\JSRT_images\JPCNN003.png")
    # get_outputs(r"results\JPCNN003_aligned.png")

if __name__ == "__main__":
    main()