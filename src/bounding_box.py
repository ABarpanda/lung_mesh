import numpy as np
import torch
import cv2
import os
from lung_mask import imwrite, segment_lungs

IMAGE_PATH = "dataset/JSRT_images/JPCNN060.png"
MODEL_PATH = "models/unet_lung_seg.pt"
INPUT_SIZE = 512
THRESHOLD = 0.5

def load_model(model_path: str = MODEL_PATH, device: str = None) -> torch.nn.Module:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # weights_only=False is required: the file contains the whole onnx2torch graph module
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device).eval()
    return model

def get_mask(image_path, model_path=MODEL_PATH, threshold=THRESHOLD):
    model = load_model(model_path)
    _, mask, _ = segment_lungs(image_path, model, threshold)
    
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
        
    out_path = f"{os.path.splitext(os.path.basename(image_path))[0]}_mask.png"
    # print(f"Saved results for '{name}' in {}")
    imwrite(out_path, mask)
    return mask

def draw_bboxes(image, boxes, color=(0, 255, 0), thickness=3):
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness)
    return bgr

def mask_to_bbox(mask):
    """Returns (x1, y1, x2, y2) around all white pixels, or None if the mask is empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def mask_to_lung_bboxes(mask, min_area_frac=0.01):
    """Returns up to 2 boxes (x1, y1, x2, y2), sorted left-to-right in the image."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    min_area = min_area_frac * mask.shape[0] * mask.shape[1]

    comps = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] >= min_area]
    comps = sorted(comps, reverse=True)[:2]          # keep the 2 largest blobs

    boxes = []
    for _, i in comps:
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        boxes.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    return sorted(boxes, key=lambda b: b[0])         # leftmost in the image first

def draw(image_path, mode=1, out_dir="results"):
    """
    Draw bounding boxes around lungs in the image.  
    
    Input:
        image_path: path to the input image  
        mode=1: draw a single box around both lungs 
        mode=2: draw per-lung boxes
        out_dir: directory to save the output image  
    
    Output:
        Saves the image with bounding boxes in the out_dir.
    """
    name = os.path.splitext(os.path.basename(image_path))[0]
    prob, mask, image = segment_lungs(image_path, model, THRESHOLD)
    if mode==1:
        both_box = mask_to_bbox(mask)
        print("Both lungs:", both_box)
        single_box = draw_bboxes(image, [both_box])
        imwrite(os.path.join(out_dir, f"{name}_bbox.png"), single_box)
    elif mode==2:
        lung_boxes = mask_to_lung_bboxes(mask)
        print("Per lung  :", lung_boxes)
        boxed = draw_bboxes(image, lung_boxes)
        imwrite(os.path.join(out_dir, f"{name}_bbox.png"), boxed)
    else:
        raise ValueError(f"Unknown mode {mode}. Options: 1=per lung, 2=both lungs")

if __name__=="__main__":
    IMAGE_PATH = "dataset/JSRT_images/JPCNN003.png"
    model = load_model(MODEL_PATH)
    prob, mask, image = segment_lungs(IMAGE_PATH, model, THRESHOLD)
    draw(IMAGE_PATH, mode=1)
