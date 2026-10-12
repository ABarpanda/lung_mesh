import os
import numpy as np
from PIL import Image
from center import align
from lung_mask import get_outputs
from scale import normalize_lung_roi, imwrite
from bounding_box import segment_lungs, mask_to_bbox, load_model

k = 2
THRESHOLD = 0.8
IMAGE_PATH = r"D:\Projects\lung_mesh\dataset\JSRT_images\JPCLN095.png"
MODEL_PATH = "models/unet_lung_seg.pt"

image = Image.open(IMAGE_PATH).convert("L")
array = np.asarray(image)

# Horizontal alignment
aligned = align(array, k=k, save_path=f"./results/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_aligned.png")

# Lung masks
get_outputs(f"./results/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_aligned.png", model_path=MODEL_PATH, threshold=THRESHOLD)

# Bounding box
model = load_model(MODEL_PATH)
prob, mask, image = segment_lungs(IMAGE_PATH, model)
bbox = mask_to_bbox(mask)
if bbox is not None:
    normalized_image, affine_matrix = normalize_lung_roi(image, bbox)
    imwrite(f"./output/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}.png", normalized_image)
    print("Normalized image shape:", normalized_image.shape)
    print("Affine transformation matrix:\n", affine_matrix)
else:
    print("No lungs detected in the mask.")