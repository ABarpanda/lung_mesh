import os
import time
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

def pipeline(IMAGE_PATH, model=None, model_path=MODEL_PATH, k=2, THRESHOLD=0.8):
    image = Image.open(IMAGE_PATH).convert("L")
    array = np.asarray(image)

    # Horizontal alignment
    aligned, min_error_index = align(array, k=k, save_path=f"./results/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_aligned.png")

    # Lung masks
    get_outputs(f"./results/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_aligned.png", model=model, model_path=model_path, threshold=THRESHOLD, save=False)

    # Bounding box
    model = model or load_model(MODEL_PATH)
    prob, mask, image = segment_lungs(IMAGE_PATH, model)
    bbox = mask_to_bbox(mask)
    if bbox is not None:
        normalized_image, affine_matrix = normalize_lung_roi(image, bbox, target_height_ratio=3.0 / 4.0)
        imwrite(f"./output/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}.png", normalized_image)
        print("Normalized image shape:", normalized_image.shape)
        print("Affine transformation matrix:\n", affine_matrix)
        with open(f"image_scale.csv", "a") as f:
            f.write(f"{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}, {affine_matrix[0, 0]}, {min_error_index}\n")
    else:
        print("No lungs detected in the mask.")

if __name__ == "__main__":
    start_time = time.time()
    model = load_model(MODEL_PATH)
    count = 4
    for IMAGE_PATH in os.listdir(r"D:\Projects\lung_mesh\dataset\JSRT_images"):
        if count > 0:
            count-=1
            continue

        print(" ")
        print(" ")
        print(" ")
        start = time.time()
        print(f"Processing {os.path.splitext(os.path.basename(IMAGE_PATH))[0]}...")
        if IMAGE_PATH.endswith(".png"):
            IMAGE_PATH = os.path.join(r"D:\Projects\lung_mesh\dataset\JSRT_images", IMAGE_PATH)
        pipeline(IMAGE_PATH, model=model, model_path=MODEL_PATH, k=k, THRESHOLD=THRESHOLD)
        end = time.time()
        print(f"Time taken for {os.path.splitext(os.path.basename(IMAGE_PATH))[0]}: {end - start} seconds")
        with open("processing_times.csv", "a") as f:
            f.write(f"{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}, {end - start} seconds\n")
    end_time = time.time()
    print(" ")
    print(" ")
    print(" ")
    print(f"Minimum processing time: {min([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')])} seconds")
    print(f"Maximum processing time: {max([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')])} seconds")
    print(f"Total processing time: {sum([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')])} seconds")
    print(f"Average processing time: {sum([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')]) / len(open('processing_times.csv').readlines())} seconds")