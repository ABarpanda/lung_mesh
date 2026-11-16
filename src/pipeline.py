import os
import time
import numpy as np
from PIL import Image
from center import align
from lung_mask import get_outputs
from scale import normalize_lung_roi, imwrite
from contrast import perform_contrast_improvement
from bounding_box import segment_lungs, mask_to_bbox, load_model

k = 2
THRESHOLD = 0.8
# IMAGE_PATH = r"D:\Projects\lung_mesh\dataset\JSRT_images\JPCLN095.png"
INPUT_DIR = r"D:\Projects\lung_mesh\RSNA_jpeg"
OUTPUT_DIR = r"D:\Projects\lung_mesh\processed_RSNA_output"
MODEL_PATH = "models/unet_lung_seg.pt"

def pipeline(IMAGE_PATH, OUTPUT_DIR, model=None, model_path=MODEL_PATH, k=2, THRESHOLD=0.8):
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
        imwrite(f"{OUTPUT_DIR}/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}.jpeg", normalized_image)
        print("Normalized image shape:", normalized_image.shape)
        print("Affine transformation matrix:\n", affine_matrix)
        with open(f"image_scale.csv", "a") as f:
            f.write(f"{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}, {affine_matrix[0, 0]}, {min_error_index}\n")
    else:
        print(f"No lungs detected in the image {IMAGE_PATH}.")
        improved_image = perform_contrast_improvement(IMAGE_PATH, SAVE_PATH=f"{IMAGE_PATH}", save=False)
        pipeline(f"{IMAGE_PATH}", OUTPUT_DIR, model=model, model_path=model_path, k=k, THRESHOLD=THRESHOLD)

if __name__ == "__main__":
    start_time = time.time()
    model = load_model(MODEL_PATH)
    count = 0
    existing_outputs = {
        os.path.splitext(f)[0]
        for f in os.listdir(OUTPUT_DIR)
    }
    for filename in os.listdir(INPUT_DIR):
        stem, ext = os.path.splitext(filename)
        if count > 0:
            count -= 1
            continue
        skipped=0
        if stem in existing_outputs:
            skipped+=1
            continue
        print(f"Skipped {skipped} already processed files.")
        full_image_path = os.path.join(INPUT_DIR, filename)
        print(f"\n\n\nProcessing {stem}...")
        start = time.time()
        pipeline(
            full_image_path,
            OUTPUT_DIR,
            model=model,
            model_path=MODEL_PATH,
            k=k,
            THRESHOLD=THRESHOLD,
        )
        elapsed = time.time() - start
        print(f"Time taken for {stem}: {elapsed:.2f} seconds")
        with open("processing_times.csv", "a") as f:
            f.write(f"{stem}, {elapsed} seconds\n")
    end_time = time.time()
    print(" ")
    print(" ")
    print(" ")
    print(f"Minimum processing time: {min([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')])} seconds")
    print(f"Maximum processing time: {max([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')])} seconds")
    print(f"Total processing time: {sum([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')])} seconds")
    print(f"Average processing time: {sum([float(line.split(', ')[1].split(' ')[0]) for line in open('processing_times.csv')]) / len(open('processing_times.csv').readlines())} seconds")