from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

'''
Bad:
r"RSNA_output/0cc48c9c-0576-4d70-abc0-dd8cdc1eb788.jpeg" - 44
r"RSNA_output/0cb47fe7-14e3-4c3f-9430-71d97ac0a3e5.jpeg" - 34
r"RSNA_output/01a6eaa6-222f-4ea8-9874-bbd89dc1a1ce.jpeg" - 40
r"RSNA_output/1c0a5548-0e11-4793-b48e-97236374c768.jpeg" - 48
r"RSNA_output/1c63da74-f68a-498d-a4b8-fc637227a4fa.jpeg" - 34

Good:
r"RSNA_output/0c861344-c755-4072-ab28-1d29be4949f5.jpeg" - 56
r"RSNA_output/0d8ca6ee-0595-45aa-bedb-1087dee15f23.jpeg" - 64

'''

IMAGE_PATH = r"RSNA_output/0cb47fe7-14e3-4c3f-9430-71d97ac0a3e5.jpeg"

def low_image_contrast(IMAGE_PATH) -> bool:
    image = Image.open(IMAGE_PATH).convert("L")
    array = np.asarray(image).flatten()
    array = array[array > 25]
    # print(np.std(array))
    # mydict = dict(Counter(array))
    # plt.bar(mydict.keys(), mydict.values())
    # plt.show()
    if np.std(array) < 50:
        print("Low contrast image detected.")
        return True
    else: 
        return False

def perform_contrast_improvement(IMAGE_PATH, SAVE_PATH, save=False)->Image.Image:
    if low_image_contrast(IMAGE_PATH):
        print("Performing contrast stretching...")
        image = Image.open(IMAGE_PATH).convert("L") 
        array = np.asarray(image)

        # transformed_array = 255 - np.abs(255 - 2 * array)
        # transformed_image = Image.fromarray(transformed_array.astype(np.uint8))

        low, high = np.percentile(array, (1, 99))
        stretched_array = np.clip(
            (array - low) * 255.0 / (high - low),
            0, 255
        ).astype(np.uint8)
        stretched_image = Image.fromarray(stretched_array)
        if save:
            stretched_image.save(f"{SAVE_PATH}")
        print(f"Contrast stretching performed and saved as {SAVE_PATH}")
        return stretched_image

if __name__ == "__main__":
    perform_contrast_improvement(IMAGE_PATH, "results", save=True)