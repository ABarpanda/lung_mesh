'''
The objective of this code is to Center align the X-ray image. It is possible that the image is off-center, so we will use this score to bring the image to the center. 
We do this by making the chest X-ray is symmetric about the vertical axis. That means when we flip the image left to right, the image should be more or less symmetrical. 
How it works:
We iteratively remove two columns of pixels from the right on each turn, and then we will flip the image left to right and compare its deviation (that is, the difference between the image before and after getting flipped). Then we will add a single column of black pixels to the right and a single column of black pixels to the left. We continue this until the error is lesser than twice the minimum error achieved yet. 
We repeat the same process for the left side as well. 
'''

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_PATH = r"D:\Projects\lung_mesh\dataset\JSRT_images\JPCNN003.png"

image = Image.open(IMAGE_PATH).convert("L")
array = np.asarray(image)

array_flipped = np.fliplr(array)

def find_flip_difference(image_array: np.ndarray) -> int:
    flipped_array = np.fliplr(image_array)
    diff = image_array.astype(np.int32) - flipped_array.astype(np.int32)
    return np.sum(np.abs(diff))

def move_image_right(image_array: np.array, n: int) -> np.array:
    columns_removed = np.array([i[:-2*n] for i in image_array])
    array_restored = np.array([[0]*n+list(row)+[0]*n for row in columns_removed])
    return array_restored

def move_image_left(image_array: np.array, n: int) -> np.array:
    columns_removed = np.array([i[2*n:] for i in image_array])
    array_restored = np.array([[0]*n+list(row)+[0]*n for row in columns_removed])
    return array_restored

def align(image_array, k=2, save_path=None):
    error_dict = {}
    error_dict[0] = find_flip_difference(move_image_right(image_array, 0))
    error_dict[0] = find_flip_difference(move_image_left(image_array, 0))
    i = 0
    while max(error_dict.values()) < k*min(error_dict.values()):
        error_dict[i] = find_flip_difference(move_image_right(image_array, i))
        error_dict[-i] = find_flip_difference(move_image_left(image_array, i))
        i+=1

    # print(f"Error dictionary = {error_dict}")

    min_error_indices = [key for key, value in error_dict.items() if value == min(error_dict.values())]
    if len(min_error_indices)>=1:
        min_error_index = min(min_error_indices, key=abs)
    else: min_error_index = min_error_indices[0]

    print(f"Minimum error index = {min_error_index}")

    if min_error_index>0:
        result_array = move_image_right(image_array, min_error_index)
    elif min_error_index<0:
        result_array = move_image_left(image_array, -min_error_index)
    else:
        result_array = image_array
    
    if save_path:
        plt.imsave(save_path, result_array, cmap="gray")

    return result_array

aligned = align(array, k=3, save_path=f"./results/{os.path.splitext(os.path.basename(IMAGE_PATH))[0]}_aligned.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

mid_x = (array.shape[1] - 1) / 2

axes[0].imshow(array, cmap="gray")
axes[0].axvline(x=mid_x, color="red", linestyle="--", linewidth=1.5)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(aligned, cmap="gray")
axes[1].axvline(x=mid_x, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title("Aligned")
axes[1].axis("off")

plt.tight_layout()
plt.show()