import cv2
import numpy as np
import matplotlib.pyplot as plt

def xray_edges(image_gray):
    # 1. Enhance local gradients using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(15, 15))
    enhanced = clahe.apply(image_gray)

    # 2. Edge-preserving blur (replaces standard Gaussian)
    # d: neighborhood diameter, sigmaColor & sigmaSpace balance smoothing vs edge retention
    smoothed = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)

    # 3. Compute gradient magnitude using a larger aperture (Sobel 5x5)
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # 4. Adaptive threshold calculation based on non-zero gradient percentiles
    non_zero_grads = magnitude[magnitude > 0]
    high_thresh = np.percentile(non_zero_grads, 85)
    low_thresh = high_thresh * 0.35

    # 5. Canny with custom aperture
    edges = cv2.Canny(
        smoothed,
        threshold1=low_thresh,
        threshold2=high_thresh,
        apertureSize=5,
        L2gradient=True
    )
    return edges

def custom(image):
    image = 255 - abs(255 - 2 * image)
    return image

def contrast(image, alpha=5):
    image = image.astype(np.float32)
    mu = np.mean(image)
    output = mu + alpha * (image - mu)
    output = np.clip(output, 0, 255)
    return output.astype(np.uint8)

# 1. Load image in grayscale (0 flag ensures grayscale mode)
image = cv2.imread('dataset/toy_dataset/input_image.png', cv2.IMREAD_GRAYSCALE)
image = image[100:-50, 5:-5]

modified = custom(image.astype(np.uint8))
edges = cv2.Canny(modified, threshold1=50, threshold2=150)

# 4. Display original grayscale vs. detected edges
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[0].axis('off')

axes[1].imshow(modified, cmap='gray')
axes[1].set_title('Modified Image')
axes[1].axis('off')

axes[2].imshow(edges, cmap='gray')
axes[2].set_title('Canny Edges')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Optional: save the output edge map
# cv2.imwrite('edges.jpg', edges)