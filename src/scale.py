import cv2
import numpy as np


def normalize_lung_roi(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    target_shape: tuple[int, int] = (2048, 2048),
    target_height_ratio: float = 2.0 / 3.0,
    enforce_width_constraint: bool = True,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | float | tuple = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalizes an image so the anatomical bounding box occupies exactly `target_height_ratio`

    of the output height and is centered horizontally and vertically.

    Parameters:
    -----------
    image : np.ndarray
        Input image array of shape (H_in, W_in) or (H_in, W_in, C).
    bbox : tuple (xmin, ymin, xmax, ymax)
        Bounding box of the region of interest (e.g., lungs) in pixel coordinates.
    target_shape : tuple (out_height, out_width)
        Desired output resolution (H_out, W_out). Default is (2048, 2048).
    target_height_ratio : float
        Fraction of vertical output height the ROI should occupy (default: 2/3).
    enforce_width_constraint : bool
        If True, downscales if the scaled width would exceed the image width.
    border_mode : int
        OpenCV pixel extrapolation mode (e.g., cv2.BORDER_CONSTANT).
    border_value : int or tuple
        Padding fill value for pixels sampled outside source boundaries.

    Returns:
    --------
    normalized_image : np.ndarray
        Transformed image of shape target_shape.
    affine_matrix : np.ndarray
        The 2x3 affine matrix used for the transformation.
    """
    H_out, W_out = target_shape
    xmin, ymin, xmax, ymax = bbox

    src_w = int(xmax - xmin)
    src_h = int(ymax - ymin)
    src_cx = xmin + src_w // 2
    src_cy = ymin + src_h // 2

    if src_w <= 0 or src_h <= 0:
        raise ValueError("Invalid bounding box coordinates with zero or negative area.")

    scale_y = (target_height_ratio * H_out) / src_h
    scale = scale_y

    if enforce_width_constraint:
        max_allowed_w = 0.95 * W_out
        if src_w * scale > max_allowed_w:
            scale = max_allowed_w / src_w

    dst_cx = W_out / 2.0
    dst_cy = H_out / 2.0

    tx = dst_cx - scale * src_cx
    ty = dst_cy - scale * src_cy

    affine_matrix = np.array([
        [scale, 0.0,   tx],
        [0.0,   scale, ty]
    ], dtype=np.float32)

    normalized_image = cv2.warpAffine(
        image,
        affine_matrix,
        (W_out, H_out),
        flags=cv2.INTER_CUBIC,
        borderMode=border_mode,
        borderValue=border_value,
    )

    return normalized_image, affine_matrix

def imwrite(path: str, img: np.ndarray):
    import os
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"Could not encode {path}")
    buf.tofile(path)  # unicode-safe on Windows

if __name__ == "__main__":
    from bounding_box import segment_lungs, mask_to_bbox, load_model

    IMAGE_PATH = "dataset/JSRT_images/JPCNN003.png"
    # IMAGE_PATH = "results/JPCNN003_aligned.png"
    MODEL_PATH = "models/unet_lung_seg.pt"
    model = load_model(MODEL_PATH)
    prob, mask, image = segment_lungs(IMAGE_PATH, model)
    bbox = mask_to_bbox(mask)
    if bbox is not None:
        normalized_image, affine_matrix = normalize_lung_roi(image, bbox)
        imwrite("normalized_lung_roi.png", normalized_image)
        print("Normalized image shape:", normalized_image.shape)
        print("Affine transformation matrix:\n", affine_matrix)
    else:
        print("No lungs detected in the mask.")