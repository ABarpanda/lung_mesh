import os
import cv2
import numpy as np

def generate_candemir_bbox(right_mask_path, left_mask_path, output_fmt="yolo"):
    """
    Computes the enclosing thoracic bounding box matching the anatomical 
    extent used in Candemir et al. (combining right and left lung fields).
    """
    # Load binary masks for left and right lung fields
    right_mask = cv2.imread(right_mask_path, cv2.IMREAD_GRAYSCALE)
    left_mask = cv2.imread(left_mask_path, cv2.IMREAD_GRAYSCALE)

    if right_mask is None or left_mask is None:
        raise ValueError("Mask files could not be read.")

    # Union the two lung fields to represent the full thoracic cavity extent
    total_thorax_mask = cv2.bitwise_or(right_mask, left_mask)
    h, w = total_thorax_mask.shape

    # Find non-zero pixel coordinates (indices where lung tissue is present)
    y_indices, x_indices = np.where(total_thorax_mask > 0)

    if len(x_indices) == 0 or len(y_indices) == 0:
        return None  # Empty mask

    # Compute minimal enclosing coordinates [xmin, ymin, xmax, ymax]
    xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
    ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))

    # Add standard anatomical buffer (Candemir et al. typically preserve 
    # margin around costophrenic angles and lateral ribs, ~2-3%)
    pad_x = int(0.02 * (xmax - xmin))
    pad_y = int(0.02 * (ymax - ymin))
    
    xmin = max(0, xmin - pad_x)
    ymin = max(0, ymin - pad_y)
    xmax = min(w - 1, xmax + pad_x)
    ymax = min(h - 1, ymax + pad_y)

    if output_fmt == "pascal_voc":
        return [xmin, ymin, xmax, ymax]

    elif output_fmt == "yolo":
        # Normalized [x_center, y_center, width, height]
        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = (xmin + box_w / 2.0) / w
        y_center = (ymin + box_h / 2.0) / h
        norm_w = box_w / w
        norm_h = box_h / h
        return [0, x_center, y_center, norm_w, norm_h]
