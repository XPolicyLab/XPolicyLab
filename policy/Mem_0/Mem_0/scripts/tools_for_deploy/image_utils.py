"""Image utilities for deployment."""

import numpy as np
from PIL import Image


def to_pil(image_array: np.ndarray, target_size) -> Image.Image:
    """
    Convert array to RGB PIL.Image and resize to target_size.
    - Accepts float [0,1] or uint8 [0,255]; outputs uint8.
    - Handles optional batch dim (1, H, W, C).
    """
    if image_array is None:
        raise ValueError("image_array is None, cannot build PIL image")
    if image_array.ndim == 4 and image_array.shape[0] == 1:
        image_array = image_array[0]
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    if image_array.dtype != np.uint8:
        if image_array.max() <= 1.0:
            image_array = (image_array * 255.0).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
    pil_img = Image.fromarray(image_array)
    if pil_img.size != tuple(target_size):
        pil_img = pil_img.resize(tuple(target_size))
    return pil_img

