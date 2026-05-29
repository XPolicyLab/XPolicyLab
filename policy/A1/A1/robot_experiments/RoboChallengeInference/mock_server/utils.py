# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image


def resize_with_pad_single(
        image: np.ndarray, dsize, method=Image.BILINEAR
) -> np.ndarray:
    width, height = dsize
    if image.shape[:2] == (height, width):
        return image

    pil_image = Image.fromarray(image)

    # Resize with padding using the existing helper function
    resized_pil = _resize_with_pad_pil(pil_image, height, width, method=method)

    # Convert back to numpy array
    return np.array(resized_pil)


def _resize_with_pad_pil(
        image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
