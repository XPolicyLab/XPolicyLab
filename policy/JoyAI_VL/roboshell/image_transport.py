"""Lossless PNGs used by the agent HTTP interface and perception tools.

Official XPolicyLab buffers are decoded by the policy server before they enter
this interface. These functions accept our own PNGs only, never trajectory JPEGs.
Arrays remain RGB; grayscale images and binary masks remain single-channel.
"""
import io

import numpy as np
from PIL import Image


def encode_png(image):
    array = np.asarray(image)
    if array.dtype != np.uint8 or not (array.ndim == 2 or
            (array.ndim == 3 and array.shape[2] in (3, 4))):
        raise ValueError("PNG transport expects uint8 grayscale, RGB or RGBA pixels")
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(array)).save(buffer, format="PNG")
    return buffer.getvalue()


def decode_png(data, mode="RGB"):
    if mode not in ("RGB", "L"):
        raise ValueError("PNG transport mode must be RGB or L")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG":
            raise ValueError("Expected an internal PNG image")
        return np.array(image.convert(mode), dtype=np.uint8, copy=True)
