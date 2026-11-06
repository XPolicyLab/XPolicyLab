"""RGB files for pinned public assets and local diagnostic media.

These paths are never XPolicyLab observation buffers or trajectory image bits.
The official protocol owns decoding/encoding those buffers. Public PNGs and
local previews use their ordinary image-file format, retaining RGB throughout.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def copy_rgb(image):
    """Copy an already-RGB OpenCV-compatible array, dropping an optional alpha."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError('Expected an RGB or RGBA image array')
    if array.dtype not in (np.uint8, np.uint16, np.float32):
        raise TypeError('Expected an OpenCV-compatible RGB image dtype')
    return array[..., :3].copy()


def load_public_rgb(path):
    """Load an authenticated static image file; never decode an observation."""
    filename = Path(path)
    with Image.open(filename) as image:
        return np.asarray(image.convert('RGB')).copy()


def write_rgb_preview(path, image, *, quality=95):
    """Write local RGB diagnostic media, never a protocol/trajectory buffer."""
    filename = Path(path)
    array = np.asarray(image)
    if array.dtype not in (np.uint8, np.uint16):
        # Match OpenCV's ordinary diagnostic-image conversion to unsigned8.
        array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    preview = Image.fromarray(array)
    options = {'quality': quality} if filename.suffix.lower() in ('.jpg', '.jpeg') else {}
    try:
        preview.save(filename, **options)
    except OSError:
        return False
    return True
