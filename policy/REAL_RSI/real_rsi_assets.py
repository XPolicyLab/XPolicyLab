"""Portable locations for public vision models and shipped RGB reference images."""
import os
from pathlib import Path

import numpy as np
from PIL import Image


def model_path(name=''):
    root = os.environ.get('REAL_RSI_MODEL_DIR') or os.environ.get('LILO_MODELS')
    if not root:
        root = Path(__file__).resolve().parent / 'model_assets'
    return str(Path(root).expanduser().resolve() / name)


def load_rgb_reference(path):
    """Read a shipped PNG asset, never an encoded XPolicyLab observation."""
    if not Path(path).is_file():
        return None
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def save_rgb_diagnostic(path, rgb):
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path)
