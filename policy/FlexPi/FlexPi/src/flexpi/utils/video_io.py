import os
from typing import Iterable

import imageio
import numpy as np
from PIL import Image

from .fs import ensure_dir


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_mp4(frames: Iterable[Image.Image], path: str, fps: int = 8):
    ensure_dir(os.path.dirname(path) or ".")
    # yuv444p (full chroma) instead of yuv420p (4:2:0 subsampled). 4:2:0
    # averages chroma in 2×1 column pairs, which is exactly the structure
    # produced by `tile_mode="repeat"` (np.repeat-style column doubling) —
    # so 4:2:0 silently turns a NN-doubled slot into a bilinearly-stretched
    # look in the saved MP4. yuv444p preserves
    # the duplication faithfully. Larger file, plays fine in modern players.
    writer = imageio.get_writer(
        path,
        fps=max(fps, 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv444p",
    )
    try:
        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            writer.append_data(_to_even_frame(arr))
    finally:
        writer.close()
