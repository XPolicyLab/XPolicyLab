"""Per-camera episode recordings: MP4 through imageio-ffmpeg when installed, else PNG frames."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from knowin_agent.runtime.backend import Observation


class FrameRecorder:
    """A `frame_sink` that keeps every ``stride``-th RGB frame of each camera."""

    def __init__(self, cameras: Optional[Sequence[str]] = None, stride: int = 1, max_frames: Optional[int] = None):
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.cameras = None if cameras is None else tuple(cameras)
        self.stride = int(stride)
        self.max_frames = max_frames
        self.frames: dict[str, list[np.ndarray]] = {}
        self._calls = 0

    def __call__(self, observation: Observation) -> None:
        index, self._calls = self._calls, self._calls + 1
        if index % self.stride:
            return
        for name, frame in observation.cameras.items():
            if self.cameras is not None and name not in self.cameras:
                continue
            frames = self.frames.setdefault(name, [])
            if self.max_frames is None or len(frames) < self.max_frames:
                frames.append(np.array(frame.rgb, dtype=np.uint8, copy=True))

    def write(self, out_dir: os.PathLike | str, fps: float = 25.0, prefix: str = "") -> dict[str, Path]:
        return write_videos(self.frames, out_dir, fps=fps / self.stride, prefix=prefix)


def _imageio():
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return None
    return imageio


def write_videos(
    frames: Mapping[str, Sequence[np.ndarray]], out_dir: os.PathLike | str, fps: float = 25.0, prefix: str = ""
) -> dict[str, Path]:
    """Write one MP4 (or one PNG directory) per camera; returns camera -> written path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    imageio = _imageio()
    written = {}
    for camera, images in frames.items():
        if not images:
            continue
        stem = f"{prefix}{camera}"
        if imageio is not None:
            path = out / f"{stem}.mp4"
            with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1) as writer:
                for image in images:
                    writer.append_data(np.asarray(image, dtype=np.uint8))
        else:
            from PIL import Image

            path = out / stem
            path.mkdir(exist_ok=True)
            for index, image in enumerate(images):
                Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path / f"{index:06d}.png")
        written[camera] = path
    return written
