import os
from typing import Any

import imageio
import numpy as np
import torch
from decord import VideoReader
from PIL import Image


def load_video(video_path: str) -> list[Image.Image]:
    """Load all frames from a video file using Decord.

    Args:
        video_path: Path to the input video.

    Returns:
        List of PIL.Image frames.
    """
    video = VideoReader(video_path)
    indexes = np.arange(len(video), dtype=int)
    frames = video.get_batch(indexes)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def save_video(save_path: str, frames: list[Any], fps: int, **kwargs) -> None:
    """Write a list of frames as a video using imageio.

    Args:
        save_path: Output video path.
        frames: Sequence of numpy/PIL frames (RGB).
        fps: Frames per second.
        **kwargs: Extra imageio arguments.
    """
    imageio.mimsave(save_path, frames, fps=fps, **kwargs)


def get_fps(video_path: str) -> float:
    """Get average FPS from a video file via Decord."""
    video = VideoReader(video_path)
    return video.get_avg_fps()


def sample_video(video: VideoReader, indexes: np.ndarray, method: int = 2) -> np.ndarray:
    """Sample frames by indices with two strategies.

    Args:
        video: Decord VideoReader instance.
        indexes: 1D array of frame indices to fetch.
        method: 1 uses direct batched fetch; 2 fetches [0..max] and selects.

    Returns:
        Numpy array of frames with shape (len(indexes), H, W, C).
    """
    if method == 1:
        frames = video.get_batch(indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        frames = video.get_batch(all_indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        frames = frames[indexes]
    else:
        assert False
    return frames


def add_music_to_video(src_video_path: str, dst_video_path: str, save_video_path: str | None = None) -> None:
    """Mux audio track from src video into dst video using ffmpeg.

    Args:
        src_video_path: Path to video containing audio.
        dst_video_path: Video to which audio will be added.
        save_video_path: Optional output path; default appends '_music'.
    """
    aud_path = src_video_path[:-4] + '.m4a'
    if save_video_path is None:
        save_video_path = dst_video_path[:-4] + '_music' + dst_video_path[-4:]
    if os.path.exists(save_video_path):
        cmd = 'rm -f %s' % save_video_path
        os.system(cmd)
    cmd = 'ffmpeg -i %s -vn -y -acodec copy %s' % (src_video_path, aud_path)
    os.system(cmd)
    cmd = 'ffmpeg -i %s -i %s -vcodec copy -acodec copy %s' % (dst_video_path, aud_path, save_video_path)
    os.system(cmd)
    cmd = 'rm -f %s' % aud_path
    os.system(cmd)
