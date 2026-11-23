"""Decode FFV1 gray16le depth videos produced by the libero/robotwin
convert pipelines.

On-disk format:
  - container: Matroska (``.mkv``)
  - codec:     FFV1 (lossless intra-frame)
  - pix_fmt:   gray16le  (one channel, uint16 little-endian)

FFV1 gray16le preserves the source ``uint16 mm`` depth values bit-exactly
(verified bit-exact round-trip). Compared to the dual-row libx264rgb
alternative it is ~4.8× smaller on disk and ~2× faster to decode.

The legacy dual-row uint8 decoder is retained for datasets produced before
this migration; new data should use ``decode_depth_mkv_at_timestamps``.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def decode_depth_mkv_at_timestamps(
    mkv_path: str | Path,
    timestamps_sec: Iterable[float],
    fps: int,
) -> torch.Tensor:
    """Decode specific frames of an FFV1 gray16le depth video.

    Because FFV1 in matroska is not part of lerobot's standard video
    pipeline, we invoke ffmpeg directly: decode the whole stream to raw
    gray16le then index the requested frames. For typical LIBERO episodes
    (≤200 frames of 512x512 depth) this is fast enough (<50 ms/episode on
    a modern CPU).

    Args:
        mkv_path: path to the .mkv file.
        timestamps_sec: iterable of timestamps (seconds). Mapped to frame
            indices via ``round(t * fps)``.
        fps: frame rate used at encode time (must match ``meta/info.json``).

    Returns:
        torch.Tensor of shape ``(T, H, W)`` with dtype ``int32`` containing
        depth in millimetres.
    """
    mkv_path = str(mkv_path)
    # ffprobe only gives width/height reliably for FFV1/matroska; nb_frames is
    # often "N/A", so we derive the frame count from the decoded byte-size.
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "default=nokey=1:noprint_wrappers=1", mkv_path,
    ]).decode().strip().splitlines()
    W, H = int(probe[0]), int(probe[1])

    raw = subprocess.check_output([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", mkv_path,
        "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
    ])
    bytes_per_frame = H * W * 2
    assert len(raw) % bytes_per_frame == 0, (
        f"Decoded byte length {len(raw)} not divisible by frame size "
        f"{bytes_per_frame} for {mkv_path}"
    )
    n_frames = len(raw) // bytes_per_frame
    frames = np.frombuffer(raw, dtype=np.uint16).reshape(n_frames, H, W)

    indices = [max(0, min(n_frames - 1, int(round(t * fps)))) for t in timestamps_sec]
    selected = frames[indices].astype(np.int32, copy=True)
    return torch.from_numpy(selected)


def decode_dual_row_depth(frame_2h_w_3: torch.Tensor) -> torch.Tensor:
    """Legacy dual-row uint8 decoder.

    Kept for datasets produced before the FFV1 migration (``depth_format =
    "dual_row_uint8_mm"``). Input: ``(..., 2H, W, 3)`` uint8 or float [0, 1].
    Output: ``(..., H, W)`` int32 depth in millimetres.
    """
    if frame_2h_w_3.is_floating_point():
        frame = (frame_2h_w_3 * 255.0).round().clamp(0, 255).to(torch.int32)
    else:
        frame = frame_2h_w_3.to(torch.int32)

    H = frame.shape[-3] // 2
    lo = frame[..., :H, :, 0]
    hi = frame[..., H:, :, 0]
    return (hi << 8) | lo


def depth_mm_to_metres(depth_mm: torch.Tensor) -> torch.Tensor:
    return depth_mm.to(torch.float32) / 1000.0
