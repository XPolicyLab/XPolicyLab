#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import importlib
import logging
import os
import time as _dl_time   # alias: do not shadow other `time` imports in callers
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import av
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image


# ─────────────────────────── DLTIMING (env-gated) ─────────────────────────────
# Temporary diagnostic harness for the deterministic ~35-s every-N-steps stall.
# Activate with FLEXPI_DL_TIMING=1. Default (unset) is a strict no-op:
# every helper short-circuits on the env check, so decode paths return the
# same tensors in the same order with no extra timing overhead.
# Env vars:
#   FLEXPI_DL_TIMING=1                       enable
#   FLEXPI_DL_TIMING_THRESHOLD_MS=2000       only emit when ms > threshold
#   FLEXPI_DL_TIMING_VERBOSE=0|1             1 = emit every sample
# Output format (one line per emit):
#   [DLTIMING] stage=... worker=... pid=... ... ms=...

def _dl_timing_enabled() -> bool:
    return os.environ.get("FLEXPI_DL_TIMING") == "1"


def _dl_worker_id() -> int:
    """DataLoader worker id, or -1 if not in a worker process."""
    try:
        import torch.utils.data as _tud
        wi = _tud.get_worker_info()
        return int(wi.id) if wi is not None else -1
    except Exception:
        return -1


def _dl_emit(payload: dict, threshold_ms: float | None = None) -> None:
    """Print a single line `[DLTIMING] k=v k=v ...` if payload['ms'] (or
    payload['total_ms']) exceeds `threshold_ms`, OR if VERBOSE=1.
    Skips entirely when FLEXPI_DL_TIMING != '1'.
    """
    if not _dl_timing_enabled():
        return
    if threshold_ms is None:
        threshold_ms = float(os.environ.get("FLEXPI_DL_TIMING_THRESHOLD_MS", "2000"))
    verbose = os.environ.get("FLEXPI_DL_TIMING_VERBOSE") == "1"
    ms = payload.get("ms", payload.get("total_ms", 0))
    if not (ms > threshold_ms or verbose):
        return
    # flush=True so worker subprocess output isn't held in the pipe buffer
    print("[DLTIMING] " + " ".join(f"{k}={v}" for k, v in payload.items()), flush=True)
# ──────────────────────────────────────────────────────────────────────────────


def _decode_depth_frames_decord_x264rgb(
    video_path: str,
    timestamps: "list[float]",
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
) -> "torch.Tensor":
    """Depth decode for libx264rgb dual-channel-packed videos via decord.

    Encoding (matches preprocess_depth_to_x264rgb): 3-channel rgb24,
    R = depth_hi, G = depth_lo, B = unused. Recovery: uint16 = R<<8 | G.
    decord get_batch is GIL-released and isolation-bench-faster than pyav
    on the same x264rgb file (3× cold, equal warm). Output shape matches
    pyav path: (T, H, W) uint16 mm.

    Required for any future sharded sampling: a single call can request
    `vr.get_batch(sorted_indices)` for many frames at once, batching the
    file open across N samples.
    """
    import decord  # local import: matches existing decord usage pattern
    import numpy as np
    _dl_t0 = _dl_time.perf_counter() if _dl_timing_enabled() else None
    # Mirror FLEXPI_DECORD_THREADS gating from decode_video_frames_decord;
    # libx264rgb is bit-exact under decord's threading. Default 1 = unchanged.
    _nt = int(os.environ.get("FLEXPI_DECORD_THREADS", "1"))
    vr = decord.VideoReader(str(video_path), num_threads=_nt)
    average_fps = vr.get_avg_fps()
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    sorted_pos = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
    sorted_indices = [frame_indices[i] for i in sorted_pos]
    frames_sorted = vr.get_batch(sorted_indices).asnumpy()  # (N, H, W, 3) uint8
    restore = sorted(range(len(sorted_pos)), key=lambda i: sorted_pos[i])
    frames = frames_sorted[restore]
    del vr

    # Tolerance check (decord rounds ts*fps to integer index; exact for fixed-fps).
    loaded_ts = torch.tensor([idx / average_fps for idx in frame_indices], dtype=torch.float32)
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    max_dev = (loaded_ts - query_ts).abs().max().item() if len(timestamps) else 0.0
    if max_dev >= tolerance_s:
        raise AssertionError(
            f"Depth (decord) timestamp alignment violation: max_dev {max_dev:.4f} > tol {tolerance_s}.\n"
            f"queried: {query_ts}\nloaded: {loaded_ts}\nvideo: {video_path}"
        )

    if log_loaded_timestamps:
        for ts in loaded_ts.tolist():
            logging.info(f"depth(decord) frame loaded at timestamp={ts:.4f}")

    # Channel-pack recovery: hi << 8 | lo.
    out = (frames[..., 0].astype(np.uint16) << 8) | frames[..., 1].astype(np.uint16)
    result = torch.from_numpy(out)  # (T, H, W) uint16
    if _dl_t0 is not None:
        _dl_emit({
            "stage": "depth_decode_decord_x264rgb",
            "worker": _dl_worker_id(), "pid": os.getpid(),
            "file": Path(str(video_path)).name,
            "n_ts": len(timestamps),
            "shape": tuple(result.shape),
            "decord_threads": _nt,
            "ms": int((_dl_time.perf_counter() - _dl_t0) * 1000),
        }, threshold_ms=float(os.environ.get("FLEXPI_DL_TIMING_THRESHOLD_MS", "2000")) / 4)
    return result


def get_safe_default_codec():
    # Manual override (e.g. LEROBOT_VIDEO_BACKEND=torchcodec) wins over auto-detect.
    forced = os.environ.get("LEROBOT_VIDEO_BACKEND")
    if forced:
        return forced
    # Prefer decord (~12–20× faster cold than torchcodec on AV1 RoboTwin files,
    # measured on GPFS 2026-05-03). decord2 wheel imports as `decord` and reads
    # AV1 where upstream decord 0.6 doesn't. We don't lose warm-cache benefit
    # since the persistent decoder cache was reverted (fork-unsafe).
    if importlib.util.find_spec("decord"):
        return "decord"
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder. "
            "WARNING: the pyav backend has a known memory leak with torchvision.io.VideoReader that causes "
            "CPU RAM to grow ~50 GB/min during training, eventually triggering OOM kills. "
            "To fix: install FFmpeg (`conda install -c conda-forge ffmpeg=7`) so torchcodec can load."
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        try:
            return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
        except Exception as err:
            warnings.warn(
                f"torchcodec video decode failed ({type(err).__name__}: {err}); falling back to torchvision/pyav."
            )
            return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend="pyav")
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    elif backend == "decord":
        return decode_video_frames_decord(video_path, timestamps, tolerance_s)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


_RGB_PYAV_DECODE_GC_INTERVAL = 64
_rgb_pyav_decode_counter = 0


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)
    _dl_t0 = _dl_time.perf_counter() if _dl_timing_enabled() else None

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # FLEXPI_PYAV_THREADS env-gates per-decoder threading. Safe (bit-exact)
    # for h264 frame/slice threading. Some torchvision versions may not expose
    # `reader.container`; the try/except keeps this a soft op on those builds.
    if backend == "pyav":
        try:
            _stream = reader.container.streams.video[0]
            _stream.thread_count = int(os.environ.get("FLEXPI_PYAV_THREADS", "2"))
            _stream.thread_type = "AUTO"
        except Exception:
            pass

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()
        del reader
        # Mitigate memory leak in torchvision.io.VideoReader with pyav backend:
        # the C-level decoder buffers are not freed until garbage collected.
        # Per-call gc.collect() costs ~100 ms each — dominates decode wall-time.
        # Periodic (every N) is cheap and still keeps RSS bounded; each worker
        # has its own counter (lock-free, no fork issue).
        global _rgb_pyav_decode_counter
        _rgb_pyav_decode_counter += 1
        if _rgb_pyav_decode_counter % _RGB_PYAV_DECODE_GC_INTERVAL == 0:
            import gc as _gc
            _gc.collect()
    else:
        del reader

    # Use float32 for timestamp distance computation (torch.cdist doesn't support bfloat16).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    if _dl_t0 is not None:
        _dl_emit({
            "stage": "rgb_decode_torchvision",
            "worker": _dl_worker_id(), "pid": os.getpid(),
            "file": Path(video_path).name,
            "backend": backend,
            "n_ts": len(timestamps),
            "shape": tuple(closest_frames.shape),
            "ms": int((_dl_time.perf_counter() - _dl_t0) * 1000),
        }, threshold_ms=float(os.environ.get("FLEXPI_DL_TIMING_THRESHOLD_MS", "2000")) / 4)
    return closest_frames


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    # initialize video decoder
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    loaded_frames = []
    loaded_ts = []
    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]

    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    # Use float32 for timestamp distance computation (torch.cdist doesn't support bfloat16).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts = torch.tensor(loaded_ts, dtype=torch.float32)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range (channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_decord(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
) -> torch.Tensor:
    """Decode frames at requested timestamps using decord (decord2 on FFmpeg ≥7).

    decord2 reads AV1 ~12× faster than torchcodec on cold open in our benchmarks
    (~4 ms vs ~57 ms). Output matches torchcodec/torchvision: float32 in [0,1],
    channel-first (N, C, H, W).
    """
    import decord  # decord2 also imports as `decord`
    _dl_t0 = _dl_time.perf_counter() if _dl_timing_enabled() else None

    # FLEXPI_DECORD_THREADS env-gates decord's per-VideoReader thread count.
    # h264 decode is bit-exact under decord's internal threading (slice/frame),
    # so this is a pure wall-time win. Default 1 = no change vs prior behavior.
    _nt = int(os.environ.get("FLEXPI_DECORD_THREADS", "1"))
    vr = decord.VideoReader(str(video_path), num_threads=_nt)
    average_fps = vr.get_avg_fps()
    frame_indices = [round(ts * average_fps) for ts in timestamps]

    # Sort + restore so decord decodes forward from each keyframe (yam_demo trick).
    sorted_pos = sorted(range(len(frame_indices)), key=lambda i: frame_indices[i])
    sorted_indices = [frame_indices[i] for i in sorted_pos]
    frames_sorted = vr.get_batch(sorted_indices).asnumpy()  # (N, H, W, 3) uint8
    restore = sorted(range(len(sorted_pos)), key=lambda i: sorted_pos[i])
    frames = frames_sorted[restore]
    del vr  # release decoder before return; no module-level state

    # Tolerance check (decord doesn't return per-frame PTS the way torchcodec does;
    # frame_indices = round(ts * fps) is exact for fixed-fps RoboTwin video).
    loaded_ts = torch.tensor([idx / average_fps for idx in frame_indices], dtype=torch.float32)
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    max_dev = (loaded_ts - query_ts).abs().max().item() if len(timestamps) else 0.0
    assert max_dev < tolerance_s, (
        f"decord query→loaded ts max dev {max_dev:.4f} > tol {tolerance_s}"
        f"\nqueried: {query_ts}\nloaded: {loaded_ts}\nvideo: {video_path}"
    )

    # (N, H, W, 3) uint8 → (N, 3, H, W) float32 in [0,1]
    result = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().float() / 255
    if _dl_t0 is not None:
        _dl_emit({
            "stage": "rgb_decode_decord",
            "worker": _dl_worker_id(), "pid": os.getpid(),
            "file": Path(str(video_path)).name,
            "n_ts": len(timestamps),
            "shape": tuple(result.shape),
            "decord_threads": _nt,
            "ms": int((_dl_time.perf_counter() - _dl_t0) * 1000),
        }, threshold_ms=float(os.environ.get("FLEXPI_DL_TIMING_THRESHOLD_MS", "2000")) / 4)
    return result


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec == "h264_nvenc" :
        return encode_video_frames_ffmpeg(
            imgs_dir, video_path, fps, pix_fmt = pix_fmt, overwrite=overwrite
        )

    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Get input frames
    template = "frame_" + ("[0-9]" * 6) + ".jpeg"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            input_image = Image.open(input_data).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")

# GPU speed up video encoding
def encode_video_frames_ffmpeg(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "h264_nvenc",
    pix_fmt: str = "yuv420p",
    g: int | None = 4,
    crf: int | None = 30,
    overwrite: bool = False,
) -> None:
    import subprocess
    import shutil

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not installed or not in system PATH")

    # Build ffmpeg command
    cmd = [
        ffmpeg_path,
        "-y" if overwrite else "-n",
        "-framerate", str(fps),
        "-pattern_type", "sequence",
        "-i", str(imgs_dir / "frame_%06d.jpeg"),
        "-vcodec", vcodec,
        "-pix_fmt", pix_fmt,
    ]

    if g is not None:
        cmd.extend(["-g", str(g)])

    if crf is not None:
        cmd.extend(["-crf", str(crf)])

    cmd.append(str(video_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg failed:\nCommand: {' '.join(cmd)}\nStdout: {e.stdout}\nStderr: {e.stderr}"
        logging.error(error_msg)
        logging.error(video_path)
        logging.error(vcodec)
        print(f"ffmpeg {' '.join(cmd)}")
        raise RuntimeError(error_msg)
    except Exception as e:
        error_msg = f"Failed to run FFmpeg command: {' '.join(cmd)}\nError: {str(e)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)


# ---------------------------------------------------------------------------
# Depth video encoding / decoding (lossless uint16)
#
# Two supported codecs, both bit-exact round-trip:
#   - "ffv1"       : MKV container, gray16le pixel format, single-channel 16-bit
#   - "libx264rgb" : MP4 container, rgb24 lossless, uint16 packed as (hi, lo, 0)
# ---------------------------------------------------------------------------

DEPTH_VIDEO_EXT = {"ffv1": ".mkv", "libx264rgb": ".mp4"}

# Periodic gc for pyav decoders. Benchmarking shows `container.close() + del` alone
# keeps RSS bounded in most cases, but upstream LeRobot saw edge-case leaks with
# certain pyav/ffmpeg combos — so we still force a gc every N decodes as a cheap
# safety net. N=64 keeps overhead under 1 % of decode wall-time. Each DataLoader
# worker has its own interpreter and its own counter, so this is lock-free.
_DEPTH_DECODE_GC_INTERVAL = 64
_depth_decode_counter = 0


def encode_depth_video_frames(
    depth: "np.ndarray",
    video_path: "Path | str",
    fps: int,
    vcodec: str,
    overwrite: bool = False,
) -> None:
    """Encode a (N, H, W) uint16 depth array to a lossless video.

    vcodec must be "ffv1" (gray16le) or "libx264rgb" (packed uint16 in rgb24).
    """
    import subprocess
    import shutil
    import numpy as np

    if depth.dtype != np.uint16:
        raise ValueError(f"depth must be uint16, got {depth.dtype}")
    if depth.ndim != 3:
        raise ValueError(f"depth must be (N, H, W), got shape {depth.shape}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH")

    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    N, H, W = depth.shape
    size_arg = f"{W}x{H}"

    if vcodec == "ffv1":
        in_pix_fmt = "gray16le"
        out_pix_fmt = "gray16le"
        payload = np.ascontiguousarray(depth, dtype="<u2").tobytes()
        codec_args = ["-c:v", "ffv1", "-level", "3", "-g", "1"]
    elif vcodec == "libx264rgb":
        hi = (depth >> 8).astype(np.uint8)
        lo = (depth & 0xFF).astype(np.uint8)
        zero = np.zeros_like(hi)
        rgb = np.stack([hi, lo, zero], axis=-1)  # (N, H, W, 3)
        payload = np.ascontiguousarray(rgb).tobytes()
        in_pix_fmt = "rgb24"
        out_pix_fmt = "rgb24"
        codec_args = ["-c:v", "libx264rgb", "-preset", "veryslow", "-qp", "0", "-g", "1"]
    else:
        raise ValueError(f"Unsupported depth vcodec: {vcodec!r} (use 'ffv1' or 'libx264rgb')")

    cmd = [
        ffmpeg_path,
        "-y" if overwrite else "-n",
        "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", in_pix_fmt,
        "-s", size_arg,
        "-framerate", str(fps),
        "-i", "-",
        *codec_args,
        "-pix_fmt", out_pix_fmt,
        str(video_path),
    ]

    proc = subprocess.run(cmd, input=payload, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg depth encode failed (rc={proc.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {proc.stderr.decode(errors='replace')[:4000]}"
        )
    if not video_path.exists():
        raise OSError(f"Depth encode did not produce file: {video_path}")


def decode_depth_frames(
    video_path: "Path | str",
    timestamps: "list[float]",
    tolerance_s: float,
    vcodec: str,
    log_loaded_timestamps: bool = False,
) -> "torch.Tensor":
    """Timestamp-aware lossless depth decode — mirror of decode_video_frames_torchvision.

    Given a lossless depth video (FFV1 gray16le MKV or libx264rgb packed-uint16 MP4)
    and a list of query timestamps (in seconds), return the frames at those times
    as `torch.uint16` shape `[T, H, W]` in millimetres. No normalization applied
    (raw mm). Uses pyav directly — torchcodec refuses FFV1 MKV containers and
    returns 8-bit data even for libx264rgb.

    Pattern matches decode_video_frames_torchvision so alignment with the RGB
    stream is guaranteed: both streams queried at identical timestamps, both
    with the same tolerance_s, both pick argmin over |query - loaded_pts|.
    """
    import gc as _gc
    import numpy as np

    global _depth_decode_counter
    _dl_t0 = _dl_time.perf_counter() if _dl_timing_enabled() else None

    if vcodec not in ("ffv1", "libx264rgb"):
        raise ValueError(f"Unsupported depth vcodec: {vcodec!r}")
    target_format = "gray16le" if vcodec == "ffv1" else "rgb24"
    video_path = str(video_path)

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    # Matroska / FFV1 quantizes PTS to milliseconds, so a query at e.g.
    # 5.03333... (= 151/30, exact float64) sees the encoded frame at
    # PTS=5.033 (rounded to 1ms). Without buffer, the strict `ts < first_ts`
    # check drops it as out-of-window, and the closest-loaded frame ends up
    # one full frame later (≈33 ms at 30fps). Widen the lower bound by 2 ms
    # so quantized frames near `first_ts` are kept; argmin selection still
    # picks the closest match within tolerance_s.
    _ts_lower_buffer_s = 0.002
    first_ts_filter = first_ts - _ts_lower_buffer_s

    # libx264rgb dual-row depth: prefer decord (3× faster than pyav under
    # FS contention; equal on quiet nodes). FFV1 mkv stays on pyav because
    # decord can't decode it.
    if vcodec == "libx264rgb" and os.environ.get("LEROBOT_DEPTH_DECORD", "1") == "1":
        return _decode_depth_frames_decord_x264rgb(
            video_path, timestamps, tolerance_s, log_loaded_timestamps
        )

    loaded_frames: list[np.ndarray] = []
    loaded_ts: list[float] = []
    container = av.open(video_path, "r")
    try:
        stream = container.streams.video[0]
        # FLEXPI_PYAV_THREADS env-gates per-decoder threading. Both h264 and
        # FFV1 are bit-exact under FRAME/SLICE threading, so this is a pure
        # wall-time win; defaults to 2 to stay safely under-subscribed at
        # num_workers*threads ~= cpus_per_task. Set to 1 to disable.
        try:
            stream.thread_count = int(os.environ.get("FLEXPI_PYAV_THREADS", "2"))
            stream.thread_type = "AUTO"
        except Exception:
            pass
        # Seek to the keyframe at-or-before first_ts before decoding. Depth
        # videos are encoded with -g 1 (all-intra) so every frame is a keyframe,
        # and seek lands exactly on the target. Without this, pyav decodes
        # every frame from 0 up to first_ts (~0.3 ms/frame for FFV1 240×320
        # gray16; ~20 ms wasted per cam at average mid-episode targets, which
        # dominates per-sample wall-time in obs-only training).
        target_pts = int(first_ts / float(stream.time_base))
        container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            ts = float(frame.pts * frame.time_base)
            if ts < first_ts_filter:
                continue  # before our window (with ms-quantization buffer)
            arr = frame.to_ndarray(format=target_format)  # (H, W) uint16 or (H, W, 3) uint8
            loaded_frames.append(arr)
            loaded_ts.append(ts)
            if log_loaded_timestamps:
                logging.info(f"depth frame loaded at timestamp={ts:.4f}")
            if ts >= last_ts:
                break
    finally:
        container.close()
        del container
        _depth_decode_counter += 1
        if _depth_decode_counter % _DEPTH_DECODE_GC_INTERVAL == 0:
            _gc.collect()

    if not loaded_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    # Unpack libx264rgb-packed depth back to uint16.
    if vcodec == "libx264rgb":
        loaded_frames = [
            (f[..., 0].astype(np.uint16) << 8) | f[..., 1].astype(np.uint16)
            for f in loaded_frames
        ]  # each → (H, W) uint16

    # Pick closest-to-query frames (same pattern as RGB decoder).
    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts_t = torch.tensor(loaded_ts, dtype=torch.float32)
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise AssertionError(
            f"Depth timestamp alignment violation: {min_[~is_within_tol]} > {tolerance_s=}.\n"
            f"queried timestamps: {query_ts}\n"
            f"loaded timestamps: {loaded_ts_t}\n"
            f"video: {video_path}"
        )

    frames_np = np.stack([loaded_frames[int(i)] for i in argmin_], axis=0)  # (T, H, W) uint16
    result = torch.from_numpy(frames_np)
    if _dl_t0 is not None:
        _dl_emit({
            "stage": "depth_decode_pyav",
            "worker": _dl_worker_id(), "pid": os.getpid(),
            "file": Path(video_path).name,
            "vcodec": vcodec,
            "n_ts": len(timestamps),
            "shape": tuple(result.shape),
            "pyav_threads": int(os.environ.get("FLEXPI_PYAV_THREADS", "2")),
            "ms": int((_dl_time.perf_counter() - _dl_t0) * 1000),
        }, threshold_ms=float(os.environ.get("FLEXPI_DL_TIMING_THRESHOLD_MS", "2000")) / 4)
    return result


def decode_depth_video_full(
    video_path: "Path | str",
    vcodec: str,
) -> "np.ndarray":
    """Decode a full depth video back to (N, H, W) uint16 via pyav.

    vcodec must match whatever was used at encode time ("ffv1" or "libx264rgb").

    Memory hygiene: `container.close() + del` keeps RSS bounded in steady state;
    a gc every N decodes runs as a safety net (see _DEPTH_DECODE_GC_INTERVAL).
    """
    import gc as _gc
    import numpy as np

    global _depth_decode_counter

    if vcodec not in ("ffv1", "libx264rgb"):
        raise ValueError(f"Unsupported depth vcodec: {vcodec!r}")
    target_format = "gray16le" if vcodec == "ffv1" else "rgb24"

    video_path = str(video_path)
    frames = []
    container = av.open(video_path, "r")
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format=target_format))
    finally:
        container.close()
        del container
        _depth_decode_counter += 1
        if _depth_decode_counter % _DEPTH_DECODE_GC_INTERVAL == 0:
            _gc.collect()

    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")

    if vcodec == "ffv1":
        return np.stack(frames, axis=0).astype(np.uint16, copy=False)
    # libx264rgb: unpack uint16 from (R<<8)|G
    rgb = np.stack(frames, axis=0)
    return (rgb[..., 0].astype(np.uint16) << 8) | rgb[..., 1].astype(np.uint16)


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
