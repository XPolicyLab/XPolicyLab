"""Apply optional H.264 conditioning to prepared inference observations."""

import io
import subprocess

import numpy as np


def h264_roundtrip(
    frames: np.ndarray,
    crf: int,
    *,
    repeat_frames: int,
    threads: int,
) -> np.ndarray:
    """Encode copies of each RGB observation and retain its last decoded frame.

    Spatial preprocessing must be complete before this operation, as it is for
    the training video asset writer. Each input frame starts a fresh stream of
    identical copies; no history or future observations are used. The output
    need not equal a frame decoded from a historical training video.
    Input and output retain HWC or THWC layout and uint8 RGB values.
    """
    single_frame = frames.ndim == 3
    video = frames[None] if single_frame else frames
    frame_count, height, width, channels = video.shape
    if frame_count == 0 or height % 2 or width % 2:
        raise ValueError("H.264/YUV420P requires nonempty frames with even height and width")
    if channels != 3 or video.dtype != np.uint8:
        raise ValueError("H.264 observation encoding requires uint8 RGB frames")
    result = np.stack([_encode_repeated_frame(frame, crf, repeat_frames, threads) for frame in video])
    return result[0] if single_frame else result


def _encode_repeated_frame(frame: np.ndarray, crf: int, count: int, threads: int) -> np.ndarray:
    """Warm the codec with copies of one observation, without RGB batch copies."""
    import av

    height, width = frame.shape[:2]
    encoded = _run_ffmpeg(
        [
            "-filter_threads", "1", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", "30", "-i", "pipe:0",
            "-vf", f"format=yuv420p,loop=loop={count - 1}:size=1:start=0",
            "-frames:v", str(count), "-an", "-c:v", "libx264", "-threads", str(threads),
            "-crf", str(crf), "-pix_fmt", "yuv420p", "-f", "h264", "pipe:1",
        ],
        frame.tobytes(),
    )
    decoded_count = 0
    last = None
    with av.open(io.BytesIO(encoded), format="h264") as container:
        for last in container.decode(video=0):
            decoded_count += 1
        if decoded_count != count:
            raise RuntimeError(f"H.264 warm-up decoded {decoded_count} frames; expected {count}")
        result = last.to_ndarray(format="rgb24")
    if result.shape != frame.shape:
        raise RuntimeError(f"H.264 warm-up returned shape {result.shape}; expected {frame.shape}")
    return result.copy()


def _run_ffmpeg(arguments: list[str], payload: bytes) -> bytes:
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", *arguments],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "H.264 observation encoding failed: " + error.stderr.decode("utf-8", errors="replace").strip()
        ) from error
    return result.stdout
