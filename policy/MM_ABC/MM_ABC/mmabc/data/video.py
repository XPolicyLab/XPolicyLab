"""RGB frame extraction from LeRobot video shards using bounded PyAV pools."""

from __future__ import annotations

import os
import signal
import threading
from collections import OrderedDict

import numpy as np

from mmabc.data.pagecache import drop_page_cache

try:
    import av

    av.logging.set_level(av.logging.ERROR)
    _HAS_AV = True
except Exception:  # pragma: no cover
    _HAS_AV = False

# Seeking lands on a keyframe at or before the target, so a bounded forward scan
# is expected. This caps the scan for shards whose timestamps are inconsistent.
_MAX_SCAN_SECONDS = 4.0

# Bound each decode to avoid stalling distributed training; zero disables the watchdog.
_DECODE_TIMEOUT = float(os.environ.get("MMABC_DECODE_TIMEOUT", "60"))


class _DecodeTimeout(Exception):
    pass


def _decode_timeout_handler(signum, frame):  # noqa: ARG001
    raise _DecodeTimeout()


class VideoPool:
    """Bounded LRU pool of PyAV containers, one pool per process."""

    # Bound decoder buffers per worker by limiting open containers.
    def __init__(
        self,
        max_open: int = int(os.environ.get("MMABC_VIDEO_POOL", "8")),
        # 224px frames decode in well under a millisecond; frame threading only
        # adds hand-off latency at that size.
        threads: int = int(os.environ.get("MMABC_VIDEO_THREADS", "1")),
    ) -> None:
        self.max_open = max_open
        self.threads = threads
        self._pool: OrderedDict[str, tuple] = OrderedDict()
        self._lock = threading.Lock()

    def _acquire(self, path: str):
        if not _HAS_AV:
            raise RuntimeError("PyAV is required for video decoding")
        with self._lock:
            entry = self._pool.pop(path, None)
            if entry is not None:
                self._pool[path] = entry
                return entry

        container = av.open(path)
        stream = container.streams.video[0]
        if self.threads > 1:
            stream.thread_type = "AUTO"
            stream.thread_count = self.threads
        entry = (container, stream, float(stream.time_base))
        with self._lock:
            self._pool[path] = entry
            while len(self._pool) > self.max_open:
                _, old = self._pool.popitem(last=False)
                try:
                    old[0].close()
                except Exception:
                    pass
        return entry

    def _evict(self, path: str) -> None:
        """Drop a container from the pool; used after a timed-out decode leaves
        it in an unknown state."""
        with self._lock:
            entry = self._pool.pop(path, None)
        if entry is not None:
            try:
                entry[0].close()
            except Exception:
                pass

    def read(self, path: str, timestamps: list[float]) -> np.ndarray:
        """Frames nearest the given presentation times, in the requested order.

        Returns (len(timestamps), H, W, 3) uint8.
        """
        container, stream, time_base = self._acquire(path)
        wanted = sorted(set(float(t) for t in timestamps))
        first, last = wanted[0], wanted[-1]

        # SIGALRM works in a worker main thread; DataLoader timeout covers other contexts.
        armed = False
        if _DECODE_TIMEOUT > 0 and threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGALRM, _decode_timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, _DECODE_TIMEOUT)
                armed = True
            except (ValueError, OSError):
                armed = False

        # Retain only the nearest frame per timestamp so decoder buffers cannot accumulate.
        best_frame: list = [None] * len(wanted)
        best_dt = [float("inf")] * len(wanted)
        try:
            with self._lock:
                container.seek(
                    int(max(first, 0.0) / time_base), stream=stream, backward=True, any_frame=False
                )
                remaining = list(wanted)
                gen = container.decode(stream)
                try:
                    for frame in gen:
                        if frame.pts is None:
                            continue
                        t = float(frame.pts) * time_base
                        for i, wt in enumerate(wanted):
                            dt = abs(t - wt)
                            if dt < best_dt[i]:
                                best_dt[i] = dt
                                best_frame[i] = frame
                        while remaining and t >= remaining[0]:
                            remaining.pop(0)
                        if not remaining or t > last + _MAX_SCAN_SECONDS:
                            break
                finally:
                    # Close the decode generator to release in-flight FFmpeg buffers after an early break.
                    gen.close()
        except _DecodeTimeout:
            self._evict(path)
            raise RuntimeError(
                f"video decode exceeded {_DECODE_TIMEOUT:.0f}s (corrupt frame?): {path}"
            )
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0)
            # Optionally release clean shard pages while retaining the pooled decoder.
            drop_page_cache(path)

        if all(f is None for f in best_frame):
            raise RuntimeError(f"decoded no frames for {timestamps} from {path}")

        # Every wanted timestamp has a nearest candidate whenever at least one
        # frame decoded (each frame is compared against all wanted times), so map
        # each requested time -- duplicates included -- back to its wanted slot.
        wanted_pos = {wt: i for i, wt in enumerate(wanted)}
        return [best_frame[wanted_pos[float(t)]] for t in timestamps]

    def clear(self) -> None:
        with self._lock:
            for entry in self._pool.values():
                try:
                    entry[0].close()
                except Exception:
                    pass
            self._pool.clear()


_POOL: VideoPool | None = None


def pool() -> VideoPool:
    """Process-local container pool. Dataloader workers each get their own."""
    global _POOL
    if _POOL is None:
        _POOL = VideoPool()
    return _POOL


def timestamp_of(from_timestamp: float, frame_in_episode: int, fps: float) -> float:
    """Presentation time of a frame, offset by the episode's start in the shard."""
    return float(from_timestamp) + float(frame_in_episode) / float(fps)


def resize_for_model(rgb: np.ndarray, size: int) -> np.ndarray:
    """Squash an HxWx3 uint8 frame to ``size x size``, keeping the full field of view.

    The one resize used both when converting Mobile episodes and on live camera
    frames at deployment, so the model sees the same pixels in both.
    """
    import cv2

    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[:2] == (size, size):
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def to_square(frame, size: int) -> np.ndarray:
    """Short-side resize to `size` then centre crop, resizing inside swscale.

    Doing the scale during colour conversion rather than afterwards matters:
    the corpus contains 1920x1200 shards, and converting those to full-size
    RGB before shrinking costs several times more than the decode itself.
    """
    w, h = frame.width, frame.height
    if w == size and h == size:
        return frame.to_ndarray(format="rgb24")
    scale = size / min(w, h)
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    arr = frame.reformat(width=nw, height=nh, format="rgb24").to_ndarray()
    top, left = (nh - size) // 2, (nw - size) // 2
    return np.ascontiguousarray(arr[top : top + size, left : left + size])
