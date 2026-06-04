# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import shutil
import subprocess
import uuid
import warnings
import weakref
from enum import Enum
from pathlib import Path

import numpy as np

__all__ = [
    "VideoBackendUnavailableError",
    "VideoEncodeError",
    "VideoFrameError",
    "VideoPixelFormat",
    "VideoWriter",
    "VideoWriterError",
]


class VideoWriterError(RuntimeError):
    """Base error raised by :class:`VideoWriter`."""


class VideoBackendUnavailableError(VideoWriterError):
    """Raised when ffmpeg is unavailable in the current environment."""


class VideoFrameError(VideoWriterError):
    """Raised when a frame violates the writer input contract."""


class VideoEncodeError(VideoWriterError):
    """Raised when ffmpeg cannot start, encode, or finalize output."""


class VideoPixelFormat(str, Enum):
    """Supported raw input pixel formats for :class:`VideoWriter`."""

    RGB24 = "rgb24"
    BGR24 = "bgr24"


class VideoWriter:
    """Write an existing sequence of color frames to one finalized video file.

    Use ``VideoWriter`` when your code already produces frames as
    ``(H, W, 3)`` color images and you want a small, explicit API for
    turning those frames into a video artifact. It is meant for callers that
    want to decide when recording starts and stops, but do not want to manage
    ffmpeg process startup, frame streaming, or failed-output cleanup
    themselves.

    The caller still owns frame production, output-path selection, and
    recording policy. ``VideoWriter`` owns the encoding session: ``open()``
    starts a logical session, the first successful ``write_frame()`` call
    starts ffmpeg, and ``close()`` finalizes the result. The first written
    frame fixes the frame size for the session, and later frames must match
    that size.

    To avoid exposing partial results as completed output, encoded data is
    written to a hidden staging file first and published to ``output_path``
    only after ``close()`` succeeds. With the default
    ``output_pixel_format="yuv420p"``, frame width and height must both be
    even.

    Examples:
        Write one video directly::

            with VideoWriter(
                "episode.mp4", pixel_format="rgb24", fps=10
            ) as writer:
                writer.write_frame(frame)

        Reuse one writer across sessions::

            writer = VideoWriter(pixel_format="rgb24", fps=10)
            with writer.open("episode_2.mp4") as opened_writer:
                opened_writer.write_frame(frame)

    Args:
        output_path (str | Path | None, optional): Output video path. If
            provided, opens a new logical session immediately, while still
            deferring ffmpeg startup until the first frame is written.
            Default is None.
        pixel_format (VideoPixelFormat | str, optional): Raw input pixel
            format of frames passed to ``write_frame()``. Default is
            ``VideoPixelFormat.RGB24``.
        fps (int, optional): Output frame rate. Default is 10.
        codec (str, optional): ffmpeg video codec name. Default is
            ``"libx264"``.
        crf (int, optional): ffmpeg CRF quality setting. Default is 23.
        output_pixel_format (str, optional): Encoded output pixel format.
            Default is ``"yuv420p"``.
        overwrite (bool, optional): Whether an existing target file may be
            replaced. Default is True.
    """

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        pixel_format: VideoPixelFormat | str = VideoPixelFormat.RGB24,
        fps: int = 10,
        codec: str = "libx264",
        crf: int = 23,
        output_pixel_format: str = "yuv420p",
        overwrite: bool = True,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}.")
        if not codec:
            raise ValueError("codec must be a non-empty string.")
        if not output_pixel_format:
            raise ValueError("output_pixel_format must be a non-empty string.")

        self.pixel_format = VideoPixelFormat(pixel_format)
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.output_pixel_format = output_pixel_format
        self.overwrite = overwrite

        self._output_path: Path | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._staging_output_path: Path | None = None
        self._proc_finalizer: weakref.finalize | None = None
        self._frame_size: tuple[int, int] | None = None
        self._frame_count = 0
        self._closed = True

        if output_path is not None:
            self.open(output_path)

    @property
    def frame_count(self) -> int:
        """Return the number of frames successfully written."""

        return self._frame_count

    @property
    def output_path(self) -> Path | None:
        """Return the current output video path, if configured."""

        return self._output_path

    @property
    def is_open(self) -> bool:
        """Return whether the writer currently has an open write session."""

        return (not self._closed) and self._output_path is not None

    @property
    def is_closed(self) -> bool:
        """Return whether the writer currently has no open write session."""

        return not self.is_open

    def open(self, output_path: str | Path | None = None) -> VideoWriter:
        """Start a new recording session.

        Use ``open()`` when reusing a writer for another output path or when
        the path is chosen later than construction time. It resets
        per-session state such as frame size and frame count, but still
        defers ffmpeg startup until the first ``write_frame()`` call.

        Args:
            output_path (str | Path | None, optional): Path for the new
                output video. If None, reuses the last configured path.

        Returns:
            VideoWriter: The opened writer itself so callers can use
                ``with writer.open(path) as opened_writer: ...``.

        Raises:
            VideoWriterError: If the writer is already open.
            ValueError: If ``output_path`` is None and the writer has never
                been configured with a path.
        """
        if self.is_open:
            raise VideoWriterError(
                "Cannot open VideoWriter because it is already open. "
                "Close it before opening a new session."
            )

        if output_path is not None:
            self._output_path = Path(output_path)
        elif self._output_path is None:
            raise ValueError(
                "output_path must be provided when opening VideoWriter "
                "without an existing path."
            )

        self._proc = None
        self._staging_output_path = None
        self._frame_size = None
        self._frame_count = 0
        self._closed = False
        return self

    def write_frame(self, frame: np.ndarray) -> None:
        """Add one frame to the current recording.

        Use ``write_frame()`` after opening a session and producing a frame
        that should become part of the output video. The first successful
        write starts ffmpeg lazily and fixes the frame size for the rest of
        the session. Later frames must keep the same width and height.

        Args:
            frame (np.ndarray): Frame with shape ``(H, W, 3)``. The frame is
                coerced to a contiguous ``uint8`` array before writing.

        Raises:
            VideoWriterError: If the writer is not open.
            VideoFrameError: If the frame shape is invalid, its size changes
                after the first write, or it violates the active output
                pixel-format constraints.
            VideoBackendUnavailableError: If ffmpeg is not available.
            VideoEncodeError: If ffmpeg cannot start or accept the frame.
        """
        if self.is_closed:
            raise VideoWriterError(
                "Cannot write a frame because VideoWriter is not open."
            )

        frame_np = self._normalize_frame(frame)
        frame_size = (int(frame_np.shape[1]), int(frame_np.shape[0]))

        if self._frame_size is not None and self._frame_size != frame_size:
            raise VideoFrameError(
                f"Expected frame size {self._frame_size}, got {frame_size}."
            )

        if self._proc is None:
            self._validate_frame_size_for_output_format(
                width=frame_size[0],
                height=frame_size[1],
            )
            self._start_process(width=frame_size[0], height=frame_size[1])

        path = self.output_path
        proc = self._proc
        if path is None or proc is None or proc.stdin is None:
            cleanup_errors = self._abort_open_session()
            raise VideoEncodeError(
                "ffmpeg stdin is unavailable for "
                f"{path}.{self._format_cleanup_error_suffix(cleanup_errors)}"
            )

        try:
            proc.stdin.write(frame_np.tobytes())
        except Exception as exc:
            cleanup_errors = self._abort_open_session()
            raise VideoEncodeError(
                "Failed to write a video frame to "
                f"{path}.{self._format_cleanup_error_suffix(cleanup_errors)}"
            ) from exc

        if self._frame_size is None:
            self._frame_size = frame_size
        self._frame_count += 1

    def close(self) -> None:
        """Finish the current recording and publish the final video file.

        Call ``close()`` after the last frame has been written. If ffmpeg was
        started, ``close()`` finalizes the encoder and then publishes the
        hidden staging file to ``output_path``. If the session never started,
        ``close()`` only cleans up any staged artifact. Calling ``close()``
        on an already closed writer is a no-op.

        Raises:
            VideoEncodeError: If finalization, staged-file cleanup, or final
                publish fails.
        """
        if self.is_closed:
            return
        self._closed = True

        proc, staging_output_path = self._take_proc()
        if proc is None:
            cleanup_errors = self._cleanup_staging_output_file(
                staging_output_path
            )
            if cleanup_errors:
                raise VideoEncodeError(
                    "Failed to clean up staged video output for "
                    f"{self.output_path}."
                    f"{self._format_cleanup_error_suffix(cleanup_errors)}"
                )
            return

        path = self.output_path
        stderr_output, return_code = self._finalize_process(
            proc,
            path=path,
            staging_output_path=staging_output_path,
        )

        if return_code != 0:
            stderr_text = stderr_output.decode(
                "utf-8", errors="ignore"
            ).strip()
            cleanup_errors = self._cleanup_staging_output_file(
                staging_output_path
            )
            detail = f" ffmpeg stderr: {stderr_text}" if stderr_text else ""
            raise VideoEncodeError(
                "ffmpeg exited with non-zero status "
                f"{return_code} while finalizing {path}.{detail}"
                f"{self._format_cleanup_error_suffix(cleanup_errors)}"
            )

        if path is None or staging_output_path is None:
            cleanup_errors = self._cleanup_staging_output_file(
                staging_output_path
            )
            raise VideoEncodeError(
                "VideoWriter has no staged output to publish for "
                f"{path}.{self._format_cleanup_error_suffix(cleanup_errors)}"
            )

        try:
            self._publish_staging_output_file(
                staging_output_path,
                output_path=path,
                overwrite=self.overwrite,
            )
        except Exception as exc:
            cleanup_errors = self._cleanup_staging_output_file(
                staging_output_path
            )
            raise VideoEncodeError(
                "Failed to publish finalized video output to "
                f"{path}.{self._format_cleanup_error_suffix(cleanup_errors)}"
            ) from exc

    def __enter__(self) -> VideoWriter:
        """Use the current open session inside a ``with`` block.

        The session must already be open through construction with
        ``output_path`` or through an earlier ``open()`` call.

        Raises:
            VideoWriterError: If no write session is open.
        """
        if self.is_closed:
            raise VideoWriterError(
                "Cannot enter VideoWriter context because it is not open. "
                "Call open() first or provide output_path at construction."
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Finish the current session when leaving a ``with`` block.

        Context-manager use is equivalent to calling ``close()`` on block
        exit. If the block body raises, ``__exit__`` still attempts to close
        the writer. Any ``VideoWriterError`` raised by ``close()`` is chained
        to the original body exception instead of replacing it.
        """
        if exc_type is None:
            self.close()
            return

        try:
            self.close()
        except VideoWriterError as close_exc:
            if exc is None:
                raise close_exc
            raise exc.with_traceback(tb) from close_exc

    def _abort_open_session(self) -> list[str]:
        # Abort means "this session is no longer recoverable": drop the
        # ffmpeg handle, mark the writer closed, and reap any staged output.
        self._closed = True
        self._frame_size = None
        proc, staging_output_path = self._take_proc()
        if proc is None:
            return self._cleanup_staging_output_file(staging_output_path)
        return self._discard_process(
            proc,
            staging_output_path=staging_output_path,
        )

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[2] != 3:
            raise VideoFrameError(
                f"Expected frame shape (H, W, 3), got {tuple(frame_np.shape)}."
            )
        if frame_np.dtype != np.uint8:
            frame_np = frame_np.astype(np.uint8)
        return np.ascontiguousarray(frame_np)

    def _validate_frame_size_for_output_format(
        self,
        *,
        width: int,
        height: int,
    ) -> None:
        if self.output_pixel_format != "yuv420p":
            return
        if width % 2 != 0 or height % 2 != 0:
            raise VideoFrameError(
                "output_pixel_format='yuv420p' requires even frame width "
                f"and height, got ({width}, {height})."
            )

    def _start_process(self, *, width: int, height: int) -> None:
        ffmpeg_binary = shutil.which("ffmpeg")
        if ffmpeg_binary is None:
            raise VideoBackendUnavailableError(
                "ffmpeg is not available in PATH."
            )

        path = self.output_path
        if path is None:
            raise VideoEncodeError(
                "VideoWriter has no configured output path."
            )
        if not self.overwrite and path.exists():
            raise VideoEncodeError(
                "Cannot start video output at "
                f"{path} because it already exists and overwrite=False."
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        staging_output_path = self._build_staging_output_path(path)
        try:
            proc = subprocess.Popen(
                [
                    ffmpeg_binary,
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    self.pixel_format.value,
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    "-",
                    "-pix_fmt",
                    self.output_pixel_format,
                    "-vcodec",
                    self.codec,
                    "-crf",
                    str(self.crf),
                    str(staging_output_path),
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            raise VideoEncodeError(
                f"Failed to start ffmpeg for {path}."
            ) from exc
        self._set_proc(
            proc,
            path=path,
            staging_output_path=staging_output_path,
        )

    @staticmethod
    def _finalize_process(
        proc: subprocess.Popen[bytes],
        *,
        path: Path | None,
        staging_output_path: Path | None,
    ) -> tuple[bytes, int]:
        stderr_output = b""
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            if proc.stderr is not None:
                stderr_output = proc.stderr.read()
            return_code = proc.wait()
        except Exception as exc:
            cleanup_errors = VideoWriter._discard_process(
                proc,
                staging_output_path=staging_output_path,
            )
            raise VideoEncodeError(
                "Failed to finalize video output at "
                f"{path}.{VideoWriter._format_cleanup_error_suffix(cleanup_errors)}"
            ) from exc
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except Exception as exc:
                raise VideoEncodeError(
                    "Failed to finalize video output at "
                    f"{path}. Cleanup details: stderr close failed: {exc}"
                ) from exc
        return stderr_output, return_code

    @staticmethod
    def _discard_process(
        proc: subprocess.Popen[bytes],
        *,
        staging_output_path: Path | None,
    ) -> list[str]:
        # This is the hard-cleanup path used after failed writes/finalize or
        # best-effort GC cleanup. It is intentionally tolerant: accumulate
        # cleanup problems instead of masking the original failure.
        cleanup_errors: list[str] = []

        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception as exc:
            cleanup_errors.append(f"stdin close failed: {exc}")

        if proc.poll() is None:
            try:
                proc.kill()
            except Exception as exc:
                cleanup_errors.append(f"kill failed: {exc}")

        try:
            proc.wait()
        except Exception as exc:
            cleanup_errors.append(f"wait failed: {exc}")

        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except Exception as exc:
                cleanup_errors.append(f"stderr close failed: {exc}")

        cleanup_errors.extend(
            VideoWriter._cleanup_staging_output_file(staging_output_path)
        )
        return cleanup_errors

    @staticmethod
    def _format_cleanup_error_suffix(cleanup_errors: list[str]) -> str:
        if not cleanup_errors:
            return ""
        return " Cleanup details: " + "; ".join(cleanup_errors)

    def _set_proc(
        self,
        proc: subprocess.Popen[bytes],
        *,
        path: Path | None,
        staging_output_path: Path | None,
    ) -> None:
        if self._proc is not None:
            raise RuntimeError(
                "Cannot replace an active ffmpeg process without cleanup."
            )
        finalizer = self._proc_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self._proc_finalizer = None
        self._proc = proc
        self._staging_output_path = staging_output_path
        # Register a safety-net cleanup for abandoned writers. The explicit
        # lifecycle API is still the primary contract; this only covers
        # callers that drop the object without closing it.
        self._proc_finalizer = weakref.finalize(
            self,
            self._cleanup_abandoned_process,
            proc,
            path,
            staging_output_path,
        )

    def _take_proc(
        self,
    ) -> tuple[subprocess.Popen[bytes] | None, Path | None]:
        # Transfer ownership of the live ffmpeg process and its staging path
        # out of the writer state so follow-up cleanup/publish runs exactly
        # once and the finalizer does not race with explicit close logic.
        proc = self._proc
        staging_output_path = self._staging_output_path
        self._proc = None
        self._staging_output_path = None
        finalizer = self._proc_finalizer
        if finalizer is not None and finalizer.alive:
            finalizer.detach()
        self._proc_finalizer = None
        return proc, staging_output_path

    @staticmethod
    def _cleanup_abandoned_process(
        proc: subprocess.Popen[bytes],
        path: Path | None,
        staging_output_path: Path | None,
    ) -> None:
        # Finalizers run outside the normal call path, so keep this branch
        # best-effort and warning-based instead of raising back into GC.
        warnings.warn(
            "VideoWriter was garbage-collected without close(); "
            f"cleaning up ffmpeg process for {path}.",
            ResourceWarning,
            stacklevel=2,
        )
        cleanup_errors = VideoWriter._discard_process(
            proc,
            staging_output_path=staging_output_path,
        )
        if cleanup_errors:
            warnings.warn(
                "VideoWriter cleanup on garbage collection encountered "
                + "; ".join(cleanup_errors),
                ResourceWarning,
                stacklevel=2,
            )

    @staticmethod
    def _cleanup_staging_output_file(
        staging_output_path: Path | None,
    ) -> list[str]:
        # Never touch the published target path here. Failed sessions only
        # clean up the hidden staging artifact owned by the active session.
        if staging_output_path is None:
            return []
        try:
            staging_output_path.unlink()
        except FileNotFoundError:
            return []
        except Exception as exc:
            return [f"staging output delete failed: {exc}"]
        return []

    @staticmethod
    def _build_staging_output_path(output_path: Path) -> Path:
        stem = output_path.stem or output_path.name
        suffix = output_path.suffix or ".tmp"
        while True:
            # Keep the staging file in the destination directory so the final
            # publish step stays on the same filesystem and can use atomic
            # rename/replace semantics.
            staging_output_path = output_path.with_name(
                f".{stem}.{uuid.uuid4().hex}{suffix}"
            )
            if not staging_output_path.exists():
                return staging_output_path

    @staticmethod
    def _publish_staging_output_file(
        staging_output_path: Path,
        *,
        output_path: Path,
        overwrite: bool,
    ) -> None:
        # Keep the staging file in the destination directory so the publish
        # step can use an atomic same-filesystem rename/replace.
        if overwrite:
            staging_output_path.replace(output_path)
            return
        if output_path.exists():
            raise FileExistsError(
                f"{output_path} already exists and overwrite=False."
            )
        staging_output_path.rename(output_path)
