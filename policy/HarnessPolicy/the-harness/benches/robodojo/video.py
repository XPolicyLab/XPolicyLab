"""Encode RoboDojo camera frames as an MP4 without importing Isaac."""

import shutil
import subprocess
import tempfile
from pathlib import Path


def _ffmpeg_command(ffmpeg, width, height, fps, path):
    return [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-an", "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-movflags", "+faststart",
        str(path),
    ]


class MotionClip:
    """One episode's motion video, streamed into one ffmpeg process per camera.

    Each frame goes to its camera's encoder as it is captured and is not kept, so
    memory stays bounded however long the episode runs (the uncompressed frames of a
    long episode do not fit in RAM). finish() returns the MP4 bytes per camera;
    discard() drops a clip nobody will read. A failure is raised from finish().
    """

    def __init__(self):
        self._encoders = {}  # camera -> (process, path, width, height)
        self._directory = None
        self._error = None

    def __bool__(self):
        return bool(self._encoders)

    def add(self, camera: str, frame, fps: float) -> None:
        """Append one RGB uint8 frame of shape (height, width, 3)."""
        if self._error is not None:
            return
        height, width = int(frame.shape[0]), int(frame.shape[1])
        try:
            if camera not in self._encoders:
                self._start(camera, width, height, fps)
            process, _path, w, h = self._encoders[camera]
            if (width, height) != (w, h):
                raise ValueError(f"{camera} frame size changed from {w}x{h} to {width}x{height}")
            process.stdin.write(frame.tobytes())
        except (OSError, ValueError, RuntimeError) as error:
            self._error = f"{camera}: {type(error).__name__}: {error}"

    def _start(self, camera, width, height, fps):
        if width < 2 or height < 2 or width % 2 or height % 2:
            raise ValueError("Video dimensions must be positive even numbers")
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            raise ValueError("Video fps must be positive")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to encode a RoboDojo motion clip")
        if self._directory is None:
            self._directory = tempfile.mkdtemp(prefix="robodojo-clip-")
        path = Path(self._directory) / f"{camera}.mp4"
        log = open(Path(self._directory) / f"{camera}.log", "wb")
        try:
            process = subprocess.Popen(_ffmpeg_command(ffmpeg, width, height, float(fps), path),
                                       stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        finally:
            log.close()
        self._encoders[camera] = (process, path, width, height)

    def finish(self) -> dict[str, bytes]:
        """Close the encoders and return the MP4 of every camera that received frames."""
        try:
            videos = {}
            for camera, (process, path, _w, _h) in self._encoders.items():
                try:
                    process.stdin.close()
                except OSError:
                    pass
                code = process.wait(timeout=120)
                if self._error is None and code != 0:
                    detail = (path.parent / f"{camera}.log").read_text(errors="replace").strip()
                    self._error = f"{camera}: {detail[:500] or f'ffmpeg exited with {code}'}"
                if self._error is None:
                    payload = path.read_bytes() if path.is_file() else b""
                    if len(payload) < 12 or payload[4:8] != b"ftyp":
                        self._error = f"{camera}: ffmpeg did not produce an MP4"
                    videos[camera] = payload
            if self._error is not None:
                raise RuntimeError(self._error)
            return videos
        finally:
            self.discard()

    def discard(self) -> None:
        """Stop the encoders and delete their files."""
        for process, _path, _w, _h in self._encoders.values():
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
        self._encoders = {}
        if self._directory is not None:
            shutil.rmtree(self._directory, ignore_errors=True)
            self._directory = None
        self._error = None


def encode_mp4(frames: list[bytes], width: int, height: int, fps: float) -> bytes:
    """Encode tightly packed RGB frames. Playback uses ``fps`` for every control frame."""
    if width < 2 or height < 2 or width % 2 or height % 2:
        raise ValueError("Video dimensions must be positive even numbers")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("Video fps must be positive")
    if not frames:
        raise ValueError("Video needs at least one frame")
    expected = width * height * 3
    for frame in frames:
        if len(frame) != expected:
            raise ValueError("RGB byte count does not match the video size")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode a RoboDojo motion clip")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.mp4"
        result = subprocess.run(
            _ffmpeg_command(ffmpeg, width, height, fps, path),
            input=b"".join(frames),
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0 or not path.is_file():
            detail = result.stderr.decode(errors="replace").strip() or "ffmpeg failed"
            raise RuntimeError(detail[:500])
        payload = path.read_bytes()
    if len(payload) < 12 or payload[4:8] != b"ftyp":
        raise RuntimeError("ffmpeg did not produce an MP4")
    return payload
