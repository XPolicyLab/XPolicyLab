"""Store action videos beside an episode trace."""

import base64
import re
import shutil
import subprocess
from pathlib import Path

_VIDEO_PREFIX = "data:video/mp4;base64,"
_STEP_VIDEO = re.compile(r"^step-(\d+)-(.+)\.mp4$")


def decode_video_data_url(value: str) -> bytes:
    if not isinstance(value, str) or not value.startswith(_VIDEO_PREFIX):
        raise ValueError("Video must be an MP4 data URL")
    try:
        payload = base64.b64decode(value.removeprefix(_VIDEO_PREFIX), validate=True)
    except Exception as error:
        raise ValueError("Video data URL is not valid base64") from error
    if len(payload) < 12 or payload[4:8] != b"ftyp":
        raise ValueError("Video payload is not an MP4")
    return payload


def materialize_videos(directory: Path, step: int, videos: dict) -> dict[str, str]:
    """Write one MP4 per camera and return paths relative to the episode directory."""
    if not videos:
        return {}
    if type(step) is not int or isinstance(step, bool) or step < 0:
        raise ValueError("Video step must be a nonnegative integer")
    folder = directory / "videos"
    folder.mkdir(exist_ok=True)
    written = {}
    for camera, url in videos.items():
        if not isinstance(camera, str) or not camera or "/" in camera or camera in (".", ".."):
            raise ValueError("Video camera name is not a single path segment")
        payload = decode_video_data_url(url)
        name = f"step-{step:04d}-{camera}.mp4"
        (folder / name).write_bytes(payload)
        written[camera] = f"videos/{name}"
    return written


def write_episode_videos(directory: Path, videos: dict) -> dict[str, str]:
    """Write one whole-episode MP4 per camera and return paths relative to the episode directory."""
    if not videos:
        return {}
    folder = directory / "videos"
    folder.mkdir(exist_ok=True)
    written = {}
    for camera, url in videos.items():
        if not isinstance(camera, str) or not camera or "/" in camera or camera in (".", ".."):
            raise ValueError("Video camera name is not a single path segment")
        name = f"episode-{camera}.mp4"
        (folder / name).write_bytes(decode_video_data_url(url))
        written[camera] = f"videos/{name}"
    return written


def concat_episode_videos(directory: Path) -> dict[str, str]:
    """Join per-step clips into one file per camera when that can be done losslessly."""
    folder = directory / "videos"
    if not folder.is_dir():
        return {}
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in folder.glob("step-*.mp4"):
        match = _STEP_VIDEO.fullmatch(path.name)
        if match is None:
            continue
        groups.setdefault(match.group(2), []).append((int(match.group(1)), path))
    written = {}
    for camera, parts in groups.items():
        ordered = [path for _step, path in sorted(parts)]
        target = folder / f"{camera}.mp4"
        if len(ordered) == 1:
            shutil.copyfile(ordered[0], target)
        else:
            _concat_with_ffmpeg(ordered, target)
        written[camera] = f"videos/{camera}.mp4"
    return written


def redact_video_payloads(value):
    """Replace inline MP4 payloads with their size for terminal output."""
    if isinstance(value, str):
        if value.startswith(_VIDEO_PREFIX):
            return f"{_VIDEO_PREFIX}<{len(value) - len(_VIDEO_PREFIX)} chars>"
        return value
    if isinstance(value, list):
        return [redact_video_payloads(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_video_payloads(item) for key, item in value.items()}
    return value


def _concat_with_ffmpeg(parts: list[Path], target: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to join more than one video clip")
    listing = "".join(f"file '{path.resolve()}'\n" for path in parts)
    list_path = target.with_suffix(".txt")
    list_path.write_text(listing)
    try:
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-c", "copy", str(target)],
            capture_output=True, text=True, timeout=120, check=False,
        )
    finally:
        list_path.unlink(missing_ok=True)
    if result.returncode != 0 or not target.is_file():
        detail = (result.stderr or result.stdout or "ffmpeg failed").strip()
        raise RuntimeError(detail[:500])
