from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class EpisodeRecorder:
    def __init__(self, env: Any, plugins: Sequence[Any], fps: float = 30.0) -> None:
        if not callable(getattr(env, "get_obs", None)):
            raise TypeError("env must provide a callable get_obs() method")

        fps = float(fps)
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        self.m_env = env
        self.m_plugins = tuple(plugins)
        self.m_fps = fps
        self.m_thread: threading.Thread | None = None
        self.m_stop_event = threading.Event()
        self.m_recording_error: BaseException | None = None
        self.m_sample_count = 0

    @property
    def sample_count(self) -> int:
        return self.m_sample_count

    def start(self, episode_dir: str | Path) -> None:
        if self.m_thread is not None:
            raise RuntimeError("episode recorder is already recording")

        episode_dir = Path(episode_dir)
        for plugin in self.m_plugins:
            plugin.start(episode_dir)

        self.m_recording_error = None
        self.m_sample_count = 0
        self.m_stop_event.clear()
        self.m_thread = threading.Thread(target=self._record_loop, name="EpisodeRecorderThread", daemon=True)
        self.m_thread.start()

    def record_obs(self, obs: dict[str, Any]) -> None:
        for plugin in self.m_plugins:
            plugin.record_obs(obs)
        self.m_sample_count += 1

    def stop(self) -> dict[str, Any]:
        self._stop_thread()
        if self.m_recording_error is not None:
            self.abort()
            self._raise_recording_error()

        return {
            plugin.name: plugin.stop()
            for plugin in self.m_plugins
        }

    def abort(self) -> None:
        self._stop_thread()
        for plugin in self.m_plugins:
            plugin.abort()

    def _record_loop(self) -> None:
        interval = 1.0 / self.m_fps
        next_sample_time = time.perf_counter()

        try:
            while not self.m_stop_event.is_set():
                wait_time = next_sample_time - time.perf_counter()
                if wait_time > 0 and self.m_stop_event.wait(wait_time):
                    break

                self.record_obs(self.m_env.get_obs())
                next_sample_time += interval
                now = time.perf_counter()
                if next_sample_time < now:
                    next_sample_time = now + interval
        except BaseException as exc:
            self.m_recording_error = exc
            self.m_stop_event.set()

    def _stop_thread(self) -> None:
        if self.m_thread is None:
            return

        self.m_stop_event.set()
        self.m_thread.join()
        self.m_thread = None

    def _raise_recording_error(self) -> None:
        assert self.m_recording_error is not None
        raise RuntimeError(f"episode recorder thread failed: {self.m_recording_error}") from self.m_recording_error


@dataclass(frozen=True)
class VideoRecorderConfig:
    fps: float = 30.0
    camera_names: tuple[str, ...] | None = None
    color_order: str = "bgr"
    crf: int = 0


@dataclass(frozen=True)
class TrajectoryRecorderConfig:
    camera_names: tuple[str, ...] | None = None


class FFmpegVideoWriter:
    def __init__(
        self,
        path: Path,
        first_frame: np.ndarray,
        fps: float,
        color_order: str,
        crf: int,
    ) -> None:
        first_frame = validate_video_frame(first_frame)
        height, width = first_frame.shape[:2]
        pix_fmt = "rgb24" if color_order == "rgb" else "bgr24"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-s",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv444p",
            "-crf",
            str(crf),
            str(path),
        ]

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg executable not found") from exc

        assert proc.stdin is not None
        self.path = path
        self.expected_hw = (height, width)
        self.m_proc = proc
        self.m_closed = False

    def write_validated(self, frame: np.ndarray) -> None:
        if self.m_closed:
            raise RuntimeError(f"cannot write to closed video writer: {self.path}")

        assert self.m_proc.stdin is not None
        try:
            self.m_proc.stdin.write(memoryview(frame))
        except BaseException as exc:
            self.abort()
            raise RuntimeError(f"ffmpeg video encoding failed for {self.path}: {exc}") from exc

    def close(self) -> None:
        if self.m_closed:
            return

        stderr = b""
        return_code = 0
        try:
            if self.m_proc.stdin is not None and not self.m_proc.stdin.closed:
                with suppress(Exception):
                    self.m_proc.stdin.close()
            stderr = self.m_proc.stderr.read() if self.m_proc.stderr is not None else b""
            return_code = self.m_proc.wait()
        finally:
            if self.m_proc.stderr is not None and not self.m_proc.stderr.closed:
                with suppress(Exception):
                    self.m_proc.stderr.close()
            self.m_closed = True

        if return_code != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg video encoding failed for {self.path}: {message}")

    def abort(self) -> None:
        if self.m_closed:
            return

        with suppress(Exception):
            if self.m_proc.stdin is not None and not self.m_proc.stdin.closed:
                self.m_proc.stdin.close()
        with suppress(Exception):
            self.m_proc.kill()
        with suppress(Exception):
            self.m_proc.wait()
        if self.m_proc.stderr is not None and not self.m_proc.stderr.closed:
            with suppress(Exception):
                self.m_proc.stderr.close()
        self.m_closed = True


class VideoRecorderPlugin:
    name = "video"
    version = "0.2"

    def __init__(self, config: VideoRecorderConfig | None = None) -> None:
        self.m_config = config or VideoRecorderConfig()
        if self.m_config.color_order not in {"rgb", "bgr"}:
            raise ValueError("color_order must be 'rgb' or 'bgr'")
        if not 0 <= int(self.m_config.crf) <= 51:
            raise ValueError("crf must be between 0 and 51")

        self.m_output_dir: Path | None = None
        self.m_writers: dict[str, FFmpegVideoWriter] = {}
        self.m_recorded_camera_names: tuple[str, ...] | None = None
        self.m_frame_count = 0

    def start(self, episode_dir: str | Path) -> None:
        self.m_output_dir = Path(episode_dir) / "recorder" / self.name
        self.m_output_dir.mkdir(parents=True, exist_ok=True)
        self.m_writers = {}
        self.m_recorded_camera_names = None
        self.m_frame_count = 0
        self._write_manifest(status="recording", outputs=[])

    def record_obs(self, obs: dict[str, Any]) -> None:
        frames = extract_camera_frames(
            obs,
            configured_camera_names=self.m_config.camera_names,
            recorded_camera_names=self.m_recorded_camera_names,
        )
        if self.m_recorded_camera_names is None:
            self.m_recorded_camera_names = tuple(frames)

        for camera_name, frame in frames.items():
            writer = self.m_writers.get(camera_name)
            expected_hw = None if writer is None else writer.expected_hw
            frames[camera_name] = validate_video_frame(frame, expected_hw=expected_hw)

        for camera_name, frame in frames.items():
            if camera_name not in self.m_writers:
                self.m_writers[camera_name] = self._open_writer(camera_name, frame)
            self.m_writers[camera_name].write_validated(frame)

        self.m_frame_count += 1

    def stop(self) -> dict[str, Any]:
        self._close_writers()
        if self.m_frame_count == 0 or not self.m_recorded_camera_names:
            raise ValueError("no camera samples recorded")
        assert self.m_output_dir is not None

        outputs = [
            {
                "name": camera_name,
                "type": "rgb_video",
                "path": f"{camera_name}.mp4",
                "fps": self.m_config.fps,
                "frame_count": self.m_frame_count,
                "encoding": "h264",
            }
            for camera_name in self.m_recorded_camera_names
        ]
        return self._write_manifest(status="committed", outputs=outputs)

    def abort(self) -> None:
        self._abort_writers()
        if self.m_output_dir is not None:
            shutil.rmtree(self.m_output_dir, ignore_errors=True)
            self.m_output_dir = None

    def _open_writer(self, camera_name: str, frame: np.ndarray) -> FFmpegVideoWriter:
        assert self.m_output_dir is not None
        return FFmpegVideoWriter(
            self.m_output_dir / f"{camera_name}.mp4",
            first_frame=frame,
            fps=self.m_config.fps,
            color_order=self.m_config.color_order,
            crf=self.m_config.crf,
        )

    def _close_writers(self) -> None:
        first_error: BaseException | None = None
        for writer in self.m_writers.values():
            try:
                writer.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self.m_writers = {}

        if first_error is not None:
            raise first_error

    def _abort_writers(self) -> None:
        for writer in self.m_writers.values():
            writer.abort()
        self.m_writers = {}

    def _write_manifest(self, status: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.m_output_dir is not None
        manifest = {
            "plugin": self.name,
            "version": self.version,
            "status": status,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": outputs,
        }
        (self.m_output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest


class TrajectoryRecorderPlugin:
    name = "trajectory"
    version = "0.1"

    def __init__(self, config: TrajectoryRecorderConfig | None = None) -> None:
        self.m_config = config or TrajectoryRecorderConfig()
        self.m_output_dir: Path | None = None
        self.m_samples: list[dict[str, Any]] = []
        self.m_camera_names: tuple[str, ...] | None = None

    def start(self, episode_dir: str | Path) -> None:
        self.m_output_dir = Path(episode_dir) / "recorder" / self.name
        self.m_output_dir.mkdir(parents=True, exist_ok=True)
        self.m_samples = []
        self.m_camera_names = None
        self._write_manifest(status="recording", outputs=[])

    def record_obs(self, obs: dict[str, Any]) -> None:
        sample = {
            "timestamp_ns": time.time_ns(),
            "state": {
                name: np.asarray(value)
                for name, value in obs["state"].items()
            },
            "vision": {},
        }
        frames = extract_camera_frames(
            obs,
            configured_camera_names=self.m_config.camera_names,
            recorded_camera_names=self.m_camera_names,
        )
        if self.m_camera_names is None:
            self.m_camera_names = tuple(frames)

        for camera_name, frame in frames.items():
            frame = validate_video_frame(frame)
            camera_obs = obs["vision"][camera_name]
            sample["vision"][camera_name] = {
                "color": frame,
                "shape": np.asarray(frame.shape, dtype=np.int32),
                "intrinsic_matrix": np.asarray(camera_obs.get("intrinsic_matrix", []), dtype=np.float32),
                "extrinsics_matrix": np.asarray(camera_obs.get("extrinsics_matrix", []), dtype=np.float32),
            }

        self.m_samples.append(sample)

    def stop(self) -> dict[str, Any]:
        if not self.m_samples:
            raise ValueError("no trajectory samples recorded")
        assert self.m_output_dir is not None

        hdf5_path = self.m_output_dir / "trajectory.hdf5"
        self._write_hdf5(hdf5_path)
        outputs = [
            {
                "name": "trajectory",
                "type": "xone_hdf5_trajectory",
                "path": hdf5_path.name,
                "sample_count": len(self.m_samples),
            }
        ]
        return self._write_manifest(status="committed", outputs=outputs)

    def abort(self) -> None:
        if self.m_output_dir is not None:
            shutil.rmtree(self.m_output_dir, ignore_errors=True)
            self.m_output_dir = None
        self.m_samples = []
        self.m_camera_names = None

    def _write_hdf5(self, hdf5_path: Path) -> None:
        import h5py

        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("timestamps", data=np.asarray([sample["timestamp_ns"] for sample in self.m_samples], dtype=np.int64))

            vision_group = f.create_group("vision")
            assert self.m_camera_names is not None
            for camera_name in self.m_camera_names:
                camera_group = vision_group.create_group(camera_name)
                colors = np.stack([sample["vision"][camera_name]["color"] for sample in self.m_samples], axis=0)
                camera_group.create_dataset("colors", data=colors)
                camera_group.create_dataset("shape", data=np.asarray(colors.shape[1:], dtype=np.int32))

                intrinsic = self.m_samples[0]["vision"][camera_name]["intrinsic_matrix"]
                extrinsic = self.m_samples[0]["vision"][camera_name]["extrinsics_matrix"]
                if intrinsic.size:
                    camera_group.create_dataset("intrinsic_matrix", data=intrinsic)
                if extrinsic.size:
                    camera_group.create_dataset("extrinsics_matrix", data=extrinsic)

            state_group = f.create_group("state")
            for state_name in self.m_samples[0]["state"]:
                state_group.create_dataset(
                    xone_state_dataset_name(state_name),
                    data=np.stack([sample["state"][state_name] for sample in self.m_samples], axis=0),
                )

    def _write_manifest(self, status: str, outputs: list[dict[str, Any]]) -> dict[str, Any]:
        assert self.m_output_dir is not None
        manifest = {
            "plugin": self.name,
            "version": self.version,
            "status": status,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "outputs": outputs,
        }
        (self.m_output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return manifest


def extract_camera_frames(
    obs: Mapping[str, Any],
    configured_camera_names: tuple[str, ...] | None,
    recorded_camera_names: tuple[str, ...] | None,
) -> dict[str, np.ndarray]:
    vision = obs["vision"]
    selected_names = recorded_camera_names or configured_camera_names or tuple(vision.keys())
    if not selected_names:
        raise ValueError("no camera frames found in obs['vision']")
    return {
        camera_name: vision[camera_name]["color"]
        for camera_name in selected_names
    }


def validate_video_frame(frame: Any, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"video frame must have shape (H, W, 3), got {frame.shape}")
    if expected_hw is not None and frame.shape[:2] != expected_hw:
        raise ValueError(f"video frame shape changed from {expected_hw} to {frame.shape[:2]}")
    if frame.dtype != np.uint8:
        raise ValueError(f"video frame dtype must be uint8, got {frame.dtype}")
    return np.ascontiguousarray(frame)


def xone_state_dataset_name(state_name: str) -> str:
    if state_name.endswith("_state"):
        return f"{state_name}s"
    if state_name.endswith("_pose"):
        return f"{state_name}s"
    return state_name


def build_episode_recorder(
    env: Any,
    fps: float = 30.0,
    record_video: bool = False,
    record_trajectory: bool = False,
    camera_names: Sequence[str] | None = None,
    color_order: str = "bgr",
    crf: int = 0,
) -> EpisodeRecorder:
    camera_tuple = tuple(camera_names) if camera_names is not None else None
    plugins = []
    if record_video:
        plugins.append(
            VideoRecorderPlugin(
                VideoRecorderConfig(
                    fps=float(fps),
                    camera_names=camera_tuple,
                    color_order=color_order,
                    crf=int(crf),
                )
            )
        )
    if record_trajectory:
        plugins.append(TrajectoryRecorderPlugin(TrajectoryRecorderConfig(camera_names=camera_tuple)))
    return EpisodeRecorder(env, plugins, fps=fps)


def build_video_recorder(
    env: Any,
    fps: float = 30.0,
    camera_names: Sequence[str] | None = None,
    color_order: str = "bgr",
    crf: int = 0,
) -> EpisodeRecorder:
    return build_episode_recorder(
        env,
        fps=fps,
        record_video=True,
        camera_names=camera_names,
        color_order=color_order,
        crf=crf,
    )
