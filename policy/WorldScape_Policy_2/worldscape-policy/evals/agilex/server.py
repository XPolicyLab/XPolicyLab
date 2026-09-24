"""Native Manifold transport for AgileX evaluation."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from evals.agilex.observation_adapter import (
    manifold_eef6d_state,
    manifold_joint_state,
    sample_camera_frames,
    sample_video,
)

try:
    from manifold_msg.api import http_protocol
except ImportError as error:  # pragma: no cover - optional robot dependency
    http_protocol = None  # type: ignore[assignment]
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

_CAMERAS = (
    ("head", "cam_high"),
    ("left_wrist", "cam_left"),
    ("right_wrist", "cam_right"),
)


class NativeManifoldRobot:
    """Manifold HTTP transport implementing the historical adapter methods."""

    context_video_poll_mode = True

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 11451,
        node_name: str = "wsp",
        model_width: int = 640,
        model_height: int = 480,
        use_wma_layout: bool = False,
    ) -> None:
        if http_protocol is None:
            raise ImportError(
                "AgileX Manifold transport requires the optional manifold_msg "
                "package available in the robot deployment environment"
            ) from _IMPORT_ERROR
        self._server = http_protocol.Server(
            host=host, port=port, node_name=node_name
        )
        if use_wma_layout:
            self._server.updateModelInfo(model_width, model_height, 4, 0.1)
        else:
            self._server.updateModelInfo(model_width, model_height, 1, 1)
        self._context_frames = {output: [] for _, output in _CAMERAS}

    def reset_episode(self) -> None:
        for frames in self._context_frames.values():
            frames.clear()

    def read_observation(
        self,
        use_history: bool = False,
        num_history_frames: int = 4,
        action_mode: str = "eef",
    ):
        while True:
            value = self.try_read_observation(
                use_history=use_history,
                num_history_frames=num_history_frames,
                action_mode=action_mode,
            )
            if value is not None:
                return value
            time.sleep(0.001)

    def try_read_observation(
        self,
        use_history: bool = False,
        num_history_frames: int = 4,
        action_mode: str = "eef",
    ):
        if not self._server.model_input_queue:
            return None
        observation = self._server.model_input_queue[-1]
        self._server.model_input_queue.clear()
        cameras = sample_camera_frames(
            (
                observation["img_front"],
                observation["img_left"],
                observation["img_right"],
            ),
            use_history=use_history,
            num_frames=num_history_frames,
        )
        state = (
            manifold_joint_state(observation)
            if action_mode == "joint"
            else manifold_eef6d_state(observation)
        )
        return cameras[0], cameras[1], cameras[2], state, observation["timestamp"]

    def send_end_pose_action(
        self,
        timestamp: Any,
        action_rate: int,
        left: list[list[float]],
        right: list[list[float]],
    ) -> None:
        self._server.send_end_pose_action(
            timestamp, action_rate, left, right, is_euler=False
        )

    def send_joint_state_action(
        self,
        timestamp: Any,
        action_rate: int,
        left: list[list[float]],
        right: list[list[float]],
    ) -> None:
        self._server.send_joint_state_action(timestamp, action_rate, left, right)

    def read_context_video(
        self, num_frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = {output: [] for _, output in _CAMERAS}
        errors: list[BaseException] = []

        def collect(camera: str, output: str) -> None:
            try:
                while True:
                    value = self._server.wait_video_observation(camera)
                    if value.get("frame") is not None:
                        frames[output].append(np.asarray(value["frame"]))
                    if bool(value.get("is_end", False)):
                        return
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [
            threading.Thread(target=collect, args=spec, daemon=True)
            for spec in _CAMERAS
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("Failed to read Manifold context video") from errors[0]
        return tuple(
            sample_video(frames[output], num_frames) for _, output in _CAMERAS
        )  # type: ignore[return-value]

    def try_read_context_video(
        self, num_frames: int
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, bool]:
        head_ended = False
        for camera, output in _CAMERAS:
            while True:
                with self._server.video_lock:
                    if self._server.video_queues[camera]:
                        value = self._server.video_queues[camera].popleft()
                        frame = self._server._decode_video_frame(value)
                        ended = False
                    elif self._server.video_end_flags.get(camera, False):
                        if self._server.video_user_consumed.get(camera, False):
                            break
                        self._server.video_user_consumed[camera] = True
                        frame = None
                        ended = True
                    else:
                        break
                if frame is not None:
                    self._context_frames[output].append(np.asarray(frame))
                if ended and output == "cam_high":
                    head_ended = True
        if not head_ended:
            return None, False
        if not self._context_frames["cam_high"]:
            self.reset_episode()
            return None, False
        videos = tuple(
            sample_video(self._context_frames[output], num_frames)
            for _, output in _CAMERAS
        )
        self.reset_episode()
        return videos, True  # type: ignore[return-value]

    def close(self) -> None:
        close = getattr(self._server, "close", None)
        if callable(close):
            close()
