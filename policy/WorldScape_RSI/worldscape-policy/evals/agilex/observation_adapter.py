"""WorldScape-owned AgileX image, state, and action conversions."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np
import torch

from evals.common.checkpoint_runtime import CheckpointTransformBundle
from evals.common.protocols import ObservationAdapter
from scipy.spatial.transform import Rotation

from worldscape_policy.geometry import quaternion_pose_to_rotation6d
from worldscape_policy.types import InteractionMode, ObservationBatch, PromptBatch

POLICY_VIDEO_NATIVE_W = 320
POLICY_VIDEO_NATIVE_H = 160


def resize_image(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(image, (POLICY_VIDEO_NATIVE_W, POLICY_VIDEO_NATIVE_H))


def quaternion_pose_to_rot6d(pose: np.ndarray) -> np.ndarray:
    return quaternion_pose_to_rotation6d(pose)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected rot6d final dimension 6, got {rot6d.shape}")
    value = rot6d.reshape(rot6d.shape[:-1] + (3, 2))
    value = np.swapaxes(value, -2, -1).reshape(value.shape[:-2] + (6,))
    x_raw = value[..., 0:3]
    y_raw = value[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw, axis=-1)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x, axis=-1)
    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)


def sample_video(frames: list[np.ndarray], num_frames: int) -> np.ndarray:
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if not frames:
        raise ValueError("frames must not be empty")
    indices = (
        np.zeros(num_frames, dtype=np.int64)
        if len(frames) == 1
        else np.linspace(0, len(frames) - 1, num_frames).round().astype(np.int64)
    )
    return np.stack([resize_image(frames[index]) for index in indices], axis=0)


def sample_camera_frames(
    cameras: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    *,
    use_history: bool,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if any(not frames for frames in cameras):
        raise ValueError("all AgileX camera streams must contain at least one frame")
    if use_history:
        indices = np.linspace(0, len(cameras[0]) - 1, num_frames).astype(int)
    else:
        indices = np.asarray([-1])
    return tuple(
        np.stack([resize_image(frames[index]) for index in indices])
        for frames in cameras
    )  # type: ignore[return-value]


def manifold_eef6d_state(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    left = quaternion_pose_to_rot6d(
        _manifold_pose7(observation["left_end_pose"][-1], name="left_end_pose")
    )
    right = quaternion_pose_to_rot6d(
        _manifold_pose7(observation["right_end_pose"][-1], name="right_end_pose")
    )
    left_gripper = np.asarray(observation["left_arm_joint_state"][-1][-1])
    right_gripper = np.asarray(observation["right_arm_joint_state"][-1][-1])
    return {
        "state.left_pos": left[:3].reshape(1, 3),
        "state.left_rot6d": left[3:].reshape(1, 6),
        "state.left_gripper": left_gripper.reshape(1, 1),
        "state.right_pos": right[:3].reshape(1, 3),
        "state.right_rot6d": right[3:].reshape(1, 6),
        "state.right_gripper": right_gripper.reshape(1, 1),
    }


def _manifold_pose7(value: Any, *, name: str) -> np.ndarray:
    """Accept Manifold's xyz+quaternion pose with optional appended gripper."""

    pose = np.asarray(value)
    if pose.ndim != 1 or pose.shape[0] not in {7, 8}:
        raise ValueError(f"{name} must have shape (7,) or (8,), got {pose.shape}")
    return pose[:7]


def manifold_joint_state(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    left = np.asarray(observation["left_arm_joint_state"][-1], dtype=np.float32)
    right = np.asarray(observation["right_arm_joint_state"][-1], dtype=np.float32)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError(
            f"left/right joint states must be equal 1D vectors, got {left.shape}, "
            f"{right.shape}"
        )
    return {
        "state.left_joint": left.reshape(1, -1),
        "state.right_joint": right.reshape(1, -1),
    }


def action_fields_to_trajectories(
    action_fields: dict[str, np.ndarray],
    *,
    action_mode: str,
    gripper_open: float = 0.08,
    gripper_closed: float = 0.0,
    gripper_threshold: float = 0.03,
) -> tuple[list[list[float]], list[list[float]]]:
    if action_mode == "joint":
        left = _first_action(
            action_fields,
            ("action.left_joint", "action.left_arm_joint_position", "action.left_joint_pos"),
        )
        right = _first_action(
            action_fields,
            (
                "action.right_joint",
                "action.right_arm_joint_position",
                "action.right_joint_pos",
            ),
        )
        if left is None or right is None:
            raise KeyError("Native joint output is missing left/right joint action fields")
        left = np.atleast_2d(left).copy()
        right = np.atleast_2d(right).copy()
        if left.shape[0] != right.shape[0]:
            raise ValueError("left/right joint trajectory lengths differ")
        left[..., -1] = np.where(
            left[..., -1] < gripper_threshold, gripper_closed, gripper_open
        )
        right[..., -1] = np.where(
            right[..., -1] < gripper_threshold, gripper_closed, gripper_open
        )
        return left.tolist(), right.tolist()
    if action_mode != "eef":
        raise ValueError(f"Unsupported action mode: {action_mode!r}")

    batch = SimpleNamespace(act=action_fields)
    left_position = np.asarray(batch.act["action.left_pos"])
    right_position = np.asarray(batch.act["action.right_pos"])
    left_quaternion = Rotation.from_matrix(
        rot6d_to_matrix(np.asarray(batch.act["action.left_rot6d"]))
    ).as_quat()
    right_quaternion = Rotation.from_matrix(
        rot6d_to_matrix(np.asarray(batch.act["action.right_rot6d"]))
    ).as_quat()
    left_gripper = np.asarray(batch.act["action.left_gripper"]).reshape(-1, 1)
    right_gripper = np.asarray(batch.act["action.right_gripper"]).reshape(-1, 1)
    left = np.concatenate([left_position, left_quaternion, left_gripper], axis=-1)
    right = np.concatenate([right_position, right_quaternion, right_gripper], axis=-1)
    left[..., -1] = np.where(
        left_gripper.reshape(left.shape[:-1]) < gripper_threshold,
        gripper_closed,
        gripper_open,
    )
    right[..., -1] = np.where(
        right_gripper.reshape(right.shape[:-1]) < gripper_threshold,
        gripper_closed,
        gripper_open,
    )
    return left.tolist(), right.tolist()


def _first_action(
    action_fields: dict[str, np.ndarray], keys: tuple[str, ...]
) -> np.ndarray | None:
    for key in keys:
        if key in action_fields:
            return np.asarray(action_fields[key], dtype=np.float32)
    return None

VisualInputRange = Literal["uint8", "zero_one", "minus_one_one"]
_UNSET = object()

_STATE_KEY_ORDER = (
    "state.left_pos",
    "state.left_rot6d",
    "state.left_gripper",
    "state.right_pos",
    "state.right_rot6d",
    "state.right_gripper",
    "state.left_joint",
    "state.right_joint",
)

def _prompts(mode: InteractionMode, instruction: str) -> PromptBatch:
    if mode is InteractionMode.AUTO:
        return PromptBatch(
            vlm_planning_text=[instruction],
        )
    return PromptBatch(
        language_instruction=[instruction],
        negative_language_instruction=[""],
    )


def _ensure_video(value: np.ndarray) -> np.ndarray:
    video = np.asarray(value)
    if video.ndim == 3:
        video = video[None]
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected camera video [T,H,W,3], got {video.shape}")
    if video.dtype != np.uint8:
        raise TypeError("AgileX camera observations must be uint8")
    return np.ascontiguousarray(video)


@dataclass(frozen=True)
class AgileXObservation:
    observation: ObservationBatch
    prompts: PromptBatch
    timestamp: Any
    raw_state: dict[str, np.ndarray]


class AgileXObservationAdapter(ObservationAdapter[Any, AgileXObservation]):
    """Convert validated AgileX observations to the native public schema."""

    def __init__(
        self,
        bundle: CheckpointTransformBundle,
        *,
        mode: InteractionMode | str,
        instruction: str,
        visual_input_range: VisualInputRange,
        device: torch.device | str,
        goal_image: np.ndarray | None = None,
        demo_video: np.ndarray | tuple[np.ndarray, ...] | None = None,
        session_id: str = "agilex",
    ) -> None:
        if visual_input_range not in {"uint8", "zero_one", "minus_one_one"}:
            raise ValueError(f"Unknown visual input range: {visual_input_range!r}")
        self.bundle = bundle
        self.mode = InteractionMode.parse(mode)
        self.instruction = instruction
        self.visual_input_range = visual_input_range
        self.device = torch.device(device)
        self.session_id = str(session_id)
        self._goal_image = goal_image
        self._demo_video = demo_video
        self._visual_prompt_pending = True
        if goal_image is not None and demo_video is not None:
            raise ValueError("Provide either a goal image or demo video, not both")
        self._validate_prompt_shape()

    def reset_episode(
        self,
        *,
        goal_image: np.ndarray | None | object = _UNSET,
        demo_video: np.ndarray | tuple[np.ndarray, ...] | None | object = _UNSET,
        session_id: str | None = None,
    ) -> None:
        """Reset one robot session and arm its persistent visual prompt."""

        if (
            goal_image is not _UNSET
            and demo_video is not _UNSET
            and goal_image is not None
            and demo_video is not None
        ):
            raise ValueError("Provide either a goal image or demo video, not both")
        if goal_image is not _UNSET:
            self._goal_image = goal_image  # type: ignore[assignment]
            if goal_image is not None:
                self._demo_video = None
        if demo_video is not _UNSET:
            self._demo_video = demo_video  # type: ignore[assignment]
            if demo_video is not None:
                self._goal_image = None
        if self._goal_image is not None and self._demo_video is not None:
            raise ValueError("Provide either a goal image or demo video, not both")
        if session_id is not None:
            self.session_id = str(session_id)
        self._validate_prompt_shape()
        self._visual_prompt_pending = True

    def _validate_prompt_shape(self) -> None:
        if self._goal_image is not None:
            goal = np.asarray(self._goal_image)
            views = 1 if goal.ndim == 3 else goal.shape[0] if goal.ndim == 4 else -1
            if views != 1:
                raise ValueError(f"Goal prompt must contain exactly one head frame (V=1), got {goal.shape}")
        if self._demo_video is not None:
            if isinstance(self._demo_video, tuple):
                views = len(self._demo_video)
                lengths = {_ensure_video(camera).shape[0] for camera in self._demo_video}
            else:
                demo = np.asarray(self._demo_video)
                views = 1 if demo.ndim == 4 else demo.shape[1] if demo.ndim == 5 else -1
                lengths = {demo.shape[0]} if demo.ndim in {4, 5} else set()
            if views not in {1, 3}:
                raise ValueError(f"Demo prompt must have V=1 or V=3, got V={views}")
            if lengths != {50}:
                raise ValueError(f"Demo prompt must contain exactly 50 frames, got {sorted(lengths)}")

    def __call__(
        self,
        raw: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any],
    ) -> AgileXObservation:
        high, left, right, raw_state, timestamp = raw
        videos = [_ensure_video(value) for value in (high, left, right)]
        if len({video.shape[0] for video in videos}) != 1:
            raise ValueError("AgileX camera histories must have equal length")
        state = self._transform_state(raw_state)
        images = torch.from_numpy(np.stack(videos, axis=1)).permute(0, 1, 4, 2, 3)
        head = torch.from_numpy(videos[0][-1:]).permute(0, 3, 1, 2)
        images = self._convert_images(images).unsqueeze(0).to(self.device)
        head = self._convert_images(head).unsqueeze(0).to(self.device)
        observation = ObservationBatch(
            images=images,
            head_view=head,
            proprioception=state.to(self.device),
            embodiment_id=torch.tensor(
                [self.bundle.embodiment_id],
                dtype=torch.long,
                device=self.device,
            ),
        )
        observation.validate()
        prompts = _prompts(self.mode, self.instruction)
        if self._visual_prompt_pending:
            prompts.goal_images = self._convert_goal(self._goal_image)
            prompts.demo_videos = self._convert_demo(self._demo_video)
            self._visual_prompt_pending = False
        prompts.validate(1)
        return AgileXObservation(
            observation=observation,
            prompts=prompts,
            timestamp=timestamp,
            raw_state={
                key: np.asarray(value).copy() for key, value in raw_state.items()
            },
        )

    def observation(
        self,
        value: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Any],
    ) -> AgileXObservation:
        """Adapt through the backend-neutral observation protocol."""

        return self(value)

    def _transform_state(self, raw_state: dict[str, Any]) -> torch.Tensor:
        data: dict[str, Any] = {
            key: np.asarray(value).copy() for key, value in raw_state.items()
        }
        native_apply = getattr(self.bundle.transform, "apply_state", None)
        if callable(native_apply):
            state = native_apply(data)
        else:
            for transform in getattr(self.bundle.transform, "transforms", ()):
                if type(transform).__name__ == "DreamTransform":
                    break
                data = transform(data)
                if "state" in data:
                    break
            state = data.get("state")
        if state is None:
            values = [
                np.asarray(data[key], dtype=np.float32).reshape(-1)
                for key in _STATE_KEY_ORDER
                if key in data
            ]
            if not values:
                raise ValueError("AgileX observation has no supported state fields")
            state = torch.from_numpy(np.concatenate(values))
        else:
            state = torch.as_tensor(state)
        state = state.float().reshape(1, 1, -1)
        if state.shape[-1] > self.bundle.max_state_dim:
            raise ValueError(
                f"State dimension {state.shape[-1]} exceeds checkpoint maximum "
                f"{self.bundle.max_state_dim}"
            )
        return torch.nn.functional.pad(
            state, (0, self.bundle.max_state_dim - state.shape[-1])
        )

    def _convert_images(self, value: torch.Tensor) -> torch.Tensor:
        native_apply = getattr(self.bundle.transform, "apply_image", None)
        if callable(native_apply):
            if self.visual_input_range != self.bundle.transform.image_input_range:
                raise ValueError(
                    "Evaluation visual_input_range does not match transform bundle: "
                    f"{self.visual_input_range!r} != "
                    f"{self.bundle.transform.image_input_range!r}"
                )
            return native_apply(value)
        if self.visual_input_range == "uint8":
            return value.to(torch.uint8)
        result = value.float().div(255.0)
        if self.visual_input_range == "minus_one_one":
            result = result.mul(2.0).sub(1.0)
        return result

    def _convert_goal(self, value: np.ndarray | None) -> torch.Tensor | None:
        if value is None:
            return None
        images = np.asarray(value)
        if images.ndim == 3:
            images = images[None]
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(
                f"Goal image must have shape [H,W,3] or [V,H,W,3], got {images.shape}"
            )
        tensor = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
        return self._convert_images(tensor).unsqueeze(0).to(self.device)

    def _convert_demo(
        self,
        value: np.ndarray | tuple[np.ndarray, ...] | None,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        if isinstance(value, tuple):
            cameras = [_ensure_video(camera) for camera in value]
            if len({camera.shape[0] for camera in cameras}) != 1:
                raise ValueError("Demo camera videos must have equal length")
            video = np.stack(cameras, axis=1)
        else:
            video = np.asarray(value)
            if video.ndim == 4:
                video = video[:, None]
        if video.ndim != 5 or video.shape[-1] != 3:
            raise ValueError(
                "Demo video must have shape [T,H,W,3] or [T,V,H,W,3], "
                f"got {video.shape}"
            )
        if video.dtype != np.uint8:
            raise TypeError("AgileX demo video must be uint8")
        tensor = torch.from_numpy(np.ascontiguousarray(video)).permute(0, 1, 4, 2, 3)
        return self._convert_images(tensor).unsqueeze(0).to(self.device)



__all__ = [
    "AgileXObservation",
    "AgileXObservationAdapter",
    "VisualInputRange",
    "action_fields_to_trajectories",
    "quaternion_pose_to_rot6d",
    "resize_image",
    "sample_camera_frames",
    "sample_video",
]
