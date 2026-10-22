# SPDX-License-Identifier: Apache-2.0

"""RoboDojo observation/action conversion at the frontend boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise ValueError(f"Unsupported dtype: {value.dtype}")
        return {
            b"nd": True,
            b"type": value.dtype.str,
            b"kind": value.dtype.kind,
            b"shape": value.shape,
            b"data": value.tobytes(),
        }
    if isinstance(value, np.generic):
        if value.dtype.kind == "O":
            raise ValueError(f"Unsupported dtype: {value.dtype}")
        return {
            b"nd": False,
            b"type": value.dtype.str,
            b"kind": value.dtype.kind,
            b"data": value.tobytes(),
        }
    return value


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if value.get(b"nd") is True:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"type"]),
            shape=tuple(value[b"shape"]),
        )
    if value.get(b"nd") is False:
        dtype = np.dtype(value[b"type"])
        data = value[b"data"]
        if isinstance(data, (bytes, bytearray, memoryview)):
            return np.frombuffer(data, dtype=dtype, count=1)[0]
        return dtype.type(data)
    # Keep decoding frames produced by OpenPI-style msgpack helpers.
    if value.get(b"__ndarray__") is True:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=tuple(value[b"shape"]),
        )
    if value.get(b"__npgeneric__") is True:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def pack_frame(frame: Mapping[str, Any]) -> bytes:
    return msgpack.packb(
        dict(frame), default=_pack_numpy, use_bin_type=True
    )


def unpack_frame(raw: bytes | bytearray) -> dict[str, Any]:
    value = msgpack.unpackb(
        bytes(raw), raw=False, object_hook=_unpack_numpy
    )
    if not isinstance(value, dict):
        raise ValueError("RoboDojo frame must be a map")
    return value


@dataclass(frozen=True)
class RoboDojoFrame:
    message_type: str
    message_id: str
    evaluation_id: str
    action_case_id: str | None = None
    trial_id: str | None = None
    repeat_index: int | None = None
    step: int = 0
    payload: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type,
            "message_id": self.message_id,
            "evaluation_id": self.evaluation_id,
            "action_case_id": self.action_case_id,
            "trial_id": self.trial_id,
            "repeat_index": self.repeat_index,
            "step": self.step,
            "payload": self.payload or {},
        }

    @classmethod
    def from_wire(cls, value: Mapping[str, Any]) -> "RoboDojoFrame":
        required = ("message_type", "message_id", "evaluation_id")
        missing = [key for key in required if value.get(key) is None]
        if missing:
            raise ValueError(f"RoboDojo frame missing {', '.join(missing)}")
        payload = value.get("payload") or {}
        if not isinstance(payload, Mapping):
            raise ValueError("RoboDojo frame payload must be a map")
        try:
            step = int(value.get("step", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("RoboDojo frame step must be an integer") from exc
        return cls(
            message_type=str(value["message_type"]),
            message_id=str(value["message_id"]),
            evaluation_id=str(value["evaluation_id"]),
            action_case_id=_optional_string(value.get("action_case_id")),
            trial_id=_optional_string(value.get("trial_id")),
            repeat_index=_optional_int(value.get("repeat_index")),
            step=step,
            payload=dict(payload),
        )


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _camera_image(vision: Mapping[str, Any], names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        value = vision.get(name)
        if value is None:
            continue
        if isinstance(value, Mapping):
            value = value.get("color", value.get("rgb"))
        if value is None:
            continue
        image = np.asarray(value)
        if image.ndim != 3:
            raise ValueError(f"RoboDojo camera {name!r} must be a 3D image")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0) * 255.0
        return image.astype(np.uint8, copy=False)
    return None


def _camera_depth(
    vision: Mapping[str, Any], names: tuple[str, ...]
) -> np.ndarray | None:
    """Return one camera's metric distance-to-image-plane depth map."""
    for name in names:
        value = vision.get(name)
        if value is None:
            continue
        if isinstance(value, Mapping):
            value = value.get("depth")
        if value is None:
            continue
        depth = np.asarray(value)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        # The official debug client supplies replicated three-channel depth.
        # Accept only equal channels, never interpret a color visualization as
        # metric depth.
        if (depth.ndim == 3 and depth.shape[-1] == 3
                and np.array_equal(depth[..., 0], depth[..., 1])
                and np.array_equal(depth[..., 0], depth[..., 2])):
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"RoboDojo camera {name!r} depth must be a 2D map")
        return depth.astype(np.float32, copy=False)
    return None


def _camera_field(
    vision: Mapping[str, Any],
    names: tuple[str, ...],
    field_names: tuple[str, ...],
) -> Any:
    for name in names:
        camera = vision.get(name)
        if not isinstance(camera, Mapping):
            continue
        for field_name in field_names:
            value = camera.get(field_name)
            if value is not None:
                return value
    return None


@dataclass(frozen=True)
class ActionLayout:
    keys: tuple[str, ...]
    dims: tuple[int, ...]

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any]) -> "ActionLayout | None":
        state = observation.get("state")
        if not isinstance(state, Mapping):
            return None
        if "arm_joint_state" in state and "ee_joint_state" in state:
            keys = ("arm_joint_state", "ee_joint_state")
        elif all(
            key in state
            for key in (
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            )
        ):
            keys = (
                "left_arm_joint_state",
                "left_ee_joint_state",
                "right_arm_joint_state",
                "right_ee_joint_state",
            )
        else:
            return None
        dims = tuple(np.asarray(state[key]).reshape(-1).shape[0] for key in keys)
        if any(dim <= 0 for dim in dims):
            raise ValueError("RoboDojo action state dimensions must be positive")
        return cls(keys, dims)

    def unpack(self, actions: Any) -> list[dict[str, np.ndarray]]:
        values = np.asarray(actions)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2:
            raise ValueError(
                f"VLA actions must have shape [horizon, action_dim], got {values.shape}"
            )
        expected = sum(self.dims)
        if values.shape[1] != expected:
            raise ValueError(
                f"VLA action dimension mismatch: expected {expected}, got {values.shape[1]}"
            )
        result = []
        for row in values:
            offset = 0
            action = {}
            for key, dim in zip(self.keys, self.dims, strict=True):
                action[key] = row[offset : offset + dim]
                offset += dim
            result.append(action)
        return result


class RoboDojoObservationAdapter:
    """Translate one RoboDojo observation to EchoPolicy's canonical schema."""

    def to_canonical(
        self,
        observation: Mapping[str, Any],
        *,
        step: int = 0,
        episode_marker: str | None = None,
    ) -> tuple[dict[str, Any], ActionLayout | None]:
        result = dict(observation)
        instruction = observation.get("instruction", observation.get("prompt", ""))
        result["prompt"] = str(instruction)

        vision = observation.get("vision")
        if isinstance(vision, Mapping):
            head_names = ("cam_head", "cam_high", "head_camera", "top_camera")
            left_wrist_names = (
                "cam_left_wrist", "left_camera", "left_wrist", "wrist_left"
            )
            right_wrist_names = (
                "cam_right_wrist", "right_camera", "right_wrist", "wrist_right"
            )
            image = _camera_image(
                vision, head_names
            )
            if image is not None:
                result["observation/exterior_image_1_left"] = image
            image = _camera_image(
                vision, left_wrist_names
            )
            if image is not None:
                result["observation/wrist_image_left"] = image
            image = _camera_image(
                vision, right_wrist_names
            )
            if image is not None:
                result["observation/wrist_image_right"] = image
            depth = _camera_depth(
                vision, head_names
            )
            if depth is not None:
                result["observation/depth_exterior_image_1_left"] = depth
                result["observation/depth_external"] = depth
            depth = _camera_depth(
                vision, left_wrist_names
            )
            if depth is not None:
                result["observation/depth_wrist_image_left"] = depth
            depth = _camera_depth(
                vision, right_wrist_names
            )
            if depth is not None:
                result["observation/depth_wrist_image_right"] = depth
            intrinsics = _camera_field(
                vision, head_names, ("intrinsic_matrix", "intrinsics_matrix")
            )
            if intrinsics is not None:
                result["observation/camera_K"] = np.asarray(
                    intrinsics, dtype=np.float64
                )
            extrinsics = _camera_field(
                vision, head_names, ("extrinsics_matrix", "extrinsic_matrix")
            )
            if extrinsics is not None:
                result["observation/camera_extrinsic"] = np.asarray(
                    extrinsics, dtype=np.float64
                )
            for arm, names in (
                ("left", left_wrist_names),
                ("right", right_wrist_names),
            ):
                intrinsics = _camera_field(
                    vision, names, ("intrinsic_matrix", "intrinsics_matrix")
                )
                if intrinsics is not None:
                    result[f"observation/wrist_camera_K_{arm}"] = np.asarray(
                        intrinsics, dtype=np.float64
                    )
                extrinsics = _camera_field(
                    vision, names, ("extrinsics_matrix", "extrinsic_matrix")
                )
                if extrinsics is not None:
                    result[f"observation/wrist_camera_extrinsic_{arm}"] = np.asarray(
                        extrinsics, dtype=np.float64
                    )

        layout = ActionLayout.from_observation(observation)
        if layout is not None:
            state = observation["state"]
            arm_keys = layout.keys[::2]
            ee_keys = layout.keys[1::2]
            result["observation/joint_position"] = np.concatenate(
                [np.asarray(state[key]).reshape(-1) for key in arm_keys]
            )
            result["observation/gripper_position"] = np.concatenate(
                [np.asarray(state[key]).reshape(-1) for key in ee_keys]
            )

        if episode_marker is not None:
            result["__episode_id"] = episode_marker
        result["__step"] = int(step)
        return result, layout

    @staticmethod
    def to_robodojo_action(response: Any, layout: ActionLayout | None) -> Any:
        if isinstance(response, Mapping):
            actions = response.get(
                "__robodojo_actions", response.get("actions", response)
            )
        else:
            actions = response
        if isinstance(actions, list) and all(isinstance(item, Mapping) for item in actions):
            return actions
        if layout is None:
            raise ValueError("Cannot decode ndarray actions without RoboDojo state layout")
        return layout.unpack(actions)
