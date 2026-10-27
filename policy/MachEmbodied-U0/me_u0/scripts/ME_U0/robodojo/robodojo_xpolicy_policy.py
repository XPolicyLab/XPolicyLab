"""XPolicyLab observation/action adapter for an ME_U0 RoboDojo policy."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


CAMERA_ORDER = ("cam_head", "cam_left_wrist", "cam_right_wrist")
JOINT_STATE_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
def _to_uint8_hwc(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"camera image must be rank-3, got {array.shape}")
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"camera image must have three color channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        finite_max = float(np.nanmax(array)) if array.size else 0.0
        if finite_max <= 1.0 + 1e-6:
            array = array * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def _extract_image(obs: dict[str, Any], camera_name: str) -> np.ndarray:
    vision = obs.get("vision", {})
    if camera_name not in vision:
        raise KeyError(f"observation.vision is missing {camera_name!r}; available={list(vision)}")
    camera = vision[camera_name]
    if isinstance(camera, dict):
        for key in ("color", "rgb", "image", "colors"):
            if key in camera:
                return _to_uint8_hwc(camera[key])
        raise KeyError(f"observation.vision.{camera_name} has no RGB field; available={list(camera)}")
    return _to_uint8_hwc(camera)


def _extract_state(obs: dict[str, Any], action_type: str) -> np.ndarray:
    state = obs.get("state", {})
    values = []
    if action_type != "joint":
        raise ValueError(f"unsupported RoboDojo action_type={action_type!r}")
    keys, expected = JOINT_STATE_KEYS, (6, 1, 6, 1)
    for key, width in zip(keys, expected):
        if key not in state:
            raise KeyError(f"observation.state is missing {key!r}; available={list(state)}")
        value = np.asarray(state[key], dtype=np.float32).reshape(-1)
        if value.shape != (width,):
            raise ValueError(f"observation.state.{key} must have shape ({width},), got {value.shape}")
        values.append(value)
    packed = np.concatenate(values).astype(np.float32)
    if not np.isfinite(packed).all():
        raise ValueError("RoboDojo state contains NaN/Inf")
    return packed


def _extract_instruction(obs: dict[str, Any]) -> str:
    value = obs.get("instruction")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"instruction list must contain exactly one string, got {value!r}")
        value = value[0]
    if value is None or not str(value).strip():
        raise ValueError("RoboDojo observation contains an empty instruction")
    return str(value)


def unpack_native_action(action: Any, action_type: str = "joint") -> dict[str, np.ndarray]:
    """Convert one native ARX-X5 action to official RoboDojo keys."""
    if action_type != "joint":
        raise ValueError(f"unsupported RoboDojo action_type={action_type!r}")
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (14,):
        raise ValueError(f"native joint action must have 14 values, got {value.shape}")
    return {
        "left_arm_joint_state": value[0:6].copy(),
        "left_ee_joint_state": np.clip(value[6:7], 0.0, 1.0).copy(),
        "right_arm_joint_state": value[7:13].copy(),
        "right_ee_joint_state": np.clip(value[13:14], 0.0, 1.0).copy(),
    }


class RoboDojoXPolicyModel:
    """Legacy Model interface consumed by XPolicyLab's WebSocket server."""

    def __init__(
        self,
        policy: Any,
        *,
        dataset_name: str = "robodojo_sim_arx_x5",
        action_chunk_size: int = 8,
        action_type: str = "joint",
    ) -> None:
        trained_horizon = int(policy.model.config.action_horizon)
        if not 1 <= int(action_chunk_size) <= trained_horizon:
            raise ValueError(
                "action_chunk_size must be within the trained horizon "
                f"[1, {trained_horizon}]"
            )
        self.policy = policy
        self.dataset_name = str(dataset_name)
        self.action_chunk_size = int(action_chunk_size)
        self.action_type = str(action_type)
        if self.action_type != "joint":
            raise ValueError("ME_U0 RoboDojo policy only supports joint actions")
        self._observations: dict[int, dict[str, Any]] = {}
        self._latest_env_indices = [0]

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs: list[dict[str, Any]]) -> None:
        if isinstance(obs, dict):
            obs = [obs]
        self._latest_env_indices = []
        for fallback_index, item in enumerate(obs):
            env_index = int(item.get("env_idx", fallback_index))
            self._latest_env_indices.append(env_index)
            self._observations[env_index] = {
                "images": [_extract_image(item, name) for name in CAMERA_ORDER],
                "instruction": _extract_instruction(item),
                "state": _extract_state(item, self.action_type),
            }

    def _predict(self, env_index: int) -> list[dict[str, np.ndarray]]:
        if env_index not in self._observations:
            raise RuntimeError(f"No observation for env_idx={env_index}; call update_obs first")
        item = self._observations[env_index]
        native_actions = self.policy.predict_action(
            images=item["images"],
            instruction=item["instruction"],
            state=item["state"],
            dataset_name=self.dataset_name,
            action_horizon=self.action_chunk_size,
        )
        native_actions = np.asarray(native_actions, dtype=np.float32)
        native_width = 14
        if native_actions.ndim != 2 or native_actions.shape[1] != native_width:
            raise ValueError(
                f"ME_U0 {self.action_type} policy must return (T,{native_width}), "
                f"got {native_actions.shape}"
            )
        horizon = min(self.action_chunk_size, native_actions.shape[0])
        return [
            unpack_native_action(native_actions[index], self.action_type)
            for index in range(horizon)
        ]

    def get_action(self) -> list[dict[str, np.ndarray]]:
        return self._predict(self._latest_env_indices[0])

    def get_action_batch(self, obs: Optional[list[int]] = None, **_: Any) -> list[list[dict[str, np.ndarray]]]:
        indices = self._latest_env_indices if obs is None else [int(index) for index in obs]
        return [self._predict(index) for index in indices]

    def reset(self) -> None:
        self._observations.clear()
        self._latest_env_indices = [0]
