"""Trial-scoped environment backends for RoboDojo evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from XPolicyLab.utils.process_data import get_robot_action_dim_info


def _camera_block() -> dict[str, Any]:
    return {
        "color": np.zeros((480, 640, 3), dtype=np.uint8),
        "depth": np.zeros((480, 640, 3), dtype=np.uint8),
        "intrinsic_matrix": [
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        "extrinsics_matrix": [
            [1.0, 0.0, 0.0, 0.10],
            [0.0, 1.0, 0.0, 1.20],
            [0.0, 0.0, 1.0, 1.50],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "shape": (480, 640),
    }


def build_mock_observation(
    env_cfg_type: str,
    *,
    instruction: str = "",
    env_idx: int = 0,
    additional_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robot_action_dim_info = get_robot_action_dim_info(env_cfg_type)
    obs: dict[str, Any] = {
        "vision": {
            "cam_head": _camera_block(),
            "cam_left_wrist": _camera_block(),
            "cam_right_wrist": _camera_block(),
        },
        "instruction": instruction,
        "state": {
            "mobile": {
                "base_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "base_twist": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "additional_info": dict(additional_info or {}),
        "data_format_version": "v1.0",
        "env_idx": env_idx,
    }

    state = obs["state"]
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]

    if len(arm_dims) == 1:
        prefixes = [""]
    elif len(arm_dims) == 2:
        prefixes = ["left_", "right_"]
    else:
        raise ValueError(f"Unsupported arm count: {len(arm_dims)}")

    for index, prefix in enumerate(prefixes):
        state[f"{prefix}arm_joint_state"] = np.zeros(arm_dims[index], dtype=np.float32)
        state[f"{prefix}ee_joint_state"] = np.zeros(ee_dims[index], dtype=np.float32)
        state[f"{prefix}ee_pose"] = np.ones(7, dtype=np.float32)
        state[f"{prefix}tcp_pose"] = np.zeros(7, dtype=np.float32)
        state[f"{prefix}delta_ee_pose"] = np.zeros(7, dtype=np.float32)

    return obs


class TrialEnv(Protocol):
    def reset(self) -> None: ...

    def get_obs(self, env_idx: int = 0) -> dict[str, Any]: ...

    def get_obs_batch(self, env_idx_list: list[int]) -> list[dict[str, Any]]: ...

    def take_action(self, action: dict[str, Any]) -> None: ...

    def take_action_batch(
        self, action_list: list[dict[str, Any]], env_idx_list: list[int]
    ) -> None: ...

    def is_episode_end(self) -> bool: ...

    def get_running_env_idx_list(self) -> list[int]: ...


def validate_robot_state_dict(state_dict: dict, robot_action_dim_info: dict) -> None:
    arm_dims = robot_action_dim_info["arm_dim"]
    ee_dims = robot_action_dim_info["ee_dim"]

    if len(arm_dims) != len(ee_dims):
        raise ValueError(
            f"robot_action_dim_info mismatch: len(arm_dim)={len(arm_dims)} "
            f"!= len(ee_dim)={len(ee_dims)}"
        )

    arm_count = len(arm_dims)
    if arm_count == 1:
        expected = {
            "arm_joint_state": arm_dims[0],
            "ee_joint_state": ee_dims[0],
            "ee_pose": 7,
            "tcp_pose": 7,
            "delta_ee_pose": 7,
        }
        forbidden_prefixes = ("left_", "right_")
    elif arm_count == 2:
        expected = {
            "left_arm_joint_state": arm_dims[0],
            "left_ee_joint_state": ee_dims[0],
            "left_ee_pose": 7,
            "left_tcp_pose": 7,
            "left_delta_ee_pose": 7,
            "right_arm_joint_state": arm_dims[1],
            "right_ee_joint_state": ee_dims[1],
            "right_ee_pose": 7,
            "right_tcp_pose": 7,
            "right_delta_ee_pose": 7,
        }
        forbidden_prefixes = ()
    else:
        raise ValueError(f"Unsupported arm count: {arm_count}")

    if forbidden_prefixes:
        bad_prefixed_keys = [
            key for key in state_dict if key.startswith(forbidden_prefixes)
        ]
        if bad_prefixed_keys:
            raise ValueError(
                "Single-arm robot should not contain prefixed keys, "
                f"but got: {bad_prefixed_keys}"
            )

    unexpected_keys = [key for key in state_dict if key not in expected]
    if unexpected_keys:
        raise ValueError(f"Unexpected state keys: {unexpected_keys}")

    for key, expected_dim in expected.items():
        if key not in state_dict:
            continue
        value = state_dict[key]
        if not isinstance(value, (np.ndarray, list, tuple)):
            raise TypeError(
                f"state_dict['{key}'] must be array-like, got {type(value)}"
            )
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise ValueError(
                f"state_dict['{key}'] must be 1D, got shape {arr.shape}"
            )
        if arr.shape[0] != expected_dim:
            raise ValueError(
                f"state_dict['{key}'] dim mismatch: expected {expected_dim}, "
                f"got shape {arr.shape}"
            )


@dataclass
class DebugTrialEnv:
    env_cfg_type: str
    instruction: str = ""
    episode_step_limit: int = 5
    batch_size: int = 10
    additional_info: dict[str, Any] | None = None
    robot_action_dim_info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.robot_action_dim_info is None:
            self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.episode_step = 0

    def reset(self) -> None:
        self.episode_step = 0

    def get_obs(self, env_idx: int = 0) -> dict[str, Any]:
        return build_mock_observation(
            self.env_cfg_type,
            instruction=self.instruction,
            env_idx=env_idx,
            additional_info=self.additional_info,
        )

    def get_obs_batch(self, env_idx_list: list[int]) -> list[dict[str, Any]]:
        return [self.get_obs(env_idx) for env_idx in env_idx_list]

    def take_action(self, action: dict[str, Any]) -> None:
        self.episode_step += 1
        validate_robot_state_dict(action, self.robot_action_dim_info)

    def take_action_batch(
        self, action_list: list[dict[str, Any]], env_idx_list: list[int]
    ) -> None:
        self.episode_step += 1
        if len(action_list) != len(env_idx_list):
            raise ValueError(
                f"action num != env num: {len(action_list)} != {len(env_idx_list)}"
            )
        for action in action_list:
            validate_robot_state_dict(action, self.robot_action_dim_info)

    def is_episode_end(self) -> bool:
        return self.episode_step >= self.episode_step_limit

    def get_running_env_idx_list(self) -> list[int]:
        return list(range(self.batch_size))
