"""XPolicyLab-facing stateful policy adapter shared by the local model bridge."""

from __future__ import annotations

from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def optional_int(value: Any) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def instruction(obs: dict[str, Any], fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item"):
        value = value.item()
    return str(value).strip() if value is not None and str(value).strip() else fallback


def rgb(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be an HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def env_index(obs: dict[str, Any]) -> int:
    if "env_idx" not in obs:
        raise ValueError("Batched observations must carry env_idx")
    return int(obs["env_idx"])


def env_indices(value: Any) -> list[int]:
    indices = [int(item) for item in np.asarray(value).reshape(-1)]
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate env_idx values: {indices}")
    return indices


class StatefulWAMAdapter(ModelTemplate):
    """Stateful WAM adapter; subclasses provide model loading.

    Batched evaluation keeps one session per env_idx, so memory frames,
    pending actions and step counters never cross environments.
    """

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg.get("action_type") or "joint")
        self.env_cfg_type = str(self.model_cfg.get("env_cfg_type") or "arx_x5")
        if self.action_type != "joint":
            raise ValueError("This checkpoint was trained for joint actions")

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dim = list(self.robot_action_dim_info.get("arm_dim", []))
        ee_dim = list(self.robot_action_dim_info.get("ee_dim", []))
        if arm_dim != [6, 6]:
            raise ValueError(f"Checkpoint requires dual ARX-X5 arms: {self.robot_action_dim_info}")
        if ee_dim != [1, 1]:
            raise ValueError(f"Checkpoint requires one gripper scalar per arm: {self.robot_action_dim_info}")
        self.gripper_slices = []
        offset = 0
        for arm, ee in zip(arm_dim, ee_dim):
            offset += arm
            self.gripper_slices.append(slice(offset, offset + ee))
            offset += ee
        self.action_dim = offset

        self.default_instruction = str(
            self.model_cfg.get("default_instruction") or "follow the instruction"
        )
        self.last_obs: dict[str, Any] | None = None
        self.last_instruction = self.default_instruction
        self.allow_dummy_policy = as_bool(self.model_cfg.get("allow_dummy_policy", False))
        self.session = None
        self.session_factory = None
        self.env_sessions: dict[int, Any] = {}
        self.env_obs: dict[int, tuple[dict[str, Any], str]] = {}
        self.env_order: list[int] = []
        self.runtime = None
        self.action_horizon = int(self.model_cfg.get("action_horizon") or 32)
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 10)
        if not 1 <= self.replan_steps <= self.action_horizon:
            raise ValueError(
                f"replan_steps must be in [1, {self.action_horizon}], got {self.replan_steps}"
            )
        if not self.allow_dummy_policy:
            self._load_real_policy()

    def _load_real_policy(self) -> None:
        raise NotImplementedError

    def _adapt_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        vision = obs["vision"]
        state = pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"Expected a finite {self.action_dim}D RoboDojo state, got {state.shape}"
            )
        return {
            "observation": {
                "head_camera": {"rgb": rgb(vision["cam_head"]["color"], "cam_head")},
                "left_camera": {
                    "rgb": rgb(vision["cam_left_wrist"]["color"], "cam_left_wrist")
                },
                "right_camera": {
                    "rgb": rgb(vision["cam_right_wrist"]["color"], "cam_right_wrist")
                },
            },
            "joint_action": {"vector": state},
        }

    def _ingest(self, session, obs: dict[str, Any]) -> tuple[dict[str, Any], str]:
        adapted = self._adapt_obs(obs)
        if session is not None and session.pending_model_actions:
            session.update_obs(adapted)
        return adapted, instruction(obs, self.default_instruction)

    def update_obs(self, obs):
        self.last_obs, self.last_instruction = self._ingest(self.session, obs)

    def _session_for(self, env_idx: int):
        if self.allow_dummy_policy:
            return None
        if env_idx not in self.env_sessions:
            self.env_sessions[env_idx] = self.session_factory()
        return self.env_sessions[env_idx]

    def update_obs_batch(self, obs_list):
        if isinstance(obs_list, dict):
            obs_list = [obs_list]
        order = env_indices([env_index(obs) for obs in obs_list])
        if not order:
            raise ValueError("update_obs_batch requires at least one observation")
        for env_idx, obs in zip(order, obs_list):
            self.env_obs[env_idx] = self._ingest(self._session_for(env_idx), obs)
        self.env_order = order

    def _dummy_actions(self) -> list[dict[str, np.ndarray]]:
        zeros = np.zeros((self.replan_steps, self.action_dim), dtype=np.float32)
        return unpack_robot_state(
            zeros, self.action_type, self.robot_action_dim_info, source_type="obs"
        )

    def _predict(self, session, observation: dict[str, Any], instruction_text: str):
        if self.allow_dummy_policy:
            return self._dummy_actions()
        if session.pending_model_actions:
            raise RuntimeError("Previous actions have not all been acknowledged")
        packed = np.asarray(
            session.get_action(
                {"observation": observation, "instruction": instruction_text}
            ),
            dtype=np.float32,
        )
        if (
            packed.ndim != 2
            or packed.shape[1] != self.action_dim
            or not np.isfinite(packed).all()
        ):
            raise ValueError(f"WAM returned an invalid action chunk: {packed.shape}")
        for gripper in self.gripper_slices:
            packed[:, gripper] = np.clip(packed[:, gripper], 0.0, 1.0)
        return unpack_robot_state(
            packed, self.action_type, self.robot_action_dim_info, source_type="obs"
        )

    def get_action(self):
        if self.last_obs is None:
            raise ValueError("Call update_obs before get_action")
        return self._predict(self.session, self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list=None):
        order = self.env_order if env_idx_list is None else env_indices(env_idx_list)
        missing = [env_idx for env_idx in order if env_idx not in self.env_obs]
        if not order or missing:
            raise ValueError(
                f"Call update_obs_batch before get_action_batch (missing env_idx: {missing})"
            )
        return [
            self._predict(self._session_for(env_idx), *self.env_obs[env_idx])
            for env_idx in order
        ]

    def get_timing_rollout(self):
        if self.env_sessions:
            return {
                str(env_idx): session.get_timing_rollout()
                for env_idx, session in sorted(self.env_sessions.items())
            }
        return {} if self.session is None else self.session.get_timing_rollout()

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.env_obs.clear()
        self.env_order = []
        for session in (self.session, *self.env_sessions.values()):
            if session is not None:
                session.reset_model()
