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
from XPolicyLab.utils import wuji58

# The two embodiment contracts a checkpoint can be trained for.
ARX_X5_JOINT = "arx_x5 joint: action_type=joint, arm_dim=[6, 6], ee_dim=[1, 1] (14-D)"
WUJI58_EE = (
    "tianji_wuji ee: action_type=ee, arm_dim=[7, 7], ee_dim=[20, 20] "
    "(58-D hybrid [L hand 20 | L wrist 9 | R hand 20 | R wrist 9])"
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

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        arm_dim = list(self.robot_action_dim_info.get("arm_dim", []))
        ee_dim = list(self.robot_action_dim_info.get("ee_dim", []))
        self.wuji58 = wuji58.is_wuji58(self.robot_action_dim_info, self.action_type)
        self.wrist_frame = None
        self.gripper_slices = []
        if self.wuji58:
            # 58-D hybrid: absolute hand joints + relative camera-frame wrist deltas.
            self.wrist_frame = wuji58.WristFrame.from_name(self.model_cfg.get("wrist_frame"))
            self.action_dim = wuji58.WUJI58_DIM
        elif self.action_type == "joint" and arm_dim == [6, 6] and ee_dim == [1, 1]:
            offset = 0
            for arm, ee in zip(arm_dim, ee_dim):
                offset += arm
                self.gripper_slices.append(slice(offset, offset + ee))
                offset += ee
            self.action_dim = offset
        else:
            raise ValueError(
                f"Unsupported embodiment for env_cfg_type={self.env_cfg_type!r}, "
                f"action_type={self.action_type!r}, dims={self.robot_action_dim_info}. "
                f"Supported contracts: [{ARX_X5_JOINT}] or [{WUJI58_EE}]"
            )

        self.default_instruction = str(
            self.model_cfg.get("default_instruction") or "follow the instruction"
        )
        self.last_obs: dict[str, Any] | None = None
        self.last_instruction = self.default_instruction
        self.last_raw_state: dict[str, np.ndarray] | None = None
        self.allow_dummy_policy = as_bool(self.model_cfg.get("allow_dummy_policy", False))
        self.session = None
        self.session_factory = None
        self.env_sessions: dict[int, Any] = {}
        self.env_obs: dict[int, tuple[dict[str, Any], str, dict[str, np.ndarray] | None]] = {}
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

    def _pack_state(self, obs: dict[str, Any]) -> np.ndarray:
        if self.wuji58:
            return wuji58.state58_from_obs(obs["state"], self.wrist_frame).astype(np.float32)
        return pack_robot_state(
            obs,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)

    def _adapt_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        vision = obs["vision"]
        state = self._pack_state(obs)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"Expected a finite {self.action_dim}D RoboDojo state, got {state.shape}"
            )
        adapted = {
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
        return adapted

    def _raw_state(self, obs: dict[str, Any]) -> dict[str, np.ndarray] | None:
        """Standard-key wrist/hand state a wuji58 chunk is decoded against (None otherwise)."""
        if not self.wuji58:
            return None
        return {
            key: np.asarray(obs["state"][key], dtype=np.float64).copy()
            for key in wuji58.STATE_KEYS
        }

    def _ingest(
        self, session, obs: dict[str, Any]
    ) -> tuple[dict[str, Any], str, dict[str, np.ndarray] | None]:
        adapted = self._adapt_obs(obs)
        if session is not None and session.pending_model_actions:
            session.update_obs(adapted)
        return adapted, instruction(obs, self.default_instruction), self._raw_state(obs)

    def update_obs(self, obs):
        self.last_obs, self.last_instruction, self.last_raw_state = self._ingest(
            self.session, obs
        )

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

    def _dummy_actions(
        self, raw_state: dict[str, np.ndarray] | None = None
    ) -> list[dict[str, np.ndarray]]:
        if self.wuji58:
            # Hold position: zero wrist delta (identity rot6d), current hand joints.
            if raw_state is None:
                raise ValueError("wuji58 dummy actions need the current observation")
            hold_wrist = np.concatenate([np.zeros(3), np.eye(3)[:2].reshape(6)])
            hold = wuji58.pack58(
                raw_state["left_ee_joint_state"],
                hold_wrist,
                raw_state["right_ee_joint_state"],
                hold_wrist,
            )
            chunk = np.tile(hold[None, :], (self.replan_steps, 1))
            return wuji58.decode_action_chunk58(chunk, raw_state, self.wrist_frame)
        zeros = np.zeros((self.replan_steps, self.action_dim), dtype=np.float32)
        return unpack_robot_state(
            zeros, self.action_type, self.robot_action_dim_info, source_type="obs"
        )

    def _predict(
        self,
        session,
        observation: dict[str, Any],
        instruction_text: str,
        raw_state: dict[str, np.ndarray] | None = None,
    ):
        if self.allow_dummy_policy:
            return self._dummy_actions(raw_state)
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
        if self.wuji58:
            # Every element decodes against the observation the chunk was predicted from.
            return wuji58.decode_action_chunk58(packed, raw_state, self.wrist_frame)
        for gripper in self.gripper_slices:
            packed[:, gripper] = np.clip(packed[:, gripper], 0.0, 1.0)
        return unpack_robot_state(
            packed, self.action_type, self.robot_action_dim_info, source_type="obs"
        )

    def get_action(self):
        if self.last_obs is None:
            raise ValueError("Call update_obs before get_action")
        return self._predict(
            self.session, self.last_obs, self.last_instruction, self.last_raw_state
        )

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
        self.last_raw_state = None
        self.env_obs.clear()
        self.env_order = []
        for session in (self.session, *self.env_sessions.values()):
            if session is not None:
                session.reset_model()
