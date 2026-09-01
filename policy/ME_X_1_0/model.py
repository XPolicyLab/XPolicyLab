from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info


POLICY_DIR = Path(__file__).resolve().parent
XPL_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = XPL_ROOT.parent
DEFAULT_RUNTIME = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "me_x_1_0"
    / "source"
    / "runtime"
)


def _required_path(value: Any, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"deploy.yml must define {name}")
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def _instruction(obs: dict[str, Any]) -> str:
    value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("ME_X_1_0 requires a non-empty instruction")
    return value.strip()


class Model(ModelTemplate):
    """Eval-only ME-X-1.0 adapter for Aloha-AgileX joint control."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg["action_type"]
        self.env_cfg_type = model_cfg["env_cfg_type"]
        if self.action_type != "joint":
            raise ValueError("ME_X_1_0 only supports action_type=joint")
        if self.env_cfg_type != "arx_x5":
            raise ValueError("ME_X_1_0 only supports env_cfg_type=arx_x5")
        self.robot_dims = get_robot_action_dim_info(self.env_cfg_type)
        arm_dims = self.robot_dims["arm_dim"]
        ee_dims = self.robot_dims["ee_dim"]
        if len(arm_dims) != 2 or len(ee_dims) != 2:
            raise ValueError(f"ME_X_1_0 requires a bimanual robot: {self.robot_dims}")
        self.action_layout = (
            ("left_arm_joint_state", arm_dims[0]),
            ("left_ee_joint_state", ee_dims[0]),
            ("right_arm_joint_state", arm_dims[1]),
            ("right_ee_joint_state", ee_dims[1]),
        )
        self.action_dim = sum(size for _, size in self.action_layout)
        upstream_value = (
            model_cfg.get("upstream_policy_path")
            or os.environ.get("MEX_RUNTIME_PATH")
            or DEFAULT_RUNTIME
        )
        upstream = _required_path(upstream_value, "upstream_policy_path")
        sys.path.insert(0, str(upstream))
        runtime = importlib.import_module("policy")

        checkpoint = resolve_checkpoint_root(model_cfg, BENCH_ROOT / "checkpoints")
        checkpoint = _required_path(checkpoint, "checkpoint_path/ckpt_name")
        wan_path = _required_path(
            model_cfg.get("wan_path")
            or os.environ.get("MEX_WAN_PATH")
            or checkpoint / "wan",
            "wan_path/MEX_WAN_PATH",
        )
        cadence = float(model_cfg.get("tactile_frame_interval_seconds", 0.06))
        if not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("tactile_frame_interval_seconds must be finite and positive")

        self.policy = runtime.MEXPolicy(
            checkpoint_path=str(checkpoint),
            wan_path=str(wan_path),
            tactile_frame_interval_seconds=cadence,
            input_color_order=str(model_cfg.get("input_color_order", "rgb")),
        )
        self._obs_ready = False

    @staticmethod
    def _camera(obs: dict[str, Any], name: str) -> np.ndarray:
        try:
            image = np.asarray(obs["vision"][name]["color"])
        except KeyError as error:
            raise ValueError(f"Missing XPolicyLab camera vision.{name}.color") from error
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Invalid {name} image shape: {image.shape}")
        return image

    def update_obs(self, obs: dict[str, Any]) -> None:
        state = obs.get("state", {})
        parts = []
        for name, size in self.action_layout:
            value = np.asarray(state.get(name), dtype=np.float32)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid state.{name}: expected {(size,)}, got {value.shape}")
            parts.append(value)
        qpos = np.concatenate(parts)

        self.policy.update_observation(
            head_rgb=self._camera(obs, "cam_head"),
            left_wrist_rgb=self._camera(obs, "cam_left_wrist"),
            right_wrist_rgb=self._camera(obs, "cam_right_wrist"),
            qpos=qpos,
            instruction=_instruction(obs),
        )
        self._obs_ready = True

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        if len(obs_list) != 1:
            raise RuntimeError("ME_X_1_0 does not support batched inference")
        self.update_obs(obs_list[0])

    def get_action(self) -> list[dict[str, np.ndarray]]:
        if not self._obs_ready:
            raise RuntimeError("Call update_obs before get_action")
        chunk = np.asarray(self.policy.get_action(), dtype=np.float32)
        if chunk.shape != (16, self.action_dim) or not np.isfinite(chunk).all():
            raise RuntimeError(f"Invalid ME-X-1.0 action chunk: {chunk.shape}")
        actions = []
        for vector in chunk:
            action = {}
            offset = 0
            for name, size in self.action_layout:
                action[name] = vector[offset : offset + size].copy()
                offset += size
            actions.append(action)
        return actions

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != 1:
            raise RuntimeError("ME_X_1_0 does not support batched inference")
        return [self.get_action()]

    def reset(self) -> None:
        self.policy.reset()
        self._obs_ready = False
