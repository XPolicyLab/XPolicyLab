"""CloudRobo-WM bridge for the standard XPolicyLab policy server.

The public inference checkout remains separate from the post-training checkout.
Checkpoint directories use XPolicyLab's shared resolution rules.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import os
from pathlib import Path
import sys

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_ROOT = Path(__file__).resolve().parent
INFERENCE_ROOT = POLICY_ROOT / "source_inference"
CAMERAS = {
    "cam_head": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _load_runtime_policy(source_root: Path):
    """Import only the bundled, pinned public inference implementation."""
    package_root = (source_root / "src" / "crwm").resolve()
    if not (package_root / "policy.py").is_file():
        raise FileNotFoundError(
            f"Missing public inference checkout: {source_root}. Run install.sh first."
        )
    source_path = str(package_root.parent)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    package = importlib.import_module("crwm")
    if Path(package.__file__).resolve().parent != package_root:
        raise RuntimeError(
            "A different crwm package is already imported. Start the policy server "
            "in the inference environment installed by this policy's install.sh."
        )
    module = importlib.import_module("crwm.policy")
    if Path(module.__file__).resolve() != package_root / "policy.py":
        raise RuntimeError("crwm.policy did not resolve to the pinned inference checkout.")
    return module.CRWMPolicy


def observation_to_robotwin(obs: Mapping, layout: dict) -> dict:
    """Convert decoded RGB and native Joint14 using the official layout helper."""
    cameras = {}
    for source_name, target_name in CAMERAS.items():
        try:
            color = np.asarray(obs["vision"][source_name]["color"])
        except KeyError as exc:
            raise ValueError(f"CloudRobo-WM requires {source_name} RGB.") from exc
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[-1] != 3:
            raise ValueError(f"{source_name}.color must be decoded uint8 RGB (H, W, 3).")
        cameras[target_name] = {"rgb": color}

    state = obs["state"]
    joints = np.asarray(pack_robot_state(obs, "joint", layout), dtype=np.float32)
    if joints.shape != (14,) or not np.isfinite(joints).all():
        raise ValueError("CloudRobo-WM requires one finite dual-arm Joint14 observation.")
    record = {
        "observation": cameras,
        "joint_action": {"vector": joints},
        "endpose": {},
    }
    for side in ("left", "right"):
        record["joint_action"][f"{side}_arm"] = np.asarray(state[f"{side}_arm_joint_state"])
        gripper = float(np.asarray(state[f"{side}_ee_joint_state"]).reshape(-1)[0])
        record["joint_action"][f"{side}_gripper"] = gripper
        record["endpose"][f"{side}_gripper"] = gripper
        if f"{side}_ee_pose" in state:
            record["endpose"][f"{side}_endpose"] = np.asarray(state[f"{side}_ee_pose"])
    return record


class Model(ModelTemplate):
    """One environment per server; execute Joint14 and retain auxiliary EEF."""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        if self.model_cfg.get("action_type") != "joint":
            raise ValueError("CloudRobo-WM requires action_type=joint.")
        if str(self.model_cfg.get("eval_batch", False)).lower() == "true":
            raise ValueError("CloudRobo-WM requires eval_batch=false and num_workers=1 per server.")
        env_cfg_type = self.model_cfg.get("env_cfg_type")
        if not isinstance(env_cfg_type, str) or not env_cfg_type.strip():
            raise ValueError("env_cfg_type is required; use the trained arx_x5 embodiment.")
        if env_cfg_type != "arx_x5":
            raise ValueError("This checkpoint is fixed to env_cfg_type=arx_x5; dimensions alone do not identify an embodiment.")
        self.layout = get_robot_action_dim_info(env_cfg_type)
        if self.layout["arm_dim"] != [6, 6] or self.layout["ee_dim"] != [1, 1]:
            raise ValueError("This checkpoint requires two 6-joint arms and two 1D grippers.")

        self.checkpoint_root = resolve_checkpoint_root(
            self.model_cfg, POLICY_ROOT / "checkpoints", policy_dir=POLICY_ROOT
        )
        if not self.checkpoint_root.is_dir():
            raise ValueError(
                "Pass a checkpoint directory containing weights/cr_wm.pt and "
                "weights/third_party/, rather than a standalone tensor file."
            )
        checkpoint = self.checkpoint_root / "weights" / "cr_wm.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing native BF16 inference checkpoint: {checkpoint}")
        os.environ["CRWM_RELEASE_ROOT"] = str(self.checkpoint_root)
        policy_class = _load_runtime_policy(INFERENCE_ROOT)
        self.model = policy_class(
            checkpoint_path=str(checkpoint),
            model_config_path=str(INFERENCE_ROOT / "configs" / "model.json"),
            normalization_path=str(INFERENCE_ROOT / "configs" / "normalization.json"),
            device=self.model_cfg.get("device") or "cuda",
            num_inference_steps=int(self.model_cfg.get("num_inference_steps", 4)),
            replan_steps=int(self.model_cfg.get("replan_steps", 32)),
            seed=int(self.model_cfg.get("seed", 42)),
        )
        self._observation = None
        self._instruction = ""
        self._env_idx = 0
        self.last_prediction = None

    def update_obs(self, obs):
        self._observation = observation_to_robotwin(obs, self.layout)
        instruction = obs.get("instruction")
        if not instruction:
            instructions = obs.get("instructions") or []
            instruction = instructions[0] if instructions else ""
        self._instruction = str(instruction)
        self._env_idx = int(obs.get("env_idx", 0))
        self.model.observe(self._observation)

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("CloudRobo-WM supports one environment per policy server.")
        self.update_obs(obs_list[0])

    def get_action(self):
        if self._observation is None:
            raise RuntimeError("Call update_obs() before get_action().")
        self.last_prediction = None
        prediction = self.model.predict_with_eef(self._observation, self._instruction)
        actions = np.asarray(prediction["joint14"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != 14:
            raise ValueError("CloudRobo-WM must return a nonempty (chunk_length, 14) array.")
        if not np.isfinite(actions).all():
            raise ValueError("CloudRobo-WM returned a nonfinite joint action.")
        poses = np.asarray(prediction["eef_pose"], dtype=np.float32)
        relative = np.asarray(prediction["eef_relative"], dtype=np.float32)
        anchor = np.asarray(prediction["anchor_pose"], dtype=np.float32)
        if (
            poses.shape != (len(actions), 2, 7)
            or relative.shape != (len(actions), 2, 6)
            or anchor.shape != (2, 7)
            or not np.isfinite(poses).all()
            or not np.isfinite(relative).all()
            or not np.isfinite(anchor).all()
            or prediction["eef_frame"] != "raw_fl_link6_fr_link6_world"
        ):
            raise ValueError("Invalid raw-link world EEF prediction or fixed anchor.")
        self.last_prediction = dict(
            eef_frame=prediction["eef_frame"],
            pose_layout="xyz_quaternion_wxyz",
            eef_pose=poses.copy(),
            eef_relative=relative.copy(),
            anchor_pose=anchor.copy(),
        )
        # The official environment accepts only its standard action schema.
        # Raw-link EEF predictions remain available via a separate RPC.
        return unpack_robot_state(actions, "joint", self.layout)

    def get_eef_prediction(self):
        """Return the latest chunk's auxiliary EEF prediction without replanning."""
        if self.last_prediction is None:
            raise RuntimeError("Call get_action() to compute an EEF prediction first.")
        return {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in self.last_prediction.items()
        }

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(env_idx_list) != [self._env_idx]:
            raise ValueError("CloudRobo-WM supports the single observed environment per server.")
        return [self.get_action()]

    def reset(self):
        self.model.reset()
        self._observation = None
        self._instruction = ""
        self._env_idx = 0
        self.last_prediction = None
