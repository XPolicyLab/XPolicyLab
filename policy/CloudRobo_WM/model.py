"""CloudRobo-WM adapter for decoded RGB and joint-space actions."""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import json
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
SOURCE_ROOT = POLICY_ROOT / "source_inference"
CAMERAS = {
    "cam_head": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def _runtime_policy_class():
    package_root = (SOURCE_ROOT / "src" / "cloudrobo_wm").resolve()
    if not (package_root / "policy.py").is_file():
        raise FileNotFoundError("The bundled inference runtime is missing.")
    runtime_path = str(SOURCE_ROOT / "src")
    if runtime_path not in sys.path:
        sys.path.insert(0, runtime_path)
    package = importlib.import_module("cloudrobo_wm")
    if Path(package.__file__).resolve().parent != package_root:
        raise RuntimeError("An unrelated inference runtime is already imported.")
    module = importlib.import_module("cloudrobo_wm.policy")
    if Path(module.__file__).resolve() != package_root / "policy.py":
        raise RuntimeError("The policy module did not resolve to this package.")
    return module.CloudRoboPolicy


def observation_to_robotwin(observation: Mapping, layout: dict) -> dict:
    cameras = {}
    for source_name, target_name in CAMERAS.items():
        try:
            color = np.asarray(observation["vision"][source_name]["color"])
        except KeyError as error:
            raise ValueError(f"Missing required RGB camera: {source_name}.") from error
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[-1] != 3:
            raise ValueError(f"{source_name} must contain decoded uint8 RGB.")
        cameras[target_name] = {"rgb": color}

    joints = np.asarray(pack_robot_state(observation, "joint", layout), dtype=np.float32)
    action_dim = sum(layout["arm_dim"]) + sum(layout["ee_dim"])
    if joints.shape != (action_dim,) or not np.isfinite(joints).all():
        raise ValueError("Expected a finite dual-arm joint observation.")
    state = observation["state"]
    record = {
        "observation": cameras,
        "joint_action": {"vector": joints},
        "endpose": {},
    }
    for side in ("left", "right"):
        arm = np.asarray(state[f"{side}_arm_joint_state"], dtype=np.float32)
        gripper = float(np.asarray(state[f"{side}_ee_joint_state"]).reshape(-1)[0])
        record["joint_action"][f"{side}_arm"] = arm
        record["joint_action"][f"{side}_gripper"] = gripper
        record["endpose"][f"{side}_gripper"] = gripper
        if f"{side}_ee_pose" in state:
            record["endpose"][f"{side}_endpose"] = np.asarray(
                state[f"{side}_ee_pose"], dtype=np.float32
            )
    return record


class Model(ModelTemplate):
    """Serve one environment and execute joint actions with continuous grippers."""

    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        if self.model_cfg.get("action_type") != "joint":
            raise ValueError("This policy supports action_type=joint.")
        if str(self.model_cfg.get("eval_batch", False)).lower() == "true":
            raise ValueError("Use eval_batch=false and one environment per server.")
        if self.model_cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("This policy supports env_cfg_type=arx_x5.")
        self.layout = get_robot_action_dim_info(self.model_cfg["env_cfg_type"])
        self.action_dim = sum(self.layout["arm_dim"]) + sum(self.layout["ee_dim"])
        if self.layout["arm_dim"] != [6, 6] or self.layout["ee_dim"] != [1, 1]:
            raise ValueError("The configured embodiment has an incompatible layout.")
        if os.environ.get("CLOUDROBO_FIXED_WEIGHTS"):
            checkpoint = Path(os.environ["CLOUDROBO_FIXED_WEIGHTS"]).expanduser().resolve()
            asset_root = Path(os.environ["CLOUDROBO_FIXED_ASSETS"]).expanduser().resolve()
            self.checkpoint_root = checkpoint.parent
        else:
            self.checkpoint_root = resolve_checkpoint_root(
                self.model_cfg, POLICY_ROOT / "checkpoints", policy_dir=POLICY_ROOT
            )
            checkpoint = self.checkpoint_root / "weights" / "cloudrobo_wm_bf16.safetensors"
            asset_root = self.checkpoint_root / "weights" / "third_party"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing inference weights: {checkpoint}")
        policy_class = _runtime_policy_class()
        self.model = policy_class(
            checkpoint_path=str(checkpoint),
            model_config_path=str(SOURCE_ROOT / "configs" / "model.json"),
            normalization_path=str(SOURCE_ROOT / "configs" / "normalization.json"),
            asset_root=str(asset_root),
            device=self.model_cfg.get("device") or "cuda",
            num_inference_steps=int(self.model_cfg.get("num_inference_steps", 4)),
            replan_steps=int(self.model_cfg.get("replan_steps", 32)),
            seed=int(self.model_cfg.get("inference_seed", 42)),
        )
        self._observation = None
        self._instruction = ""
        self._env_idx = 0
        self.last_prediction = None
        self._case_metadata = None

    def prepare_case(self, case_metadata):
        metadata = dict(case_metadata)
        self._case_metadata = {
            name: metadata[name]
            for name in ("task_name", "seed", "instruction", "action_type")
            if name in metadata
        }
        print(json.dumps({"event": "case_prepared", **self._case_metadata}, ensure_ascii=False), flush=True)

    def on_trial_end(self, trial_metadata):
        metadata = dict(trial_metadata)
        print(
            json.dumps(
                {"event": "trial_ended", **{
                    name: metadata[name]
                    for name in ("task_name", "seed", "success")
                    if name in metadata
                }},
                ensure_ascii=False,
            ),
            flush=True,
        )

    def update_obs(self, observation):
        self._observation = observation_to_robotwin(observation, self.layout)
        instruction = observation.get("instruction")
        if not instruction:
            instructions = observation.get("instructions") or []
            instruction = instructions[0] if instructions else ""
        self._instruction = str(instruction)
        self._env_idx = int(observation.get("env_idx", 0))
        self.model.observe(self._observation)

    def update_obs_batch(self, observations):
        if len(observations) != 1:
            raise ValueError("Only one environment is supported per server.")
        self.update_obs(observations[0])

    def get_action(self):
        if self._observation is None:
            raise RuntimeError("Call update_obs before requesting actions.")
        self.last_prediction = None
        prediction = self.model.predict_with_eef(self._observation, self._instruction)
        actions = np.asarray(prediction["joint14"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != self.action_dim:
            raise ValueError("Expected a nonempty joint action chunk.")
        if not np.isfinite(actions).all():
            raise ValueError("The action chunk contains a nonfinite value.")
        self.last_prediction = {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in prediction.items()
            if name != "joint14"
        }
        return unpack_robot_state(actions, "joint", self.layout)

    def get_eef_prediction(self):
        if self.last_prediction is None:
            raise RuntimeError("Request an action chunk before its auxiliary prediction.")
        return {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in self.last_prediction.items()
        }

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(env_idx_list) != [self._env_idx]:
            raise ValueError("Only the observed environment can request actions.")
        return [self.get_action()]

    def reset(self):
        self.model.reset()
        self._observation = None
        self._instruction = ""
        self._env_idx = 0
        self.last_prediction = None
