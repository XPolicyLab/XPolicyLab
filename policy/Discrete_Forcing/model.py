from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import candidate_checkpoint_roots
from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state


POLICY_DIR = Path(__file__).resolve().parent


def _load_policy(checkpoint, model_root):
    sys.path.insert(0, str(model_root))
    import starVLA
    import torch
    from starVLA.model.framework.base_framework import baseframework

    if not Path(starVLA.__file__).resolve().is_relative_to(model_root.resolve()):
        raise ImportError("Another starVLA package is loaded; use a separate policy environment.")
    return baseframework.from_pretrained(str(checkpoint)).to(torch.bfloat16).to("cuda").eval()


class Model(ModelTemplate):
    """Published RoboTwin policy with native XPolicyLab observations and actions."""

    def __init__(self, model_cfg):
        if model_cfg.get("action_type", "joint") != "joint":
            raise ValueError("Discrete_Forcing supports absolute joint actions only.")
        self.robot_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if list(self.robot_info["arm_dim"]) != [6, 6] or list(self.robot_info["ee_dim"]) != [1, 1]:
            raise ValueError("This release requires two 6-joint arms with one gripper joint each.")
        self.action_dim = sum(self.robot_info["arm_dim"]) + sum(self.robot_info["ee_dim"])

        # Preserve checkpoint symlinks for the upstream run-directory sidecar lookup.
        candidates = candidate_checkpoint_roots(model_cfg, POLICY_DIR / "checkpoints", resolve=False)
        root = next((path for path in candidates if path.exists()), None)
        if root is None:
            raise FileNotFoundError(f"No checkpoint found among: {candidates}")
        checkpoint = root if root.is_file() else root / "checkpoints" / model_cfg.get(
            "checkpoint_file", "steps_60000_pytorch_model.pt"
        )
        if not checkpoint.is_file() or checkpoint.suffix != ".pt" or checkpoint.parent.name != "checkpoints":
            raise FileNotFoundError(f"Expected a .pt weight in its run's checkpoints/ directory: {checkpoint}")
        with (checkpoint.parent.parent / "config.yaml").open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        action_config = config["framework"]["action_model"]
        if config["framework"]["name"] != "QwenPILF_v3":
            raise ValueError("Expected a Discrete Forcing QwenPILF_v3 checkpoint.")
        if action_config.get("isolate_branch_tokens_before_shared") is not True:
            raise ValueError("Expected the published branch-isolated method checkpoint.")
        if config["datasets"]["vla_data"].get("include_state") is not False:
            raise ValueError("This adapter requires a checkpoint trained without state input.")
        if action_config["action_dim"] != self.action_dim or action_config["action_horizon"] != 50:
            raise ValueError("Expected the RoboTwin 50-step joint-action configuration.")
        self.action_chunk_size = int(action_config["action_horizon"])

        with (checkpoint.parent.parent / "dataset_statistics.json").open(encoding="utf-8") as stream:
            statistics = json.load(stream)
        key = model_cfg.get("unnorm_key", "new_embodiment")
        self.low = np.asarray(statistics[key]["action"]["min"], dtype=np.float32)
        self.high = np.asarray(statistics[key]["action"]["max"], dtype=np.float32)
        self.mask = np.asarray(statistics[key]["action"]["mask"], dtype=bool)
        for values in (self.low, self.high, self.mask):
            if values.shape != (self.action_dim,):
                raise ValueError("Action statistics do not match the robot action dimension.")
        if not self.mask[:-2].all() or self.mask[-2:].any():
            raise ValueError("Expected min-max arm statistics followed by two binary grippers.")
        if not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or (self.high < self.low).any():
            raise ValueError("Invalid action normalization bounds.")

        model_root = Path(os.environ.get("DF_ROOT") or model_cfg.get("model_root", "source_discrete_forcing")).expanduser()
        if not model_root.is_absolute():
            model_root = POLICY_DIR / model_root
        if not (model_root / "starVLA" / "model" / "framework" / "QwenPILF_v3.py").is_file():
            raise FileNotFoundError(f"Discrete Forcing code not found at {model_root}; run install.sh or set DF_ROOT.")
        self.model = _load_policy(checkpoint, model_root)
        self.reset()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        indices = [int(obs.get("env_idx", 0)) for obs in obs_list]
        if len(indices) != len(set(indices)):
            raise ValueError("Batched observations require distinct env_idx values.")
        for index, obs in zip(indices, obs_list):
            images = []
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                image = np.asarray(obs["vision"][camera]["color"])
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"Expected decoded HWC uint8 RGB image for {camera}.")
                images.append(cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA))
            instruction = obs.get("instruction") or obs.get("instructions")
            if isinstance(instruction, (list, tuple)):
                instruction = instruction[0] if instruction else None
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("Observation must contain a nonempty task instruction.")
            self.observations[index] = {"lang": instruction, "image": images}
        self.latest_indices = indices

    def get_action(self):
        if len(self.latest_indices) != 1:
            raise ValueError("Call update_obs with one observation before get_action.")
        return self.get_action_batch(self.latest_indices)[0]

    def get_action_batch(self, env_idx_list=None):
        indices = self.latest_indices if env_idx_list is None else list(env_idx_list)
        if not indices:
            return []
        examples = [self.observations[int(index)] for index in indices]
        output = self.model.predict_action(examples=examples, do_sample=False, use_ddim=True)
        normalized = np.asarray(output["normalized_actions"], dtype=np.float32)
        expected = (len(indices), self.action_chunk_size, self.action_dim)
        if normalized.shape != expected or not np.isfinite(normalized).all():
            raise ValueError(f"Expected finite normalized actions with shape {expected}, got {normalized.shape}.")
        normalized = np.clip(normalized, -1, 1)
        actions = np.where(
            self.mask,
            0.5 * (normalized + 1) * (self.high - self.low) + self.low,
            (normalized >= 0.49).astype(np.float32),
        )
        # Training order is both arms followed by both grippers; XPolicyLab interleaves them.
        arm_size = self.robot_info["arm_dim"][0]
        both_arms = sum(self.robot_info["arm_dim"])
        actions = np.concatenate([
            actions[..., :arm_size], actions[..., both_arms:both_arms + 1],
            actions[..., arm_size:both_arms], actions[..., both_arms + 1:],
        ], axis=-1)
        return [unpack_robot_state(chunk, "joint", self.robot_info) for chunk in actions]

    def reset(self):
        self.observations = {}
        self.latest_indices = []
