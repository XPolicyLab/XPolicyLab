"""Evaluate saved LeRobot policies through their public, per-frame API."""

from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.relative_action_processor import bind_relative_anchor
import numpy as np
import torch

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.action_type = model_cfg["action_type"]
        if self.action_type != "joint" or model_cfg.get("eval_batch", False):
            raise ValueError("LeRobot currently supports joint actions and one environment only.")
        self.robot_dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        policy_dir = Path(__file__).resolve().parent
        checkpoint = resolve_checkpoint_root(model_cfg, policy_dir / "checkpoints")
        device = model_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        action_dim = sum(self.robot_dims["arm_dim"]) + sum(self.robot_dims["ee_dim"])
        if config.output_features["action"].shape != (action_dim,):
            raise ValueError("Checkpoint action dimensions do not match the configured robot.")
        self.camera_mapping = model_cfg["camera_mapping"]
        if set(self.camera_mapping) != set(config.image_features):
            raise ValueError("camera_mapping must map every checkpoint image key to a runtime camera.")
        self.policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        bind_relative_anchor(self.policy, self.preprocessor)
        self.default_prompt = model_cfg.get("prompt") or model_cfg["task_name"]
        self.reset()

    def update_obs(self, obs):
        self.observation = obs

    @torch.inference_mode()
    def get_action(self):
        if self.observation is None:
            raise RuntimeError("Call update_obs once per control frame before get_action.")
        obs = self.observation
        self.observation = None
        state = pack_robot_state(obs, self.action_type, self.robot_dims).astype(np.float32)
        batch = {
            "observation.state": torch.from_numpy(state),
            "task": obs.get("instruction") or obs.get("instructions") or self.default_prompt,
        }
        for key, camera in self.camera_mapping.items():
            image = np.asarray(obs["vision"][camera]["color"])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Camera {camera} must provide an HWC uint8 RGB image.")
            batch[key] = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        # One action per request keeps MEM history at the control-frame frequency;
        # LeRobot owns chunk prediction, queued actions, normalization and padding.
        action = self.postprocessor(self.policy.select_action(self.preprocessor(batch)))
        return [unpack_robot_state(action.squeeze(0).cpu().numpy(), self.action_type, self.robot_dims)]

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("LeRobot currently supports one environment only.")
        self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != 1:
            raise ValueError("LeRobot currently supports one environment only.")
        return [self.get_action()]

    def reset(self):
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
        self.observation = None
