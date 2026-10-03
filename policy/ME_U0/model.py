"""ME-U0 evaluation adapter for RoboDojo's ARX-X5 robot."""
from pathlib import Path
import os
import random
import sys

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent / "me_u0"))

import numpy as np
import torch
from omegaconf import OmegaConf

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from leap.core.config import instantiate
from leap.data.world_unified.metadata import metadata_by_dataset_from_config
from leap.models.builder import load_pretrained_weights
from leap.serving.ME_U0_policy import MachEmbodiedUnifiedPolicy
from leap.serving.me_u0_robodojo_codec import build_robodojo_codecs_from_config
from scripts.ME_U0.robodojo.robodojo_xpolicy_policy import (
    CAMERA_ORDER, JOINT_STATE_KEYS, _extract_image,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        if model_cfg["env_cfg_type"] != "arx_x5" or model_cfg["action_type"] != "joint":
            raise ValueError("This checkpoint supports arx_x5 joint actions only")
        seed = int(model_cfg.get("seed") or 0)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        dims = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        self.widths = [width for pair in zip(dims["arm_dim"], dims["ee_dim"]) for width in pair]
        root = resolve_checkpoint_root(model_cfg, Path(__file__).parent / "checkpoints")
        cfg = OmegaConf.load(root / "config.yaml")
        cfg.model.checkpoint_root = str(root / "assets")
        cfg.model.load_backbone_weights = False
        stats_path = root / "stats/robodojo_sim_arx_x5_delta_joint_horizon48.json"
        codecs = build_robodojo_codecs_from_config(cfg, stats_path_override=str(stats_path))
        self.dataset_name, codec = next(iter(codecs.items()))
        self.model = instantiate(cfg.model).to(torch.bfloat16)
        load_pretrained_weights(self.model, str(root / "model.pt"), strict=True)
        self.model.to("cuda").eval()
        source = cfg.data.train.datasets[0]
        self.policy = MachEmbodiedUnifiedPolicy(
            self.model,
            per_view_size=tuple(source.decode_resize),
            mosaic_layout="pyramid",
            quantize_images=True,
            raw_action_dim=codec.canonical_action_dim,
            num_inference_steps=int(model_cfg.get("num_inference_steps", 12)),
            domain_id=int(source.logical_domain_id),
            max_text_len=int(cfg.model.max_text_len),
            device="cuda",
            action_codecs=codecs,
            domain_ids={self.dataset_name: int(source.logical_domain_id)},
            metadata=metadata_by_dataset_from_config(cfg),
        )
        self.action_chunk_size = int(model_cfg.get("action_chunk_size", 36))
        self.reset()

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self.indices = []
        for i, obs in enumerate(obs_list):
            index = int(obs.get("env_idx", i))
            self.indices.append(index)
            self.observations[index] = obs

    def _predict(self, index):
        obs = self.observations[index]
        state = np.concatenate([
            np.asarray(obs["state"][key], dtype=np.float32).reshape(width)
            for key, width in zip(JOINT_STATE_KEYS, self.widths)
        ])
        instruction = obs.get("instruction", obs.get("instructions"))
        if isinstance(instruction, (list, tuple)):
            instruction, = instruction
        actions = self.policy.predict_action(
            images=[_extract_image(obs, name) for name in CAMERA_ORDER],
            instruction=instruction,
            state=state,
            dataset_name=self.dataset_name,
            action_horizon=self.action_chunk_size,
        )
        result = []
        for action in actions[:self.action_chunk_size]:
            fields = np.split(np.asarray(action, dtype=np.float32), np.cumsum(self.widths)[:-1])
            result.append({key: value.copy() for key, value in zip(JOINT_STATE_KEYS, fields)})
        return result

    def get_action(self):
        return self._predict(self.indices[0])

    def get_action_batch(self, env_idx_list=None):
        indices = self.indices if env_idx_list is None else env_idx_list
        return [self._predict(int(index)) for index in indices]

    def reset(self):
        self.observations = {}
        self.indices = []
