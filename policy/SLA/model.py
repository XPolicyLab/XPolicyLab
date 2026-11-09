"""SLA RoboTwin adapter. Observation images are decoded by the policy server."""
from pathlib import Path
import sys

import numpy as np

_POLICY_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _POLICY_DIR.parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from .adapter_utils import (
    DEFAULT_BASE_POSE, eef_actions_to_xpl, encode_observation, finite_array,
    resolve_model_checkpoint, resolve_norm_asset, validate_robot_dims, validate_support,
)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        validate_support(model_cfg.get("bench_name"), model_cfg.get("env_cfg_type"), model_cfg.get("action_type"))
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        validate_robot_dims(self.robot_action_dim_info)
        self.base_pose = model_cfg.get("robot_base_pose", DEFAULT_BASE_POSE)
        self.chunk_size = int(model_cfg.get("action_chunk_size", 50))
        if not 1 <= self.chunk_size <= 50:
            raise ValueError("action_chunk_size must be in [1, 50].")
        self.policy = self._load_policy(model_cfg)
        self.model = self.policy
        self.reset()

    def _load_policy(self, model_cfg):
        import dataclasses
        from openpi.policies import policy_config, robotwin_policy
        from openpi.shared import normalize
        from openpi.training import config

        if model_cfg.get("train_config_name", "train_robotwin") != "train_robotwin":
            raise ValueError("This adapter supports the released train_robotwin configuration.")
        checkpoint = resolve_model_checkpoint(model_cfg)
        asset = resolve_norm_asset(checkpoint, model_cfg.get("norm_stats_asset_id"))
        train_cfg = config.get_config("train_robotwin")
        stats = normalize.load(asset)
        expected = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        for key in ("state", "actions"):
            if key not in stats:
                raise ValueError(f"Normalization asset has no {key} statistics: {asset}")
            for statistic in ("mean", "std", "q01", "q99"):
                finite_array(getattr(stats[key], statistic), f"{key}.{statistic}", (expected,))
        train_cfg = dataclasses.replace(train_cfg, data=dataclasses.replace(
            train_cfg.data, assets=config.AssetsConfig(assets_dir=str(checkpoint / "assets"), asset_id=asset.name),
        ))
        self.common_origin = robotwin_policy._robotwin_common_eef_origin()
        steps = int(model_cfg.get("num_inference_steps", 10))
        if steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        policy = policy_config.create_trained_policy(
            train_cfg, checkpoint, norm_stats=stats, sample_kwargs={"num_steps": steps},
        )
        import jax
        policy._rng = jax.random.key(int(model_cfg.get("seed") or 0))
        print(f"[SLA] Loaded {checkpoint}; normalization={asset.name}; actions=ee")
        return policy

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        observations = {}
        for index, obs in enumerate(obs_list):
            env_idx = int(obs.get("env_idx", index))
            if env_idx in observations:
                raise ValueError(f"Duplicate env_idx: {env_idx}")
            observations[env_idx] = encode_observation(obs, self.robot_action_dim_info)
        self._observations = observations

    def get_action(self):
        if len(self._observations) != 1:
            raise ValueError("get_action requires exactly one observation; use update_obs first.")
        return self.get_action_batch()[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) == 0:
            return []
        if not self._observations:
            raise ValueError("Call update_obs/update_obs_batch before requesting actions.")
        indices = list(self._observations) if env_idx_list is None else [int(i) for i in env_idx_list]
        if len(indices) != len(set(indices)):
            raise ValueError("env_idx_list contains duplicates.")
        missing = set(indices) - self._observations.keys()
        if missing:
            raise KeyError(f"No current observation for env_idx {sorted(missing)}.")
        result = []
        for env_idx in indices:
            actions = np.asarray(self.policy.infer(self._observations[env_idx])["actions"])
            if actions.ndim != 2 or actions.shape[0] < self.chunk_size:
                raise ValueError(f"Expected at least {self.chunk_size} action steps; got {actions.shape}.")
            result.append(eef_actions_to_xpl(
                actions[:self.chunk_size], self.robot_action_dim_info, self.base_pose, self.common_origin,
            ))
        return result

    def reset(self):
        self._observations = {}
