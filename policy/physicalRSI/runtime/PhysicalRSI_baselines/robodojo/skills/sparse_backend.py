"""RoboDojo XPolicyLab adapter that records every observed simulator step."""

from __future__ import annotations

import os
from typing import Any

import jax
import numpy as np
from .pi05_backend import Model as Pi05Model
from .pi05_backend import encode_obs
from XPolicyLab.utils.process_data import unpack_robot_state

from openpi.models.pi0_config import SparseVisualMemoryConfig
from openpi.policies.sparse_memory import SparseVisualHistory


def _stack_observations(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([item["state"] for item in items]),
        "images": {key: np.stack([item["images"][key] for item in items]) for key in items[0]["images"]},
        "history_images": {
            key: np.stack([item["history_images"][key] for item in items]) for key in items[0]["history_images"]
        },
        "history_image_masks": {
            key: np.stack([item["history_image_masks"][key] for item in items])
            for key in items[0]["history_image_masks"]
        },
        "prompt": [item["prompt"] for item in items],
    }


def _slice_observation(batch: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "state": batch["state"][index],
        "images": {key: value[index] for key, value in batch["images"].items()},
        "history_images": {key: value[index] for key, value in batch["history_images"].items()},
        "history_image_masks": {key: value[index] for key, value in batch["history_image_masks"].items()},
        "prompt": batch["prompt"][index],
    }


class Model(Pi05Model):
    """Execution skill backend with isolated causal visual histories."""

    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__(model_cfg)
        sparse = self.policy._model.sparse_memory_config  # noqa: SLF001
        if sparse is None:
            raise ValueError("RoboDojo sparse adapter requires a sparse-memory checkpoint config")
        self._visual_history = SparseVisualHistory(sparse)
        self._memory_camera_map = {
            "base_0_rgb": "cam_high",
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [int(obs.get("env_idx", index)) for index, obs in enumerate(obs_list)]
        items = []
        for env_idx, observation in zip(self._latest_env_idx_list, obs_list, strict=True):
            item = encode_obs(observation, self.action_type, self.robot_action_dim_info)
            self._visual_history.append(
                env_idx,
                {key: item["images"][self._memory_camera_map[key]] for key in self._visual_history.config.image_keys},
            )
            item.update(self._visual_history.materialize(env_idx))
            items.append(item)
        self.observation_window = _stack_observations(items)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        env_idx_list = [int(value) for value in (env_idx_list or self._latest_env_idx_list)]
        action_list = []
        for batch_index, env_idx in enumerate(env_idx_list):
            observation = _slice_observation(self.observation_window, batch_index)
            if self.ensemble_seeds:
                samples = []
                for seed in self.ensemble_seeds:
                    self.policy._rng = jax.random.key(seed)  # noqa: SLF001
                    samples.append(self.policy.infer(observation, **kwargs)["actions"])
                actions = np.mean(np.stack(samples), axis=0)
            else:
                if self.reset_policy_rng_each_call:
                    self.policy._rng = jax.random.key(self.policy_seed)  # noqa: SLF001
                actions = self.policy.infer(observation, **kwargs)["actions"]
            if self.temporal_ensemble_decay is not None:
                history = self._action_history.setdefault(env_idx, [])
                history.append(np.asarray(actions))
                max_history = max(1, (len(actions) + self.exec_horizon - 1) // self.exec_horizon)
                del history[:-max_history]
                blended = []
                for action_index in range(min(self.exec_horizon, len(actions))):
                    candidates, ages = [], []
                    for age, previous_actions in enumerate(reversed(history)):
                        previous_index = age * self.exec_horizon + action_index
                        if previous_index < len(previous_actions):
                            candidates.append(previous_actions[previous_index])
                            ages.append(age)
                    weights = np.exp(-self.temporal_ensemble_decay * np.asarray(ages, dtype=np.float32))
                    weights /= weights.sum()
                    blended.append(np.tensordot(weights, np.stack(candidates), axes=(0, 0)))
                actions = np.stack(blended)
            actions = actions[: self.exec_horizon]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )
            self._debug_action_calls += 1
        return action_list

    def reset(self):
        super().reset()
        self._visual_history.reset()


def sparse_config_from_environment() -> SparseVisualMemoryConfig:
    """Utility for explicit adapter tests and external launch validation."""
    return SparseVisualMemoryConfig(
        history_frames=int(os.environ.get("PI05_MEMORY_HISTORY_FRAMES", "20")),
        history_stride=int(os.environ.get("PI05_MEMORY_HISTORY_STRIDE", "25")),
        pool_size=tuple(int(value) for value in os.environ.get("PI05_MEMORY_POOL_SIZE", "4,4").split(",")),
        image_keys=tuple(os.environ.get("PI05_MEMORY_IMAGE_KEYS", "base_0_rgb").split(",")),
    )
