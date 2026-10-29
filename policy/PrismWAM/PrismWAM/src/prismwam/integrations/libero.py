"""Run a released LIBERO model on raw LIBERO simulator observations.

The simulator sends both views in its raw OpenGL array orientation (no 180-degree
rotation), metric depth in metres (0 for invalid pixels) and the 8-D state
``[eef_pos, axis-angle(eef_quat), gripper_qpos]``. Actions come back in LIBERO's
convention, with the gripper binarized to -1 (open) / +1 (close).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from hydra.utils import instantiate

from prismwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from prismwam.datasets.libero_rgbd_dataset import _compose_views
from prismwam.deployment.history import RGBHistoryBuffer


class LiberoPolicy:
    """A loaded LIBERO model with the dataset processor it was trained with."""

    def __init__(self, policy):
        self.policy = policy
        config = policy.config
        processor = instantiate(config["data"]["train"]["processor"])
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(config["deployment"]["norm_stats_path"])
        )
        normalizers = processor.normalizer.normalizers
        self._state = normalizers["state"]["default"]
        self._action = normalizers["action"]["default"]
        self.depth_max = float(config["data"]["train"].get("depth_max_meters", 1.5))
        self.replan_steps = policy.replan_steps
        self.history = RGBHistoryBuffer.from_config(config)
        self._prompt = None
        self._observation = None

    def reset(self, instruction: str) -> None:
        self.policy.reset()
        self.history.reset()
        self._prompt = self.policy.format_prompt(instruction)
        self._observation = None

    def observe(
        self,
        rgb: Sequence[np.ndarray],
        depth: Sequence[np.ndarray],
        state: np.ndarray,
        step: int,
    ) -> None:
        """Agent and wrist views: ``rgb`` HxWx3 uint8, ``depth`` HxW metres."""
        views = [
            torch.from_numpy(np.array(view)).permute(2, 0, 1).float()[None] / 127.5 - 1
            for view in rgb
        ]
        frame = _compose_views(views)[0].contiguous()
        self.history.append(frame, step=int(step))
        depths = [
            torch.from_numpy(np.array(view, dtype=np.float32))[None, None]
            .clamp(0, self.depth_max)
            .mul(2 / self.depth_max)
            .sub(1)
            for view in depth
        ]
        state = torch.from_numpy(np.array(state, dtype=np.float32))[None]
        self._observation = {
            "input_image": _compose_views(depths).contiguous(),
            "rgb_input_image": frame[None],
            "rgb_history_images": self.history.tensor(),
            "proprio": self._state.forward(state),
            "prompt": self._prompt,
        }

    def act(self, seed: int = 42) -> np.ndarray:
        """The next ``replan_steps`` simulator actions, shape ``[replan_steps, 7]``."""
        if self._observation is None:
            raise RuntimeError("Call observe() before act().")
        predicted = self.policy.predict(self._observation, seed=seed)["action"]
        action = self._action.backward(predicted).numpy()[: self.replan_steps]
        # Training maps LIBERO's gripper command g to (1 - g) / 2.
        action[:, -1] = np.where(1 - 2 * action[:, -1] >= 0, 1.0, -1.0)
        return action
