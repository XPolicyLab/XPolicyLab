"""Keep every task seed when the rollout batch does not divide its seed pool."""

import json
from pathlib import Path

from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv as NativeRoboTwinEnv
import torch

from ..layouts import LayoutSampler


class RoboTwinEnv(NativeRoboTwinEnv):
    def _init_reset_state_ids(self):
        path = Path(self.cfg.seeds_path)
        seeds = None
        if path.is_file():
            with path.open() as stream:
                seeds = json.load(stream)[self.task_name].get("success_seeds")
        if seeds is None:
            return super()._init_reset_state_ids()
        self.layout_sampler = LayoutSampler(
            seeds,
            seed=self.base_seed,
            groups=self.num_group,
            rank=self.seed_offset,
            world_size=self.total_num_processes,
        )
        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None):
        if not hasattr(self, "layout_sampler"):
            return super().update_reset_state_ids(env_idx)
        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return
        ids = torch.from_numpy(self.layout_sampler.take()).repeat_interleave(self.group_size)
        if env_idx is None:
            self.reset_state_ids = ids
        else:
            env_idx = torch.as_tensor(env_idx, device=ids.device, dtype=torch.long)
            self.reset_state_ids[env_idx] = ids[env_idx]
