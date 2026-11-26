"""Every episode starts in code; an explicit task request switches to WAM."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from XPolicyLab.model_template import ModelTemplate

from .gm.isolated import IsolatedGm
from .skill_switch import SkillSwitchController


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.code = IsolatedGm(model_cfg)
        self.wam = None
        self.skill_switch = SkillSwitchController(self)
        print("[go3_agent_preview] code policy ready; WAM starts on an explicit task switch", flush=True)

    def _wam(self):
        if self.wam is None:
            from .wam.model import Model as WamModel
            self.wam = WamModel(self.model_cfg)
        return self.wam

    @staticmethod
    def _env_indices(obs_list: Sequence[Mapping]) -> list[int]:
        indices = [int(obs.get("env_idx", i)) for i, obs in enumerate(obs_list)]
        if len(indices) != len(set(indices)):
            raise ValueError("observation batch contains duplicate env_idx values")
        return indices

    def update_obs(self, obs):
        self.skill_switch.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self.skill_switch.update_obs_batch(obs_list)

    def get_action(self):
        return self.skill_switch.get_action_batch(self.skill_switch.latest[:1])[0]

    def get_action_batch(self, env_idx_list=None):
        return self.skill_switch.get_action_batch(env_idx_list)

    def reset(self):
        self.skill_switch.reset()
        self.code.reset()
        if self.wam is not None:
            self.wam.reset()

    def close(self):
        self.skill_switch.reset("close")
        self.code.close()
        if self.wam is not None:
            self.wam.close()
            self.wam = None
