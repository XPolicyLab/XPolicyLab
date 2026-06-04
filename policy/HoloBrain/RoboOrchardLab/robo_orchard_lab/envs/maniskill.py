# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from typing import Any, Optional, Sequence, TypeAlias

import gymnasium as gym
import numpy as np
import torch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import SimConfig
from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_lab.envs.base import EnvBase, EnvBaseCfg, EnvStepReturn

logger = LoggerManager().get_child(__name__)


ManiSkillEnvReturn: TypeAlias = EnvStepReturn[dict[str, Any], torch.Tensor]


class ManiSkillEnv(EnvBase[dict[str, Any], torch.Tensor]):
    """The ManiSkill environment base class.

    This class is the base class for all ManiSkill environments. It provides
    the basic functionality for the environment, such as reset, step, render,
    and close.
    """

    env: BaseEnv
    """The ManiSkill environment instance."""

    def __init__(self, cfg: ManiSkillEnvCfg):
        """Initialize the ManiSkill environment.

        Args:
            cfg (ManiSkillEnvCfg): The configuration for the environment.
        """
        self.cfg = cfg
        self.env = self.cfg.make_env()

    def step(
        self, action: None | np.ndarray | torch.Tensor | dict
    ) -> ManiSkillEnvReturn:
        ret: tuple = self.env.step(action)
        return ManiSkillEnvReturn(*ret)

    def reset(
        self,
        seed: int | list[int] | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict | None = None,
    ) -> Any:
        if options is None:
            options = {}
        if env_ids is not None:
            opt_env_idx = options.get("env_idx", None)
            if opt_env_idx is not None:
                raise ValueError(
                    "Cannot specify both `env_ids` and `options['env_idx']`."
                )
            options["env_idx"] = env_ids

        return self.env.reset(seed=seed, options=options)

    def close(self):
        return self.env.close()

    @property
    def num_envs(self) -> int:
        """The number of environments in the environment."""
        return self.env.num_envs

    @property
    def action_space(self) -> gym.Space:
        """The action space of the environment.

        Returns:
            gym.Space: The action space of the environment.
        """
        return self.env.action_space

    @property
    def observation_space(self) -> gym.Space:
        """The observation space of the environment.

        Returns:
            gym.Space: The observation space of the environment.
        """
        return self.env.observation_space

    def unwrapped_env(self) -> BaseEnv:
        """Get the original ManiSkill environment."""
        return self.env


class ManiSkillEnvCfg(EnvBaseCfg[ManiSkillEnv]):
    """The configuration class for ManiSkill environments.

    All fields in this class will be used to initialize the
    ManiSkill environment.
    """

    class_type: type[ManiSkillEnv] = ManiSkillEnv

    env_id: str
    """The unique identifier for the environment such as 'PickCube-v1'."""

    num_envs: int = 1
    obs_mode: Optional[str] = None
    reward_mode: Optional[str] = None
    control_mode: Optional[str] = None
    render_mode: Optional[str] = None
    shader_dir: Optional[str] = None
    enable_shadow: bool = False
    sensor_configs: Optional[dict] = dict()
    human_render_camera_configs: Optional[dict] = dict()
    viewer_camera_configs: Optional[dict] = dict()
    robot_uids: str | BaseAgent | list[str | BaseAgent] | None = None
    sim_config: SimConfig | dict = dict()
    reconfiguration_freq: Optional[int] = None
    sim_backend: str = "auto"
    render_backend: str = "gpu"
    parallel_in_single_scene: bool = False
    enhanced_determinism: bool = False

    def make_env(self) -> BaseEnv:
        """Create the ManiSkill environment."""
        cfg = self.to_dict(include_config_type=False, exclude_defaults=True)
        cfg.pop("class_type", None)
        cfg.pop("env_id", None)

        return gym.make(id=self.env_id, **cfg)  # type: ignore
