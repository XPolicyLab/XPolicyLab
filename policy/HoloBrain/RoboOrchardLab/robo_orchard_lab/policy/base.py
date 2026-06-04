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
import copy
from typing import Any, TypeVar, cast

import gymnasium as gym
import torch
from robo_orchard_core.policy.base import (
    ACTType,
    OBSType,
    PolicyConfig as _PolicyConfig,
    PolicyMixin as _PolicyMixin,
)
from robo_orchard_core.utils.config import ClassType_co, ConfigInstanceOf

from robo_orchard_lab.pipeline.inference.mixin import (
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
)
from robo_orchard_lab.utils.state import (
    State,
    StateSaveLoadMixin,
)

__all__ = [
    "InferencePipelinePolicy",
    "InferencePipelinePolicyCfg",
    "PolicyConfig",
    "PolicyMixin",
]


class PolicyMixin(StateSaveLoadMixin, _PolicyMixin[OBSType, ACTType]):
    """Base policy contract with canonical State-based recovery support."""

    def _get_state(self) -> State:
        """Get the state of the object for saving."""
        # pull out cfg from state for better clarity
        ret = super()._get_state()
        ret.config = cast(Any, ret.state.pop("cfg", None))
        return ret

    def _set_state(self, state: State) -> None:
        """Set the state of the object from the unpickled state."""
        # push cfg back to state for consistency
        state.state["cfg"] = state.config
        state.config = None
        super()._set_state(state)

    def to(self, device: torch.device | str):
        """Move the policy to the specified device.

        Args:
            device (torch.device | str): The target device.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.to() must be implemented by concrete "
            "policy classes."
        )


PolicyType = TypeVar("PolicyType", bound=PolicyMixin, covariant=True)


class PolicyConfig(_PolicyConfig[PolicyType]):
    """Configuration for PolicyMixin."""

    def __call__(self, *args, **kwargs) -> PolicyType:
        return self.create_instance_by_cfg(*args, **kwargs)


class InferencePipelinePolicy(PolicyMixin[OBSType, ACTType]):
    """A policy that uses an inference pipeline to generate actions.

    This adapter keeps both persistence surfaces exposed by the wrapped
    pipeline stack:

    - :class:`StateSaveLoadMixin` ``save`` / ``load`` snapshot the runtime
      object state of the policy and its nested pipeline.
    - ``pipeline.save_pipeline`` / ``pipeline.load_pipeline`` manage exported
      inference artifacts such as model weights and config files.

    Args:
        cfg (InferencePipelinePolicyCfg): The configuration for the policy.
        observation_space (gym.Space | None, optional): The observation space
            of the environment. Defaults to None.
        action_space (gym.Space | None, optional): The action space of
            the environment. Defaults to None.
        pipeline (robo_orchard_lab.pipeline.inference.mixin.
            InferencePipelineMixin | None, optional): The inference pipeline
            to use. If None, it will be created from the configuration. If
            provided, Defaults to None.
    """

    cfg: InferencePipelinePolicyCfg

    pipeline: InferencePipelineMixin[OBSType, ACTType]

    def __init__(
        self,
        cfg: InferencePipelinePolicyCfg,
        observation_space: gym.Space[OBSType] | None = None,
        action_space: gym.Space[ACTType] | None = None,
        pipeline: InferencePipelineMixin[OBSType, ACTType] | None = None,
    ):
        if [cfg.pipeline_cfg, pipeline].count(None) != 1:
            raise ValueError(
                "Either pipeline_cfg in cfg or pipeline must be provided.",
            )

        if pipeline is None:
            assert cfg.pipeline_cfg is not None
            pipeline = cfg.pipeline_cfg()
        else:
            if cfg.pipeline_cfg is None:
                cfg = copy.deepcopy(cfg)
                cfg.pipeline_cfg = pipeline.cfg

        assert pipeline is not None

        super().__init__(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
        )

        self._setup(
            cfg,
            observation_space=observation_space,
            action_space=action_space,
            pipeline=pipeline,
        )

    def _setup(
        self,
        cfg: InferencePipelinePolicyCfg,
        observation_space: gym.Space[OBSType] | None,
        action_space: gym.Space[ACTType] | None,
        pipeline: InferencePipelineMixin[OBSType, ACTType],
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        if (
            pipeline.cfg is not cfg.pipeline_cfg
            and pipeline.cfg != cfg.pipeline_cfg
        ):
            raise ValueError(
                "The pipeline's cfg does not match the policy's pipeline_cfg. "
                f"Got pipeline.cfg: {pipeline.cfg}, "
                f"policy.pipeline_cfg: {cfg.pipeline_cfg}",
            )
        self.cfg = cfg
        self.pipeline = pipeline

    def _set_state(self, state: State) -> None:
        super()._set_state(state)
        self._setup(
            cfg=self.cfg,
            observation_space=self.observation_space,
            action_space=self.action_space,
            pipeline=self.pipeline,
        )

    def act(self, obs: OBSType) -> ACTType:
        """Generate an action based on the observation.

        Args:
            obs (OBSType): The observation from the environment.

        Returns:
            ACTType: The action to be taken in the environment.
        """

        action = self.pipeline(obs)
        return action

    def reset(self, **kwargs) -> None:
        """Reset policy runtime state and forward pipeline reset kwargs.

        Args:
            kwargs: Keyword arguments consumed by the wrapped pipeline reset
                hook or by concrete subclasses that extend this adapter.
        """
        self.pipeline.reset(**kwargs)

    def to(self, device: torch.device | str):
        """Move the wrapped pipeline to the specified device.

        Args:
            device (torch.device | str): The target device.
        """
        self.pipeline.to(device)

    @property
    def device(self) -> torch.device:
        """The device where the pipeline's parameters are located."""
        return self.pipeline.device


class InferencePipelinePolicyCfg(PolicyConfig[InferencePipelinePolicy]):
    class_type: ClassType_co[InferencePipelinePolicy] = InferencePipelinePolicy

    pipeline_cfg: ConfigInstanceOf[InferencePipelineMixinCfg[Any]] | None = (
        None
    )
    """Configuration for the inference pipeline. If None, it must be provided
    when creating the policy."""
