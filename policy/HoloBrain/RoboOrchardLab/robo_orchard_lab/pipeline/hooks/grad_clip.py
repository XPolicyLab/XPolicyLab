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
from typing import Literal

import torch

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["GradientClippingHook", "GradientClippingHookConfig"]


class GradientClippingHook(PipelineHooks):
    """A hook for gradient clipping during training.

    This hook is responsible for clipping the gradients of the model
    parameters to prevent exploding gradients. It performs the clipping
    after each step of the training process.

    Note:
        If you are using OptimizerHook and GradientClippingHook together,
        make sure that GradientClippingHook is registered before OptimizerHook
        in the pipeline. This ensures that the gradients are clipped before
        the optimizer step is performed.


    Args:
        cfg (GradientClippingHookConfig): The configuration for the
            GradientClippingHook.
    """

    def __init__(
        self,
        cfg: GradientClippingHookConfig,
    ):
        super().__init__()
        self.clip_mode = cfg.clip_mode
        self.clip_value = cfg.clip_value
        self.max_norm = cfg.max_norm
        self.norm_type = cfg.norm_type

        self.register_hook(
            "on_step",
            HookContext.from_callable(
                after=self._gradient_clipping,
                before=None,
            ),
        )

    def _gradient_clipping(
        self,
        hook_args: PipelineHookArgs,
    ) -> None:
        """Performs gradient clipping.

        Args:
            hook_args (PipelineHookArgs): The workspace for the gradient
                clipping hook. It should contain the following attributes:
                  - accelerator: The Accelerator instance.
                  - optimizer: The optimizer instance.

        """
        if hook_args.optimizer is None:
            raise ValueError("Optimizer is not set in the hook arguments.")

        optimizer = hook_args.optimizer
        accelerator = hook_args.accelerator

        params: list[torch.Tensor] = []
        for param_group in optimizer.param_groups:
            params.extend(param_group["params"])

        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params)
        )
        if len(params) > 0 and accelerator.sync_gradients:
            if self.clip_mode == "value":
                accelerator.clip_grad_value_(params, self.clip_value)
            elif self.clip_mode == "norm":
                accelerator.clip_grad_norm_(
                    params,
                    self.max_norm,
                    self.norm_type,  # type: ignore
                )


class GradientClippingHookConfig(PipelineHooksConfig[GradientClippingHook]):
    """Configuration class for GradientClippingHook."""

    class_type: type[GradientClippingHook] = GradientClippingHook

    clip_mode: Literal["norm", "value"]
    """The mode of gradient clipping.
        - "norm": Clips gradients by norm.
        - "value": Clips gradients by value.
    """
    clip_value: float | None = None
    """ The maximum norm to clip the gradients to. This parameter
    is only used when `clip_mode` is "norm"."""
    max_norm: float | None = None
    """The maximum norm to clip the gradients to. This parameter is only
    used when `clip_mode` is "norm"."""
    norm_type: float = 2.0
    """The type of norm to use for clipping. Default is 2.0 (L2 norm)."""

    def __post_init__(self) -> None:
        """Post-initialization method to validate the configuration."""
        if self.clip_mode == "value":
            if self.clip_value is None:
                raise ValueError(
                    "clip_value must be specified when clip_mode is 'value'."
                )
            if self.clip_value < 0:
                raise ValueError(
                    "clip_value must be non-negative when clip_mode "
                    "is 'value'."
                )
        elif self.clip_mode == "norm":
            if self.max_norm is None:
                raise ValueError(
                    "max_norm must be specified when clip_mode is 'norm'."
                )
            if self.max_norm < 0:
                raise ValueError(
                    "max_norm must be non-negative when clip_mode is 'norm'."
                )
