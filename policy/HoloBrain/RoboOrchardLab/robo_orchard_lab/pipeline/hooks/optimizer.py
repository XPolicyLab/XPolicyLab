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

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["OptimizerHook", "OptimizerHookConfig"]


class OptimizerHook(PipelineHooks):
    """A hook for optimizing the model during training.

    This hook is responsible for performing the optimization step
    and updating the learning rate scheduler. It performs the
    updating after each step of the training process.

    """

    def __init__(self, cfg: OptimizerHookConfig | None):
        super().__init__()
        self.register_hook(
            "on_step",
            HookContext.from_callable(
                after=self._optimizer_step,
                before=None,
            ),
        )

    def _optimizer_step(
        self,
        hook_args: PipelineHookArgs,
    ) -> None:
        """Performs an optimization step.

        Args:
            hook_args (PipelineHookArgs): The workspace for the optimizer
                hook. It should contain the following attributes:
                  - accelerator: The Accelerator instance.
                  - optimizer: The optimizer instance.
                  - lr_scheduler: The learning rate scheduler instance.

        """

        if hook_args.optimizer is None:
            raise ValueError("Optimizer is not set in the hook arguments.")
        if hook_args.lr_scheduler is None:
            raise ValueError(
                "Learning rate scheduler is not set in the hook arguments."
            )
        # Perform the optimization step
        hook_args.optimizer.step()
        hook_args.lr_scheduler.step()
        hook_args.optimizer.zero_grad()


class OptimizerHookConfig(PipelineHooksConfig[OptimizerHook]):
    class_type: type[OptimizerHook] = OptimizerHook
