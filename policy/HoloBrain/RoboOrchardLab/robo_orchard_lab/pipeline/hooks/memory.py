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

from accelerate.utils.memory import clear_device_cache

from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["ClearCacheHook", "ClearCacheHookConfig"]


class ClearCacheHook(PipelineHooks):
    """A Hook to periodically release cached device memory.

    This class is designed to work with training pipelines that use hooks
    for step and epoch management. It integrates with PyTorch's
    `accelerate.utils.memory.clear_device_cache()` to clear unused memory,
    helping to avoid out-of-memory (OOM) errors during
    long-running training loops.

    Args:
        cfg (ClearCacheHookConfig): Configuration for the ClearCacheHook.


    Examples:
        Basic Usage:
            >>> from robo_orchard_lab.pipeline.hooks import (
            ...     PipelineHookArgs,
            ...     ClearCacheHookConfig,
            ... )
            >>>
            >>> memory_manager = ClearCacheHookConfig(
            >>>     empty_cache_at="step", empty_cache_freq=10
            >>> )()
            >>> # Simulate a training step
            >>> hook_args = PipelineHookArgs(global_step_id=9, epoch_id=0)
            >>> with memory_manager.begin("on_step", hook_args) as hook_args:
            >>>     ... # Simulate the end of a step
            # Clears the cache after 10 steps

        Epoch-Based Clearing:
            >>> memory_manager = ClearCacheHookConfig(
            >>>     empty_cache_at="epoch", empty_cache_freq=2
            >>> )()
            >>> # Simulate the end of an epoch
            >>> hook_args = PipelineHookArgs(global_step_id=99, epoch_id=1)
            >>> with memory_manager.begin("on_epoch", hook_args) as hook_args:
            >>>     ... # Simulate the end of an epoch
            # Clears the cache after every 2 epochs
    """

    def __init__(
        self,
        cfg: ClearCacheHookConfig,
    ):
        super().__init__()
        self.empty_cache_at = cfg.empty_cache_at
        self.empty_cache_freq = cfg.empty_cache_freq
        self.garbage_collection = cfg.garbage_collection

        self.register_hook(
            "on_step", hook=HookContext.from_callable(after=self._on_step_end)
        )
        self.register_hook(
            "on_epoch",
            hook=HookContext.from_callable(after=self._on_epoch_end),
        )

    def _on_step_end(self, arg: PipelineHookArgs) -> None:
        """Hook invoked at the end of a training step.

        Clears the CUDA cache if `empty_cache_at` is set to "step" and the
        current step satisfies the clearing frequency (`empty_cache_freq`).

        Args:
            arg (PipelineHookArgs): Arguments passed by the pipeline, including
            `global_step_id`.

        """
        if (
            self.empty_cache_at == "step"
            and (arg.global_step_id + 1) % self.empty_cache_freq == 0
        ):
            clear_device_cache(garbage_collection=self.garbage_collection)

    def _on_epoch_end(self, arg: PipelineHookArgs) -> None:
        """Hook invoked at the end of a training epoch.

        Clears the CUDA cache if `empty_cache_at` is set to "epoch" and the
        current epoch satisfies the clearing frequency (`empty_cache_freq`).

        Args:
            arg (PipelineHookArgs): Arguments passed by the pipeline,
            including `epoch_id`.

        """
        if (
            self.empty_cache_at == "epoch"
            and (arg.epoch_id + 1) % self.empty_cache_freq == 0
        ):
            clear_device_cache(garbage_collection=self.garbage_collection)


class ClearCacheHookConfig(PipelineHooksConfig[ClearCacheHook]):
    """Configuration class for ClearCacheHook."""

    class_type: type[ClearCacheHook] = ClearCacheHook

    empty_cache_at: Literal["step", "epoch"] = "epoch"
    """Specifies whether to clear the cache at the end of each step or
    epoch. Default is "epoch"."""

    empty_cache_freq: int = 1
    """The frequency of cache clearing. For example:
        - If `empty_cache_at="step"`, this clears the cache every
        `empty_cache_freq` steps.
        - If `empty_cache_at="epoch"`, this clears the cache every
        `empty_cache_freq` epochs. Default is 1 (clear after every step
        or epoch).
    """

    garbage_collection: bool = False
    """Whether to perform garbage collection before clearing the cache."""
