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
import logging

import torch
from accelerate import Accelerator
from pydantic import Field
from typing_extensions import deprecated

from robo_orchard_lab.pipeline.hooks.mixin import (
    ClassType,
    HookContext,
    ModelOutput,
    ModelOutputHasLossKeys,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = [
    "LossTracker",
    "LossTrackerConfig",
    "LossMovingAverageTrackerConfig",
    "LossMovingAverageTracker",
]


logger = logging.getLogger(__name__)


class LossTracker(PipelineHooks):
    cfg: LossTrackerConfig

    def __init__(self, cfg: LossTrackerConfig):
        super().__init__()
        self.cfg = cfg
        self.reset_cached_loss()

        self.register_hook(
            "on_step", HookContext.from_callable(after=self._on_step_end)
        )

    def reset_cached_loss(self):
        self.cached_loss: dict[str, float] = {}
        self.moving_average_count: dict[str, float] = {}

    def update(self, accelerator: Accelerator, model_outputs: ModelOutput):
        """Track and update the loss values.

        This method only runs on the main process, and all loss values are
        reduced (averaged) across all processes.
        """

        # find loss keys
        if isinstance(model_outputs, ModelOutputHasLossKeys):
            loss_keys = list(model_outputs.loss_keys())
        else:
            loss_keys = [k for k in model_outputs.keys() if "loss" in k]
        for k in loss_keys:
            loss = model_outputs[k]
            assert isinstance(loss, torch.Tensor), f"Loss {k} is not a tensor."

            reduced_loss: torch.Tensor = accelerator.reduce(
                loss, reduction="mean"
            )  # type: ignore
            if reduced_loss.numel() != 1:
                logger.warning(
                    f"The loss {k} is not a scalar, "
                    f"got shape {reduced_loss.shape}. "
                    "Using mean value as the loss."
                )
                reduced_loss = reduced_loss.mean()

            reduced_loss_value = reduced_loss.item()

            if k not in self.cached_loss:
                self.cached_loss[k] = 0.0
                self.moving_average_count[k] = 0
            cnt = float(self.moving_average_count[k])
            self.cached_loss[k] = (
                cnt / (cnt + 1) * self.cached_loss[k]
                + 1 / (cnt + 1) * reduced_loss_value
            )
            if self.cfg.moving_average:
                self.moving_average_count[k] += 1

    def _on_step_end(self, args: PipelineHookArgs):
        """Callback when step ends.

        Logs losses and optionally resets them at the end of a step based
        on `step_log_freq`.

        Args:
            args (PipelineHookArgs): Arguments containing the current step
                and epoch IDs.
        """
        if args.model_outputs is not None:
            self.update(args.accelerator, args.model_outputs)

        if (args.step_id + 1) % self.cfg.step_log_freq == 0:
            # only log in main process
            if args.accelerator.is_main_process:
                msg = "Epoch[{}/{}] Step[{}] GlobalStep[{}/{}]: ".format(
                    args.epoch_id,
                    args.max_epoch - 1 if args.max_epoch is not None else "NA",
                    args.step_id,
                    args.global_step_id,
                    args.max_step - 1 if args.max_step is not None else "NA",
                )
                total_loss = 0
                for k, v in self.cached_loss.items():
                    msg += f"{k}[{v:.4f}]\t"
                    total_loss += v
                    args.accelerator.log(
                        {f"Loss/{k}": v},
                        step=args.global_step_id,
                    )

                if self.cfg.log_total_loss:
                    msg += f"total_loss[{total_loss:.4f}]"
                    args.accelerator.log(
                        {"Loss/Total_Loss": total_loss},
                        step=args.global_step_id,
                    )

                logger.info(msg)
            self.reset_cached_loss()


class LossTrackerConfig(PipelineHooksConfig[LossTracker]):
    class_type: ClassType[LossTracker] = LossTracker

    step_log_freq: int = Field(ge=1, default=5)
    """The frequency (in steps) to log the loss."""

    log_total_loss: bool = False
    """If True, log the total loss as well."""

    moving_average: bool = True
    """If True, compute moving average of the loss.

    The moving average window size is the same as `step_log_freq`.
    """


@deprecated(
    "LossMovingAverageTrackerConfig is deprecated, "
    "please use LossTrackerConfig instead."
)
class LossMovingAverageTrackerConfig(LossTrackerConfig):
    step_log_freq: int = 25
    log_total_loss: bool = True


@deprecated(
    "LossMovingAverageTracker is deprecated, please use LossTracker instead."
)
class LossMovingAverageTracker(LossTracker):
    def __init__(self, cfg: LossMovingAverageTrackerConfig):
        super().__init__(cfg)
