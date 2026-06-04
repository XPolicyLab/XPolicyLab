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

import logging
import weakref
from inspect import signature
from typing import Any, Iterable, Literal, Optional

import deprecated
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from robo_orchard_lab.pipeline.hooks.mixin import (
    PipelineHooks,
)
from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    GradientClippingHookConfig as GradClipConfig,
    HookBasedTrainer,
    PipelineHookOrConfigType,
    ResumeCheckpointConfig,
    ValidationHookConfig as ValidationConfig,
)
from robo_orchard_lab.processing.step_processor import BatchStepProcessorMixin

__all__ = ["SimpleTrainer"]


logger = logging.getLogger(__name__)


@deprecated.deprecated(
    reason="This class is deprecated. Use `HookBasedTrainer` instead.",
    version="0.2.0",
)
class SimpleTrainer(HookBasedTrainer):
    """A base trainer class that extends SimpleTrainer for training models.

    This trainer integrates with the `Accelerate` library for distributed
    training, supports custom batch processors, and provides hooks for
    monitoring and extending the training process.

    Args:
            model (torch.nn.Module): The model to be trained.
            accelerator (Accelerator): The `Accelerator` instance managing
                    distributed training.
            batch_processor (BatchStepProcessorMixin): A processor that
                    defines how to handle each batch during training.
            dataloader (Optional[DataLoader]): The data loader for feeding
                batches to the model.
            optimizer (Optional[torch.optim.Optimizer]): The optimizer used
                    for training.
            lr_scheduler (Optional[torch.optim.lr_scheduler.LRScheduler]):
                    The learning rate scheduler.
            lr_scheduler_step_at (str): Whether the learning rate scheduler
                    steps at "epoch" or "step".
            max_step (Optional[int]): The maximum number of steps to train.
            max_epoch (Optional[int]): The maximum number of epochs to train.
            val_dataloader (Optional[DataLoader]): The data loader for
                validation.
            metric (Optional[Any]): The metric used for evaluation, with
                update, compute and reset functions.
            step_eval_freq (Optional[int]): The frequency of evaluation in
                    terms of steps.
            epoch_eval_freq (Optional[int]): The frequency of evaluation in
                    terms of epochs.
            resume_from (Optional[str]): The path or URL to resume training
                from.
            resume_share_dir (Optional[str]): The directory to save resume
                files.
            grad_clip_mode (Optional[str]): The mode for gradient clipping
                    ("value" or "norm").
            grad_clip_value (Optional[float]): The value for gradient clipping.
            grad_max_norm (Optional[float]): The maximum norm for gradient
                    clipping.
            grad_norm_type (int): The type of norm used for gradient clipping.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Accelerator,
        batch_processor: BatchStepProcessorMixin,
        dataloader: DataLoader | Iterable,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        lr_scheduler_step_at: Literal["step"] = "step",
        max_step: Optional[int] = None,
        max_epoch: Optional[int] = None,
        val_dataloader: Optional[DataLoader | Iterable] = None,
        metric: Any = None,
        step_eval_freq: Optional[int] = None,
        epoch_eval_freq: Optional[int] = None,
        resume_from: Optional[str] = None,
        resume_share_dir: Optional[str] = None,
        grad_clip_mode: Optional[Literal["value", "norm"]] = None,
        grad_clip_value: Optional[float] = None,
        grad_max_norm: Optional[float] = None,
        grad_norm_type: int = 2,
        hooks: PipelineHookOrConfigType
        | Iterable[PipelineHookOrConfigType]
        | None = None,
    ):
        # make sure that lr_scheduler_step_at is always "step" or "epoch"
        assert lr_scheduler_step_at in [
            "step",
        ], "lr_scheduler_step_at is deprecated. It should always be 'step'."

        # convert resume parameters to ResumeCheckpointConfig
        if resume_from is not None:
            resume_cfg = ResumeCheckpointConfig(
                resume_from=resume_from,
                cache_dir=resume_share_dir
                if resume_share_dir
                else "/tmp/resume_from",
            )
        else:
            resume_cfg = None

        # convert grad_clip parameters to GradClipConfig
        if grad_clip_mode is not None:
            grad_clip_cfg = GradClipConfig(
                clip_mode=grad_clip_mode,
                clip_value=grad_clip_value,
                max_norm=grad_max_norm,
                norm_type=grad_norm_type,
            )
        else:
            grad_clip_cfg = None

        # convert validation parameters to ValidationConfig
        if val_dataloader is not None:
            # use weakref.proxy to avoid circular reference
            self_proxy: SimpleTrainer = weakref.proxy(self)  # type: ignore

            def eval_callback(model: torch.nn.Module):
                self_proxy.eval()

            val_cfg = ValidationConfig(
                eval_callback=eval_callback,
                step_eval_freq=step_eval_freq,
                epoch_eval_freq=epoch_eval_freq,
            )
        else:
            val_cfg = None

        super().__init__(
            model=model,
            accelerator=accelerator,
            batch_processor=batch_processor,
            dataloader=dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            max_step=max_step,
            max_epoch=max_epoch,
            hooks=PipelineHooks.from_hooks(hooks),
            resume_from=resume_cfg,
            grad_clip=grad_clip_cfg,
            validation=val_cfg,
        )
        # only val_dataloader and metric are required for this class,
        # as the rest of the parameters are used by the parent class!

        self.metric = metric
        if val_dataloader is not None:
            self.val_dataloader = accelerator.prepare(val_dataloader)
        else:
            self.val_dataloader = None

        # self.resume(resume_from)

    @torch.no_grad()
    def eval(self) -> Optional[Any]:
        """Evaluates the model on the validation dataset.

        Returns:
                Optional[Any]: The evaluation metric, or None if evaluation
                        is not performed.
        """
        assert self.val_dataloader is not None and self.metric is not None, (
            "val_dataloader and metric should not be None"
        )
        training = self.model.training
        self.model.eval()
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            logger.info("\n" + "=" * 50 + "BEGIN EVAL" + "=" * 50)
        for val_step_id, batch in enumerate(self.val_dataloader):
            model_outputs = self.model(batch)
            self.metric.update(batch, model_outputs)
            if (
                val_step_id + 1
            ) % 10 == 0 and self.accelerator.is_main_process:
                logger.info(f"eval: {val_step_id + 1}")
        self.accelerator.wait_for_everyone()
        if "accelerator" in signature(self.metric.compute).parameters:
            metric = self.metric.compute(accelerator=self.accelerator)
        else:
            metric = self.metric.compute()
        self.accelerator.wait_for_everyone()
        self.metric.reset()
        torch.cuda.empty_cache()
        self.model.train(training)
        return metric
