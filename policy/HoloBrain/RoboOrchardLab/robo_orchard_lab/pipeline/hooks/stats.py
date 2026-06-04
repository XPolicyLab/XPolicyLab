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
import datetime
import logging
import time
from collections import deque
from typing import Iterable, Optional

from accelerate import Accelerator
from accelerate.data_loader import (
    DataLoaderDispatcher,
    DataLoaderShard,
    IterableDatasetShard,
)
from torch.utils.data import DataLoader as TorchDataLoader

from robo_orchard_lab.dataset.robot.dataset_ex import IterableDatasetMixin
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)

__all__ = ["StatsMonitor", "StatsMonitorConfig"]


logger = logging.getLogger(__name__)


def _get_dataset_batch_size(dataset: object) -> int | None:
    if isinstance(dataset, IterableDatasetMixin):
        if dataset.batch_loader_kwargs is None:
            return None
        return dataset.batch_loader_kwargs.batch_size

    if isinstance(dataset, IterableDatasetShard):
        inner_dataset = dataset.dataset
        if isinstance(inner_dataset, IterableDatasetShard):
            raise ValueError(
                "Detected a dataloader that was prepared more than once. "
                "Call accelerator.prepare(...) only once for each "
                "dataloader."
            )
        return _get_dataset_batch_size(inner_dataset)

    return None


def _get_dataset_side_batch_size(dataloader: Iterable | None) -> int | None:
    """Infer the per-process batch size from dataset-side batching metadata.

    Some iterable datasets batch samples internally via
    ``batch_loader_kwargs``. In that mode the outer dataloader intentionally
    forwards one already-formed batch at a time, so its exposed
    ``batch_size`` becomes ``1`` and can no longer be treated as the true
    sample batch size.
    """
    if dataloader is None:
        return None

    if isinstance(dataloader, DataLoaderShard):
        return _get_dataset_batch_size(dataloader.dataset)

    if isinstance(dataloader, DataLoaderDispatcher):
        return _get_dataset_batch_size(dataloader.dataset)

    if isinstance(dataloader, TorchDataLoader):
        return _get_dataset_batch_size(dataloader.dataset)

    return None


class StatsMonitor(PipelineHooks):
    """A hook to monitor and log training statistics.

    Including training speed, estimated time remaining, learning rate.
    """

    def __init__(self, cfg: StatsMonitorConfig):
        """Initializes the StatsMonitor hook.

        Args:
            batch_size (Optional[int]): The batch size per process. If None,
                it attempts to be inferred from the dataloader.
            steps_per_epoch (Optional[int]): The number of steps per epoch.
                If None, it attemps to be inferred from the dataloader.
            step_log_freq (int): Frequency to log stats at the step level.
                Logs are output every `step_log_freq` steps.
            epoch_log_freq (int): Frequency to log stats at the epoch level.
                Logs are output every `epoch_log_freq` epochs.
        """
        super().__init__()
        self.batch_size = cfg.batch_size
        self.steps_per_epoch = cfg.steps_per_epoch

        self.step_log_freq = cfg.step_log_freq
        self.epoch_log_freq = cfg.epoch_log_freq

        self._data_stats_estimated = False
        self.total_batch_size = None

        self._start_time = None

        # accumulated step tiems
        self._step_times = deque(maxlen=cfg.window_size)
        self._last_step_end_time = 0.0

        # statistics for epoch callback
        self._epoch_start_time = None
        self._last_epoch_id = 0
        self._epoch_start_step_id = 0

        self.register_hook(
            channel="on_loop",
            hook=HookContext.from_callable(before=self._on_loop_begin),
        )
        self.register_hook(
            channel="on_step",
            hook=HookContext.from_callable(
                before=None, after=self._on_step_end
            ),
        )
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(
                before=self._on_epoch_begin, after=self._on_epoch_end
            ),
        )

    def _estimate_data_stats(
        self,
        accelerator: Accelerator,
        dataloader: Iterable | None,
    ) -> None:
        """Estimates data-related statistics needed for monitoring.

        This method infers essential data statistics such as the batch size per
        process, total batch size across all processes, and the number of steps
        per epoch. These statistics are crucial for accurate logging of
        training speed and estimating the remaining training time.

        Args:
            accelerator (Accelerator): The `Accelerator` instance managing
                distributed training and hardware acceleration.
            dataloader (Iterable | None): The data loader used
                in the training loop.

        Raises:
            ValueError: If the batch size cannot be inferred from the
                dataloader and was not provided during initialization.

        Notes:
            - **Batch Size per Process**:
                - If `self.batch_size` is not provided during initialization,
                it attempts to infer it from `dataloader.batch_size`.
                - Raises a `ValueError` if the batch size cannot be inferred.
            - **Total Batch Size**:
                - Calculated as `self.batch_size * accelerator.num_processes`.
                - Represents the total number of samples processed in parallel
                across all devices.
            - **Steps per Epoch**:
                - If `self.steps_per_epoch` is not provided, it attempts to
                infer it from `len(dataloader)` if the dataloader is a
                finite iterable.
                - If the dataloader does not have a length (e.g., it is an
                infinite iterator), `steps_per_epoch` remains `None`, and
                epoch-based estimations might not be possible.
        """
        if self._data_stats_estimated:
            return

        dataset_side_batch_size = _get_dataset_side_batch_size(dataloader)

        if self.batch_size is not None:
            self.total_batch_size = self.batch_size * accelerator.num_processes
        elif dataset_side_batch_size is not None:
            # Dataset-side batching is the source of truth here. The outer
            # dataloader may report ``batch_size=1`` because it only forwards
            # one ready-made batch per iteration.
            self.batch_size = dataset_side_batch_size
            self.total_batch_size = self.batch_size * accelerator.num_processes
        else:
            if isinstance(dataloader, DataLoaderShard):
                self.total_batch_size = dataloader.total_batch_size
                self.batch_size = int(
                    dataloader.total_batch_size / accelerator.num_processes
                )
            elif isinstance(dataloader, DataLoaderDispatcher):
                self.batch_size = dataloader.batch_size
                self.total_batch_size = (
                    self.batch_size * accelerator.num_processes
                )
            elif isinstance(dataloader, TorchDataLoader):
                self.batch_size = dataloader.batch_size
                self.total_batch_size = (
                    self.batch_size * accelerator.num_processes
                )
            else:
                raise ValueError(
                    "Cannot infer batch size from dataloader. Please provide "
                    "the batch size when initializing StatsMonitor."
                )

        if self.batch_size is None:
            raise ValueError(
                "cannot estimate batch_size from dataloader, please provide it"
            )

        if self.steps_per_epoch is None:
            try:
                self.steps_per_epoch = len(dataloader)  # type: ignore
            except TypeError:
                if accelerator.is_main_process:
                    logger.warning(
                        "Cannot estimate 'steps_per_epoch' before the first "
                        "epoch is completed when 'max_epoch' is specified. "
                        "As a result, the estimation of remaining training "
                        "time will not be possible until after the first "
                        "epoch. Please ensure that 'steps_per_epoch' is "
                        "provided or the dataloader implements '__len__'."
                    )

        self._data_stats_estimated = True

    def _estimate_remaining_time(
        self,
        avg_step_time: float,
        current_step: int,
        current_epoch: int,
        start_step: int,
        start_epoch: int,
        max_step: Optional[int],
        max_epoch: Optional[int],
        steps_per_epoch: Optional[int],
    ) -> Optional[float]:
        """Estimates the remaining time based on average step time.

        Args:
            avg_step_time (float): The average time per step, calculated from
                a smoothing window.
            current_step (int): The current global step in the
                training loop.
            current_epoch (int): The current epoch in the training loop.
            start_step (int): The starting global step in the training loop.
            start_epoch (int): The starting epoch in the training loop.
            max_step (Optional[int]): The maximum number of global
                steps in the training loop.
            max_epoch (Optional[int]): The maximum number of epochs
                in the training loop.
            steps_per_epoch (Optional[int]): The number of steps per
                epoch, if known.

        Returns:
            Optional[float]: Estimated remaining time in seconds, or None
            if estimation isn't possible.

        Raises:
            ValueError: If both `max_step` and `max_epoch` are None.
        """
        if max_step is None and max_epoch is None:
            raise ValueError(
                "At least one of `max_step` or `max_epoch` must be specified."
            )

        # Cannot estimate without steps_per_epoch if using max_epoch
        if max_epoch is not None and steps_per_epoch is None:
            return None

        total_steps = (
            float("inf") if max_step is None else max_step - start_step
        )

        if max_epoch is not None:
            total_steps = min(
                steps_per_epoch * (max_epoch - start_epoch) - start_step,  # type: ignore
                total_steps,
            )

        remain_steps = total_steps - current_step - 1
        if remain_steps <= 0:
            return 0.0

        estimated_time = remain_steps * avg_step_time
        return estimated_time

    def _on_loop_begin(self, args: PipelineHookArgs) -> None:
        """Initializes timing and total batch size at the start of training.

        Args:
            args (PipelineHookArgs): Hook arguments including accelerator
            and dataloader.
        """
        current_time = time.time()
        self._start_time = current_time
        self._estimate_data_stats(args.accelerator, args.dataloader)
        self._last_step_end_time = time.time()

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        """Callback when step ends.

        Logs the training speed and estimated remaining time at the end of
        each step.

        Args:
            args (PipelineHookArgs): Hook arguments including current step
                and epoch information.
        """
        assert (
            self._start_time is not None and self.total_batch_size is not None
        ), "Please call `on_loop_begin` first."

        # only log in the main process
        if not args.accelerator.is_main_process:
            return

        current_step_end_time = time.time()
        step_duration = current_step_end_time - self._last_step_end_time
        self._last_step_end_time = current_step_end_time
        self._step_times.append(step_duration)

        if (
            self.step_log_freq > 0
            and (args.global_step_id + 1) % self.step_log_freq == 0
        ):
            # logging training speed and estimated remaining time
            avg_step_time = sum(self._step_times) / len(self._step_times)
            speed = self.total_batch_size / avg_step_time

            remaining_time = self._estimate_remaining_time(
                avg_step_time=avg_step_time,
                current_step=args.global_step_id,
                current_epoch=args.epoch_id,
                start_step=args.start_step,
                start_epoch=args.start_epoch,
                max_step=args.max_step,
                max_epoch=args.max_epoch,
                steps_per_epoch=self.steps_per_epoch,
            )

            if remaining_time is not None:
                remaining_time_str = str(
                    datetime.timedelta(seconds=int(remaining_time))
                )
            else:
                remaining_time_str = "N/A"

            # reset states
            self._last_step_id = args.global_step_id
            self._step_start_time = time.time()

            msg = f"Epoch[{args.epoch_id}] Step[{args.step_id}] GlobalStep[{args.global_step_id}] "  # noqa: E501
            msg += f"Training Speed: {speed:.2f} samples/sec across all devices.\t"  # noqa: E501
            msg += f"Average Step Time: {avg_step_time:.2f} sec.\t"
            msg += f"Estimated Remaining Time: {remaining_time_str}.\t"

            if args.optimizer is not None:
                for idx, param in enumerate(args.optimizer.param_groups):
                    msg += "Learning Rate Group {}: {:.5e}.\t".format(
                        idx, param["lr"]
                    )
                    args.accelerator.log(
                        {f"LR/group{idx}": param["lr"]},
                        step=args.global_step_id,
                    )

            logger.info(msg)

    def _on_epoch_begin(self, args: PipelineHookArgs) -> None:
        if self._epoch_start_time is None:
            self._epoch_start_time = time.time()
            self._last_epoch_id = args.epoch_id
            self._epoch_start_step_id = args.global_step_id

    def _on_epoch_end(self, args: PipelineHookArgs) -> None:
        """Logs the average epoch time and resets the epoch start time.

        Args:
            args (PipelineHookArgs): Hook arguments including current epoch
            information.
        """
        assert self._start_time is not None, (
            "Please call `on_loop_begin` first."
        )
        assert (
            self._epoch_start_time is not None
            and self.total_batch_size is not None
        ), "Please call `on_epoch_begin` first."

        # only log in the main process
        if not args.accelerator.is_main_process:
            return

        # estimate steps_per_epoch from last epoch
        if self.steps_per_epoch is None:
            self.steps_per_epoch = args.start_step + args.step_id + 1

        if (
            self.epoch_log_freq > 0
            and (args.epoch_id + 1) % self.epoch_log_freq == 0
        ):
            epoch_duration = time.time() - self._epoch_start_time
            elapsed_epochs = args.epoch_id - self._last_epoch_id + 1
            avg_epoch_time = epoch_duration / elapsed_epochs
            elapsed_steps = args.global_step_id - self._epoch_start_step_id

            if elapsed_steps > 0:
                avg_step_time = epoch_duration / elapsed_steps
                speed = self.total_batch_size / avg_step_time
                speed_str = f"{speed:.2f}"
                avg_step_time_str = f"{avg_step_time:.2f} sec."
            else:
                avg_step_time = None
                speed_str = "N/A"
                avg_step_time_str = "N/A"

            if len(self._step_times) > 0:
                smooth_avg_step_time = sum(self._step_times) / len(
                    self._step_times
                )
                remaining_time = self._estimate_remaining_time(
                    avg_step_time=smooth_avg_step_time,
                    current_step=args.global_step_id - 1,
                    current_epoch=args.epoch_id,
                    start_step=args.start_step,
                    start_epoch=args.start_epoch,
                    max_step=args.max_step,
                    max_epoch=args.max_epoch,
                    steps_per_epoch=self.steps_per_epoch,
                )
            else:
                remaining_time = None

            if remaining_time is not None:
                remaining_time_str = str(
                    datetime.timedelta(seconds=int(remaining_time))
                )
            else:
                remaining_time_str = "N/A"

            # reset states for next epoch log
            self._epoch_start_time = time.time()
            self._last_epoch_id = args.epoch_id
            self._epoch_start_step_id = args.global_step_id

            msg = f"Epoch[{args.epoch_id}] completed. "
            msg += f"Epoch Time: {epoch_duration:.2f} sec.\t"
            msg += f"Average Training Speed: {speed_str} samples/sec across all devices.\t"  # noqa: E501
            msg += f"Average Epoch Time: {avg_epoch_time:.2f} sec.\t"
            msg += f"Average Step Time: {avg_step_time_str}\t"
            msg += f"Estimated Remaining Time: {remaining_time_str}.\t"
            if args.optimizer is not None:
                for idx, param in enumerate(args.optimizer.param_groups):
                    msg += "Learning Rate Group {}: {:.5e}.\t".format(
                        idx, param["lr"]
                    )
            logger.info(msg)


class StatsMonitorConfig(PipelineHooksConfig[StatsMonitor]):
    """Configuration class for StatsMonitor."""

    class_type: type[StatsMonitor] = StatsMonitor

    batch_size: Optional[int] = None
    """The batch size per process. If None, it attempts to be inferred
    from the dataloader."""

    steps_per_epoch: Optional[int] = None
    """The number of steps per epoch. If None, it attempts to be inferred
    from the dataloader."""

    step_log_freq: int = 512
    """Frequency to log stats at the step level. Logs are output every
    `step_log_freq` steps."""

    epoch_log_freq: int = 1
    """Frequency to log stats at the epoch level. Logs are output every
    `epoch_log_freq` epochs."""

    window_size: int = 128
    """The size of the sliding window used to smooth the step time
    calculation, making the remaining time estimate more stable."""
