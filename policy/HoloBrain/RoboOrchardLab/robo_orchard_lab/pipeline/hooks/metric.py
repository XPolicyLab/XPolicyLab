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
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Iterable,
    Literal,
    Sequence,
    Tuple,
)

from robo_orchard_core.utils.config import ClassType

from robo_orchard_lab.metrics.base import MetricProtocol
from robo_orchard_lab.pipeline.hooks.mixin import (
    HookContext,
    ModelOutput,
    PipelineHookArgs,
    PipelineHooks,
    PipelineHooksConfig,
)
from robo_orchard_lab.utils import as_sequence

__all__ = ["MetricEntry", "MetricTracker", "MetricTrackerConfig"]

logger = logging.getLogger(__name__)


@dataclass
class MetricEntry:
    """A class representing an entry for tracking a metric."""

    names: Sequence[str] | str  # type: ignore
    metric: MetricProtocol

    def __post_init__(self) -> None:
        """Initializes the names attribute as a sequence."""
        self.names: Sequence[str] = as_sequence(self.names)

    def get(self) -> Iterable[Tuple[str, Any]]:
        """Computes and retrieves the metric values along with their names.

        Returns:
            Iterable[Tuple[str, Any]]: A sequence of (name, value) pairs for
                the metric.

        Raises:
            AssertionError: If the number of names does not match the number
                of metric values.
        """
        values = as_sequence(self.metric.compute())
        assert len(self.names) == len(values), (
            "The length of names and metric values should be the same, but get {} vs. {}".format(  # noqa: E501
                len(self.names), len(values)
            )
        )
        return zip(self.names, values, strict=False)


class MetricTracker(PipelineHooks):
    """A hook for updating and logging metrics."""

    def __init__(
        self,
        cfg: MetricTrackerConfig,
    ) -> None:
        """Initialization.

        Args:
            metric_entrys (MetricEntry|Sequence[MetricEntry]): Single
                or multiple metric entries to update.
            reset_by (Literal["epoch", "step"], optional): Basis for resetting
                metrics; either "epoch" or "step".
            reset_freq (int, optional): Frequency to reset metrics.
                Defaults to 1.
            step_log_freq (int, optional): Frequency to log at the step level.
                Defaults to 512.
            epoch_log_freq (int, optional): Frequency to log at the epoch
                level. Defaults to 1.
            log_main_process_only (int, optional): Only logging in the main
                processor or not. Defaults to True.
        """
        super().__init__()
        self.metric_entrys: Sequence[MetricEntry] = as_sequence(
            cfg.metric_entrys
        )
        self.metrics = [entry_i.metric for entry_i in self.metric_entrys]
        self.reset_by = cfg.reset_by
        self.reset_freq = cfg.reset_freq
        self.step_log_freq = cfg.step_log_freq
        self.epoch_log_freq = cfg.epoch_log_freq
        self.log_main_process_only = cfg.log_main_process_only

        self._reset()

        self.register_hook(
            channel="on_loop",
            hook=HookContext.from_callable(before=self._on_loop_begin),
        )
        self.register_hook(
            channel="on_batch",
            hook=HookContext.from_callable(after=self._on_batch_end),
        )
        self.register_hook(
            channel="on_step",
            hook=HookContext.from_callable(after=self._on_step_end),
        )
        self.register_hook(
            channel="on_epoch",
            hook=HookContext.from_callable(after=self._on_epoch_end),
        )

    @abstractmethod
    def update_metric(self, batch: Any, model_outputs: ModelOutput) -> None:
        """Updates the metrics using the current batch data and model outputs.

        This method must be implemented in subclasses to define how the metrics
        in `self.metrics` are updated during training or evaluation.
        The implementation should handle the logic for calling `update`
        on each metric.

        Args:
            batch (Any): The current batch of data from the dataset. It is
                expected to contain all necessary inputs (e.g., features,
                labels) required for metric computation.
            model_outputs (ModelOutput): The outputs produced by the model
                for the given batch. This could include predictions,
                probabilities, logits, or other outputs, depending on the
                model architecture and task.

        Example:
            A typical implementation in a subclass might look like this:

            .. code-block:: python

                class CustomMetricTracker(MetricTracker):
                    def update_metric(self, batch, model_outputs):
                        for metric in self.metrics:
                            metric.update(
                                batch["label"], model_outputs["pred"]
                            )
        """

    def _reset(self) -> None:
        """Resets all metrics to their initial states."""
        for metric_i in self.metrics:
            metric_i.reset()

    def _log(self, prefix: str = "", is_main_process: bool = True) -> None:
        """Logs the computed metric values with an optional prefix."""
        msg = prefix
        for entry_i in self.metric_entrys:
            for k, v in entry_i.get():
                if isinstance(v, (int, float)):
                    msg += "%s[%.4f] " % (k, v)
                else:
                    msg += "%s[%s] " % (str(k), str(v))
        if self.log_main_process_only and not is_main_process:
            return
        logger.info(msg)

    def _on_loop_begin(self, args: PipelineHookArgs) -> None:
        """Callback when loop begins.

        Prepares metrics for computation by moving them to the appropriate
        device.

        Args:
            args (PipelineHookArgs): Arguments containing the accelerator with
            device information.
        """
        for metric_i in self.metrics:
            if hasattr(metric_i, "to"):
                metric_i.to(device=args.accelerator.device)  # type: ignore

    def _on_batch_end(self, args: PipelineHookArgs) -> None:
        """Callback when batch ends.

        Updates metrics using the `metric_update_fn` at the end of each batch.

        Args:
            args (PipelineHookArgs): Arguments containing the batch data and
            model outputs.
        """
        if args.model_outputs is None:
            raise ValueError("Model outputs are required to update metrics.")
        self.update_metric(args.batch, args.model_outputs)

    def _on_step_end(self, args: PipelineHookArgs) -> None:
        """Callback when step ends.

        Logs metrics and optionally resets them at the end of a step based
        on `step_log_freq`.

        Args:
            args (PipelineHookArgs): Arguments containing the current step
                and epoch IDs.
        """

        if (
            self.step_log_freq > 0
            and (args.step_id + 1) % self.step_log_freq == 0
        ):
            prefix = "Epoch[{}] Step[{}] GlobalStep[{}]: ".format(
                args.epoch_id, args.step_id, args.global_step_id
            )
            self._log(
                prefix=prefix, is_main_process=args.accelerator.is_main_process
            )

        if (
            self.reset_by == "step"
            and (args.global_step_id + 1) % self.reset_freq == 0
        ):
            self._reset()

    def _on_epoch_end(self, args: PipelineHookArgs) -> None:
        """Callback when epoch ends.

        Logs metrics and optionally resets them at the end of an epoch
        based on `epoch_log_freq`.

        Args:
            args (PipelineHookArgs): Arguments containing the current epoch ID.
        """

        if (
            self.epoch_log_freq > 0
            and (args.epoch_id + 1) % self.epoch_log_freq == 0
        ):
            prefix = "Epoch[{}]: ".format(args.epoch_id)
            self._log(
                prefix=prefix, is_main_process=args.accelerator.is_main_process
            )

        if (
            self.reset_by == "epoch"
            and (args.epoch_id + 1) % self.reset_freq == 0
        ):
            self._reset()


class MetricTrackerConfig(PipelineHooksConfig[MetricTracker]):
    class_type: ClassType[MetricTracker] = MetricTracker

    metric_entrys: Sequence[MetricEntry] | MetricEntry
    """Single or multiple metric entries to update."""
    reset_by: Literal["epoch", "step"] = "epoch"
    """Frequency basis for metric reset ("epoch" or "step")."""
    reset_freq: int = 1
    """Frequency to reset metrics."""
    step_log_freq: int = 512
    """Frequency of logging at the step level."""
    epoch_log_freq: int = 1
    """Frequency of logging at the epoch level."""
    log_main_process_only: bool = True
    """Only logging in the main processor or not."""
