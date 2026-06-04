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

from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from typing_extensions import deprecated

from robo_orchard_lab.processing.io_processor.base import ModelIOProcessor
from robo_orchard_lab.processing.step_processor.pipeline_step import (
    DeprecatedError,
    LossNotProvidedError,
    SimpleStepProcessor,
    StepProcessorFromCallable,
)

forward_fn_type = Callable[[Callable, Any], Tuple[Any, Optional[torch.Tensor]]]

_TRANSFORMS_UNSUPPORTED_MESSAGE = (
    "transforms is not supported anymore. "
    "If you want to transform the input batch, "
    "please implement it in the forward function or "
    "in the data pipeline."
)

__all__ = [
    "DeprecatedError",
    "LossNotProvidedError",
    "SimpleBatchProcessor",
    "BatchProcessorFromCallable",
]


@deprecated(
    "Use `robo_orchard_lab.processing.step_processor.pipeline_step."
    "SimpleStepProcessor` instead.",
    category=None,
)
class SimpleBatchProcessor(SimpleStepProcessor):
    """Backward-compatible facade for the historical simple batch processor.

    This deprecated class preserves the old
    ``robo_orchard_lab.pipeline.batch_processor.simple.SimpleBatchProcessor``
    import path while delegating to
    :class:`robo_orchard_lab.processing.step_processor.pipeline_step.SimpleStepProcessor`.

    The underlying implementation still supports the familiar lifecycle of
    batch execution: optional pre-processing via an I/O processor, forward
    execution, optional post-processing, loss reduction, and optional backward
    propagation.
    """

    def __init__(
        self,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
        *,
        io_processor: ModelIOProcessor | None = None,
        apply_post_process: bool = False,
    ) -> None:
        if transforms is not None:
            raise DeprecatedError(_TRANSFORMS_UNSUPPORTED_MESSAGE)

        super().__init__(
            need_backward=need_backward,
            io_processor=io_processor,
            apply_post_process=apply_post_process,
        )

    @staticmethod
    def from_callable(
        forward_fn: forward_fn_type,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
        *,
        io_processor: ModelIOProcessor | None = None,
        apply_post_process: bool = False,
    ) -> "BatchProcessorFromCallable":
        return BatchProcessorFromCallable(
            forward_fn=forward_fn,
            need_backward=need_backward,
            transforms=transforms,
            io_processor=io_processor,
            apply_post_process=apply_post_process,
        )


@deprecated(
    "Use `robo_orchard_lab.processing.step_processor.pipeline_step."
    "StepProcessorFromCallable` instead.",
    category=None,
)
class BatchProcessorFromCallable(StepProcessorFromCallable):
    """Backward-compatible facade for the callable-backed batch processor.

    This deprecated class preserves the old
    ``BatchProcessorFromCallable`` import path while delegating to
    :class:`robo_orchard_lab.processing.step_processor.pipeline_step.StepProcessorFromCallable`.
    It remains the lightweight adapter for using a plain
    ``(model, batch) -> (outputs, loss)`` callable as a batch processor.
    """

    def __init__(
        self,
        forward_fn: forward_fn_type,
        need_backward: bool = True,
        transforms: Optional[Callable | Sequence[Callable]] = None,
        *,
        io_processor: ModelIOProcessor | None = None,
        apply_post_process: bool = False,
    ) -> None:
        super().__init__(
            forward_fn=forward_fn,
            need_backward=need_backward,
            io_processor=io_processor,
            apply_post_process=apply_post_process,
        )
        if transforms is not None:
            raise DeprecatedError(_TRANSFORMS_UNSUPPORTED_MESSAGE)
