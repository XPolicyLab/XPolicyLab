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

from typing_extensions import deprecated

from robo_orchard_lab.processing.step_processor import BatchStepProcessorMixin

__all__ = ["BatchProcessorMixin"]


@deprecated(
    "Use `robo_orchard_lab.processing.step_processor."
    "BatchStepProcessorMixin` instead.",
    category=None,
)
class BatchProcessorMixin(BatchStepProcessorMixin):
    """Backward-compatible facade for the historical batch processor mixin.

    This deprecated class preserves the legacy
    ``robo_orchard_lab.pipeline.batch_processor.mixin.BatchProcessorMixin``
    import path while delegating to
    :class:`robo_orchard_lab.processing.step_processor.BatchStepProcessorMixin`.

    The interface still represents one batch step that receives pipeline
    hooks, mutable batch hook arguments, and a model callable, then publishes
    outputs and reduced loss back into the hook workspace.
    """

    pass
