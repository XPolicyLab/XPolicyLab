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
from typing import Generic, TypeAlias, TypeVar

from typing_extensions import deprecated

from robo_orchard_lab.pipeline.inference.basic import (
    DatasetType as _DatasetType,
    InferencePipeline as _InferencePipeline,
    InferencePipelineCfg as _InferencePipelineCfg,
)
from robo_orchard_lab.pipeline.inference.mixin import (
    ClassType_co,
    InputType,
    OutputType,
)

__all__ = [
    "DatasetType",
    "InferencePipeline",
    "InferencePipelineCfg",
    "InferencePipelineType_co",
]

DatasetType: TypeAlias = _DatasetType

InferencePipelineType_co = TypeVar(
    "InferencePipelineType_co",
    bound=_InferencePipeline,
    covariant=True,
)


@deprecated(
    "Use `robo_orchard_lab.pipeline.inference.basic.InferencePipeline` "
    "instead.",
    category=None,
)
class InferencePipeline(
    _InferencePipeline[InputType, OutputType], Generic[InputType, OutputType]
):
    """Backward-compatible facade for the historical inference pipeline.

    This deprecated class preserves the original
    ``robo_orchard_lab.inference.basic.InferencePipeline`` import path while
    delegating the implementation to
    :class:`robo_orchard_lab.pipeline.inference.basic.InferencePipeline`.

    The underlying implementation still provides the same high-level runtime
    workflow:

    1. Pre-process raw input data using the configured processor.
    2. Collate processed data into a mini-batch when batching is needed.
    3. Perform model inference.
    4. Post-process the model outputs.

    New code should import the canonical class from
    ``robo_orchard_lab.pipeline.inference.basic``.
    """

    pass


@deprecated(
    "Use `robo_orchard_lab.pipeline.inference.basic.InferencePipelineCfg` "
    "instead.",
    category=None,
)
class InferencePipelineCfg(_InferencePipelineCfg[InferencePipelineType_co]):
    """Backward-compatible config for :class:`InferencePipeline`.

    This deprecated facade preserves legacy serialized config paths while
    reusing the canonical config implementation from
    ``robo_orchard_lab.pipeline.inference.basic``.

    Attributes:
        processor: Optional legacy model I/O processor config.
        collate_fn: Optional callable or collator config used to combine
            dataset-like inputs into mini-batches.
        batch_size: Number of samples to process in each inference mini-batch.
    """

    class_type: ClassType_co[InferencePipelineType_co] = InferencePipeline  # type: ignore
