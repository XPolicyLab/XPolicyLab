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

from typing_extensions import deprecated

from robo_orchard_lab.processing.io_processor.base import (
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.processing.io_processor.compose import (
    ComposedIOProcessor,
    ComposedIOProcessorCfg,
)

__all__ = [
    "ComposeProcessor",
    "ComposeProcessorCfg",
]


@deprecated(
    "Use `robo_orchard_lab.processing.io_processor.compose."
    "ComposedIOProcessor` instead.",
    category=None,
)
class ComposeProcessor(ComposedIOProcessor):
    """Backward-compatible facade for the historical composed processor.

    This deprecated class preserves the legacy
    ``robo_orchard_lab.inference.processor.ComposeProcessor`` import path while
    delegating behavior to
    :class:`robo_orchard_lab.processing.io_processor.compose.ComposedIOProcessor`.

    It still represents an ordered processor chain whose ``pre_process``
    methods run from left to right and whose ``post_process`` methods run in
    reverse order.
    """

    def __add__(
        self, other: ModelIOProcessor | ComposedIOProcessor
    ) -> ComposeProcessor:
        """Build a legacy composed processor by appending ``other``.

        This preserves the historical ``ComposeProcessor`` return type while
        keeping the runtime behavior aligned with the legacy processor chain.

        Args:
            other (ModelIOProcessor | ComposedIOProcessor): Processor to
                append to the legacy chain.

        Returns:
            ComposeProcessor: A new legacy composed processor instance.
        """
        if not isinstance(other, ModelIOProcessor):
            raise TypeError(
                "Can only concatenate processor objects that implement "
                "ModelIOProcessor."
            )

        new_processor = ComposeProcessor.__new__(ComposeProcessor)
        new_processor.cfg = self.cfg.model_copy(deep=False)
        new_processor.processors = list(self.processors)
        ComposedIOProcessor.__iadd__(new_processor, other)
        return new_processor


@deprecated(
    "Use `robo_orchard_lab.processing.io_processor.compose."
    "ComposedIOProcessorCfg` instead.",
    category=None,
)
class ComposeProcessorCfg(ComposedIOProcessorCfg):
    """Backward-compatible config for :class:`ComposeProcessor`.

    This deprecated config preserves legacy serialized config paths for
    ordered processor chains and remains compatible with the historical
    ``processors`` field semantics.
    """

    class_type: type[ComposeProcessor] = ComposeProcessor

    def __add__(
        self, other: ModelIOProcessorCfg | ComposedIOProcessorCfg
    ) -> ComposeProcessorCfg:
        """Build a legacy composed processor config by appending ``other``.

        Args:
            other (ModelIOProcessorCfg | ComposedIOProcessorCfg): Processor
                config to append to the legacy chain.

        Returns:
            ComposeProcessorCfg: A new legacy composed processor config.
        """
        if not isinstance(other, ModelIOProcessorCfg):
            raise TypeError(
                "Can only concatenate processor config objects that "
                "implement ModelIOProcessorCfg."
            )

        new_cfg = self.model_copy(deep=False)
        ComposedIOProcessorCfg.__iadd__(new_cfg, other)
        return new_cfg
