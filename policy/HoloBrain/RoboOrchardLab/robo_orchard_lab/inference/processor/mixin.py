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
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import deprecated

from robo_orchard_lab.processing.io_processor.base import (
    ClassType_co,
    ModelIOProcessor,
    ModelIOProcessorCfg,
)

if TYPE_CHECKING:
    from robo_orchard_lab.inference.processor.compose import (
        ComposeProcessor,
        ComposeProcessorCfg,
    )


__all__ = [
    "ClassType_co",
    "ProcessorMixin",
    "ProcessorMixinType_co",
    "ProcessorMixinCfg",
    "ProcessorMixinCfgType_co",
]


@deprecated(
    "Use `robo_orchard_lab.processing.io_processor.ModelIOProcessor` instead.",
    category=None,
)
class ProcessorMixin(ModelIOProcessor):
    """Backward-compatible facade for the historical data processor base class.

    This deprecated class preserves the public semantics of
    ``robo_orchard_lab.inference.processor.mixin.ProcessorMixin`` while
    delegating the canonical implementation to
    :class:`robo_orchard_lab.processing.io_processor.ModelIOProcessor`.

    A processor still encapsulates the pre-processing and post-processing logic
    required around a model. Subclasses implement ``pre_process`` to convert
    raw inputs into model-ready data and may override ``post_process`` to turn
    raw model outputs into a user-friendly representation.
    """

    def __add__(self, other: ModelIOProcessor | ComposeProcessor):
        """Build a legacy composed processor by appending ``other``.

        This preserves the historical ``ProcessorMixin`` composition behavior
        and returns a ``ComposeProcessor`` facade rather than the canonical
        ``ComposedIOProcessor`` type.

        Args:
            other (ModelIOProcessor | ComposeProcessor): Processor to append.

        Returns:
            ComposeProcessor: A new legacy composed processor instance.
        """
        if not isinstance(other, ModelIOProcessor):
            raise TypeError(
                "Can only concatenate processor objects that implement "
                "ModelIOProcessor."
            )

        from robo_orchard_lab.inference.processor.compose import (
            ComposeProcessor,
            ComposeProcessorCfg,
        )

        new_processor = ComposeProcessor.__new__(ComposeProcessor)
        if isinstance(self, ComposeProcessor):
            new_processor.cfg = self.cfg.model_copy(deep=False)
            new_processor.processors = list(self.processors)
        else:
            new_processor.cfg = ComposeProcessorCfg(processors=[self.cfg])
            new_processor.processors = [self]

        new_processor += other
        return new_processor


ProcessorMixinType_co = TypeVar(
    "ProcessorMixinType_co",
    bound=ProcessorMixin,
    covariant=True,
)


@deprecated(
    "Use `robo_orchard_lab.processing.io_processor.ModelIOProcessorCfg` "
    "instead.",
    category=None,
)
class ProcessorMixinCfg(ModelIOProcessorCfg[ProcessorMixinType_co]):
    """Backward-compatible facade for the historical processor config base.

    This deprecated config preserves the legacy serialized config path for
    processor configurations while delegating the canonical implementation to
    :class:`robo_orchard_lab.processing.io_processor.ModelIOProcessorCfg`.
    """

    def __add__(self, other: ModelIOProcessorCfg | ComposeProcessorCfg):
        """Build a legacy composed processor config by appending ``other``.

        Args:
            other (ModelIOProcessorCfg | ComposeProcessorCfg): Processor
                config to append.

        Returns:
            ComposeProcessorCfg: A new legacy composed processor config.
        """
        if not isinstance(other, ModelIOProcessorCfg):
            raise TypeError(
                "Can only concatenate processor config objects that "
                "implement ModelIOProcessorCfg."
            )

        from robo_orchard_lab.inference.processor.compose import (
            ComposeProcessorCfg,
        )

        if isinstance(self, ComposeProcessorCfg):
            new_cfg = self.model_copy(deep=False)
        else:
            new_cfg = ComposeProcessorCfg(processors=[self])

        new_cfg += other
        return new_cfg


ProcessorMixinCfgType_co = TypeVar(
    "ProcessorMixinCfgType_co",
    bound=ProcessorMixinCfg,
    covariant=True,
)
