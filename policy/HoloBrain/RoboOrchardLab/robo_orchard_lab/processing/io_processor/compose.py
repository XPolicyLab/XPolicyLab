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

from robo_orchard_lab.processing.io_processor.base import (
    ClassType_co,
    ModelIOProcessor,
    ModelIOProcessorCfg,
    ModelIOProcessorCfgType_co,
)

__all__ = [
    "ComposedIOProcessor",
    "ComposedIOProcessorCfg",
]


class ComposedIOProcessor(ModelIOProcessor):
    """A legacy-compatible composed processor chain.

    This processor acts as a container that applies a sequence of other
    processors serially. It is useful for building complex model I/O pipelines
    from smaller, reusable components. ``pre_process`` is applied from left to
    right, while ``post_process`` is applied in reverse order so the output
    transformation mirrors the input transformation stack.

    This compose family is still supported for compatibility, but new code
    should prefer :class:`ComposedEnvelopeIOProcessor` via
    ``compose_envelope(...)`` or envelope-family ``+`` composition.

    The composed processor preserves the standard processor call semantics:
    child ``pre_process`` methods are fed one sample at a time before
    collation, while child ``post_process`` methods usually receive batched
    model outputs and collated model inputs.
    """

    cfg: "ComposedIOProcessorCfg"

    def __init__(self, cfg: "ComposedIOProcessorCfg"):
        """Instantiate all configured child processors.

        Args:
            cfg (ComposedIOProcessorCfg): Configuration describing the ordered
                child processor chain.
        """
        super().__init__(cfg)
        self.processors: list[ModelIOProcessor] = [
            cfg_i() for cfg_i in self.cfg.processors
        ]

    def __getitem__(self, index: int) -> ModelIOProcessor:
        """Return a child processor by position."""
        return self.processors[index]

    def __iadd__(
        self, other: ModelIOProcessor | ComposedIOProcessor
    ) -> ComposedIOProcessor:
        """Append another processor to this composed processor in place.

        When ``other`` is itself a composed processor, its child processors are
        appended in order so the resulting chain behaves as if both pipelines
        had been concatenated.

        Args:
            other: Another processor or composed processor to append.

        Returns:
            ComposedIOProcessor: The current composed processor instance.
        """
        if not isinstance(other, ModelIOProcessor):
            raise TypeError(
                "Can only concatenate ModelIOProcessor or "
                "ComposedIOProcessor objects."
            )

        if isinstance(other, ComposedIOProcessor):
            self.processors.extend(other.processors)
        else:
            self.processors.append(other)

        self.cfg += other.cfg
        return self

    def pre_process(self, data):
        """Apply ``pre_process`` for each child processor in sequence.

        The output of one processor becomes the input of the next processor in
        the chain. In the default inference flow this input is typically one
        sample before batching.

        Args:
            data: The initial raw input data.

        Returns:
            The data after being transformed by all child processors.
        """
        for processor in self.processors:
            data = processor.pre_process(data)
        return data

    def post_process(self, model_outputs, model_input):
        """Apply ``post_process`` for each child processor in reverse order.

        Running post-processing in reverse order keeps output transformations
        aligned with the inverse of the pre-processing chain. In the default
        inference flow, ``model_outputs`` and ``model_input`` are usually the
        batched values produced after collation and model forward.

        Args:
            model_outputs: Raw model outputs to transform.
            model_input: Model input that may be needed as post-processing
                context.

        Returns:
            The final outputs after all child processors have run.
        """
        for processor in reversed(self.processors):
            model_outputs = processor.post_process(model_outputs, model_input)
        return model_outputs


class ComposedIOProcessorCfg(ModelIOProcessorCfg[ComposedIOProcessor]):
    """Configuration for the legacy-compatible composed processor family.

    New code should prefer :class:`ComposedEnvelopeIOProcessorCfg` and
    ``compose_envelope_cfg(...)`` when defining composed processors directly
    in the envelope family.
    """

    class_type: ClassType_co[ComposedIOProcessor] = ComposedIOProcessor
    processors: list[ModelIOProcessorCfgType_co]  # type: ignore
    """Ordered processor configs to instantiate and apply serially."""

    def __getitem__(self, item):
        """Return a child processor config by position."""
        return self.processors[item]

    def __iadd__(
        self, other: ModelIOProcessorCfg | ComposedIOProcessorCfg
    ) -> ComposedIOProcessorCfg:
        """Append another processor config to this composed config.

        When ``other`` is a composed config, its child configs are appended in
        order so the resulting config matches the equivalent chained runtime
        processor.

        Args:
            other: Another processor config or composed processor config.

        Returns:
            ComposedIOProcessorCfg: The current composed config instance.
        """
        if not isinstance(other, ModelIOProcessorCfg):
            raise TypeError(
                "Can only concatenate ModelIOProcessorCfg objects."
            )

        if isinstance(other, ComposedIOProcessorCfg):
            self.processors = list(self.processors) + list(other.processors)
        else:
            self.processors = list(self.processors) + [other]

        return self
