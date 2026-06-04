# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar, cast

from robo_orchard_core.utils.config import ConfigInstanceOf

from robo_orchard_lab.processing.io_processor.base import (
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.processing.io_processor.envelope import (
    ClassType_co,
    EnvelopeIOProcessor,
    EnvelopeIOProcessorCfg,
    PipelineEnvelope,
    normalize_pipeline_envelope,
    resolve_envelope_processor,
    resolve_envelope_processor_cfg,
)

__all__ = [
    "ComposedEnvelopeIOProcessor",
    "ComposedEnvelopeIOProcessorCfg",
    "ProcessorContextStack",
    "compose_envelope",
    "compose_envelope_cfg",
]

ProcessorContextEntryT = TypeVar("ProcessorContextEntryT")


@dataclass
class ProcessorContextStack(Generic[ProcessorContextEntryT]):
    """Structured per-child context stack for one composed envelope path.

    Args:
        processor_context_stack (list[Any]): Context values produced by child
            processors during ``pre_process``, aligned with the compose chain
            in forward order.
    """

    processor_context_stack: list[ProcessorContextEntryT]


class ComposedEnvelopeIOProcessor(EnvelopeIOProcessor):
    """Compose processor for ordered envelope-processor chains.

    Main public APIs are :meth:`from_processors`, :meth:`pre_process`,
    :meth:`post_process`, and :meth:`__iadd__`. Prefer
    :func:`compose_envelope` or :meth:`from_processors` when composing runtime
    processors directly.

    Example:
        ``composed = compose_envelope(proc_a, proc_b)``
        ``envelope = composed.pre_process(PipelineEnvelope(model_input=x))``
    """

    cfg: "ComposedEnvelopeIOProcessorCfg"

    def __init__(self, cfg: "ComposedEnvelopeIOProcessorCfg"):
        """Instantiate all configured child envelope processors."""
        super().__init__(cfg)
        self.processors: list[EnvelopeIOProcessor] = [
            cfg_i() for cfg_i in self.cfg.processors
        ]

    @classmethod
    def from_processors(
        cls,
        processors: Sequence[EnvelopeIOProcessor | None],
    ) -> "ComposedEnvelopeIOProcessor":
        """Build a composed envelope processor from runtime child objects.

        Args:
            processors (list[EnvelopeIOProcessor | None]): Runtime envelope
                processors to chain.

        Returns:
            ComposedEnvelopeIOProcessor: Flattened compose processor built
                from the provided runtime children.
        """

        envelope_processors: list[EnvelopeIOProcessor] = []
        for processor in processors:
            if processor is None:
                continue
            if isinstance(processor, cls):
                envelope_processors.extend(processor.processors)
            else:
                envelope_processors.append(
                    cast(EnvelopeIOProcessor, processor)
                )
        instance = cls.__new__(cls)
        EnvelopeIOProcessor._setup(
            instance,
            ComposedEnvelopeIOProcessorCfg(
                processors=[processor.cfg for processor in envelope_processors]
            ),
        )
        instance.processors = envelope_processors
        return instance

    def __getitem__(self, index: int) -> EnvelopeIOProcessor:
        """Return a child envelope processor by position.

        Args:
            index (int): Child processor index.

        Returns:
            EnvelopeIOProcessor: Child processor at ``index``.
        """

        return self.processors[index]

    def __iadd__(
        self,
        other: EnvelopeIOProcessor | ModelIOProcessor,
    ) -> "ComposedEnvelopeIOProcessor":
        """Append another processor to this envelope compose chain.

        Args:
            other (EnvelopeIOProcessor | ModelIOProcessor):
                Processor to append.

        Returns:
            ComposedEnvelopeIOProcessor: ``self`` after appending ``other``.
        """

        if not isinstance(
            other,
            (
                EnvelopeIOProcessor,
                ModelIOProcessor,
            ),
        ):
            raise TypeError(
                "Can only concatenate EnvelopeIOProcessor or "
                "ModelIOProcessor objects."
            )

        resolved_other = resolve_envelope_processor(other)
        if isinstance(resolved_other, ComposedEnvelopeIOProcessor):
            self.processors.extend(resolved_other.processors)
        else:
            self.processors.append(cast(EnvelopeIOProcessor, resolved_other))

        self.cfg += other.cfg
        return self

    def pre_process(
        self,
        data: PipelineEnvelope,
    ) -> PipelineEnvelope:
        """Apply child envelope processors in sequence and record context.

        ``processor_context`` values are stored exactly as returned by each
        child processor. This compose layer does not copy them. Processors
        that need isolated snapshots must return fresh context objects
        themselves.

        Args:
            data (PipelineEnvelope): Envelope to transform.

        Returns:
            PipelineEnvelope: Envelope whose ``processor_context`` is
                either the original passthrough value or a structured
                :class:`ProcessorContextStack` aligned with child processors.
        """

        current = normalize_pipeline_envelope(data)
        processor_context_stack: list[Any] = []
        for processor in self.processors:
            current = normalize_pipeline_envelope(
                processor.pre_process(current)
            )
            processor_context_stack.append(current.processor_context)

        if len(processor_context_stack) == 0:
            return current

        return PipelineEnvelope(
            model_input=current.model_input,
            processor_context=ProcessorContextStack(
                processor_context_stack=processor_context_stack,
            ),
        )

    def _normalize_processor_context_batch(
        self,
        processor_context: Any,
    ) -> tuple[list[ProcessorContextStack], bool]:
        """Normalize one or many composed context stacks for replay.

        Args:
            processor_context (Any): Either one
                :class:`ProcessorContextStack` from a non-collated path or a
                list of stacks aligned with a collated batch.

        Returns:
            tuple[list[ProcessorContextStack], bool]: Normalized stack list and
                a flag indicating whether the original input was already a
                collated batch of stacks.
        """
        if isinstance(processor_context, ProcessorContextStack):
            return [processor_context], False

        if isinstance(processor_context, list) and all(
            isinstance(item, ProcessorContextStack)
            for item in processor_context
        ):
            return cast(list[ProcessorContextStack], processor_context), True

        raise TypeError(
            "Composed envelope post_process expects a ProcessorContextStack "
            "or a list of ProcessorContextStack values, but got "
            f"{type(processor_context).__name__}."
        )

    def post_process(
        self,
        model_outputs,
        *,
        model_input: Any = None,
        processor_context: Any = None,
    ):
        """Replay child envelope processors in reverse order.

        Args:
            model_outputs: Raw outputs to replay through the composed child
                processors.
            model_input (Any, optional): Final model-facing input shared by all
                child ``post_process`` calls. Default is None.
            processor_context (Any, optional): Either a single
                :class:`ProcessorContextStack` for non-collated paths or a list
                of :class:`ProcessorContextStack` values aligned with the batch
                dimension for collated inference paths. Each stack entry must
                match ``self.processors`` in forward order. Direct callers may
                pass None when no structured compose context is available; in
                that case this compose layer replays None to each child instead
                of raising. Default is None.

        Returns:
            Any: Model outputs after replaying every child processor in reverse
                order.
        """

        if len(self.processors) == 0:
            return model_outputs

        if processor_context is None:
            # Direct post_process callers may not have structured compose
            # context; replay None to each child so that path degrades
            # gracefully instead of raising.
            for processor in reversed(self.processors):
                model_outputs = processor.post_process(
                    model_outputs,
                    model_input=model_input,
                    processor_context=None,
                )
            return model_outputs

        processor_context_batch, is_collated = (
            self._normalize_processor_context_batch(processor_context)
        )
        expected_len = len(self.processors)
        for context_stack in processor_context_batch:
            if len(context_stack.processor_context_stack) != expected_len:
                raise ValueError(
                    "Envelope compose context stack length does not match "
                    "child processor count."
                )

        for processor_index, processor in enumerate(reversed(self.processors)):
            stack_index = expected_len - processor_index - 1
            child_processor_contexts = [
                context_stack.processor_context_stack[stack_index]
                for context_stack in processor_context_batch
            ]
            child_processor_context: Any
            if is_collated:
                child_processor_context = child_processor_contexts
            else:
                child_processor_context = child_processor_contexts[0]
            model_outputs = processor.post_process(
                model_outputs,
                model_input=model_input,
                processor_context=child_processor_context,
            )
        return model_outputs


class ComposedEnvelopeIOProcessorCfg(
    EnvelopeIOProcessorCfg[ComposedEnvelopeIOProcessor]
):
    """Configuration for :class:`ComposedEnvelopeIOProcessor`."""

    class_type: ClassType_co[ComposedEnvelopeIOProcessor] = (
        ComposedEnvelopeIOProcessor
    )
    processors: list[ConfigInstanceOf[EnvelopeIOProcessorCfg]]

    def __getitem__(self, item: int) -> EnvelopeIOProcessorCfg:
        """Return a child envelope processor config by position.

        Args:
            item (int): Child processor config index.

        Returns:
            EnvelopeIOProcessorCfg: Child config at ``item``.
        """

        return self.processors[item]

    def __iadd__(
        self,
        other: EnvelopeIOProcessorCfg | ModelIOProcessorCfg,
    ) -> "ComposedEnvelopeIOProcessorCfg":
        """Append another config to this envelope compose config.

        Args:
            other (EnvelopeIOProcessorCfg | ModelIOProcessorCfg): Config to
                append.

        Returns:
            ComposedEnvelopeIOProcessorCfg: ``self`` after appending ``other``.
        """

        if not isinstance(
            other,
            (
                EnvelopeIOProcessorCfg,
                ModelIOProcessorCfg,
            ),
        ):
            raise TypeError(
                "Can only concatenate EnvelopeIOProcessorCfg or "
                "ModelIOProcessorCfg objects."
            )

        resolved_other = resolve_envelope_processor_cfg(other)
        if isinstance(resolved_other, ComposedEnvelopeIOProcessorCfg):
            self.processors = list(self.processors) + list(
                resolved_other.processors
            )
        else:
            self.processors = list(self.processors) + [
                cast(EnvelopeIOProcessorCfg, resolved_other)
            ]
        return self


def compose_envelope(
    *processors: EnvelopeIOProcessor | ModelIOProcessor | None,
) -> ComposedEnvelopeIOProcessor:
    """Build an envelope compose runtime chain.

    Args:
        *processors (EnvelopeIOProcessor | ModelIOProcessor | None):
            Envelope processors, legacy processors, or ``None`` placeholders
            to compose. ``None`` entries are ignored.

    Returns:
        ComposedEnvelopeIOProcessor: Flattened composed runtime chain.
    """

    if len(processors) == 0:
        return ComposedEnvelopeIOProcessorCfg(processors=[])()

    resolved_processors: list[EnvelopeIOProcessor] = []
    for processor in processors:
        resolved_processors.append(
            cast(
                EnvelopeIOProcessor,
                resolve_envelope_processor(processor),
            )
        )
    if len(resolved_processors) == 1 and isinstance(
        resolved_processors[0], ComposedEnvelopeIOProcessor
    ):
        return ComposedEnvelopeIOProcessor.from_processors(
            list(resolved_processors[0].processors)
        )

    return ComposedEnvelopeIOProcessor.from_processors(resolved_processors)


def compose_envelope_cfg(
    *processors: EnvelopeIOProcessorCfg | ModelIOProcessorCfg,
) -> ComposedEnvelopeIOProcessorCfg:
    """Build an envelope compose config from envelope or legacy cfg inputs.

    Args:
        *processors (EnvelopeIOProcessorCfg | ModelIOProcessorCfg): Envelope
            or legacy processor configs to compose.

    Returns:
        ComposedEnvelopeIOProcessorCfg: Flattened composed envelope config.
    """

    if len(processors) == 0:
        return ComposedEnvelopeIOProcessorCfg(processors=[])

    resolved_cfgs: list[EnvelopeIOProcessorCfg] = []
    for cfg in processors:
        resolved_cfgs.append(
            cast(
                EnvelopeIOProcessorCfg,
                resolve_envelope_processor_cfg(cfg),
            )
        )
    composed_cfg = ComposedEnvelopeIOProcessorCfg(processors=[])
    for cfg in resolved_cfgs:
        composed_cfg += cfg
    return composed_cfg
