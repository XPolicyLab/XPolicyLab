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
import abc
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,
    ConfigInstanceOf,
)

from robo_orchard_lab.processing.io_processor.base import (
    ModelIOProcessor,
    ModelIOProcessorCfg,
    ModelIOProcessorType_co,
)
from robo_orchard_lab.utils.state import State, StateSaveLoadMixin

if TYPE_CHECKING:
    from robo_orchard_lab.processing.io_processor.compose import (
        ComposedIOProcessor,
        ComposedIOProcessorCfg,
    )
    from robo_orchard_lab.processing.io_processor.compose_envelope import (
        ComposedEnvelopeIOProcessor,
        ComposedEnvelopeIOProcessorCfg,
    )

__all__ = [
    "ClassType_co",
    "PipelineEnvelope",
    "EnvelopeIOProcessor",
    "EnvelopeIOProcessorCfg",
    "EnvelopeIOProcessorCfgType_co",
    "EnvelopeIOProcessorType_co",
    "ModelIOProcessorEnvelopeAdapter",
    "ModelIOProcessorEnvelopeAdapterCfg",
    "adapt_model_io_processor_to_envelope",
    "normalize_pipeline_envelope",
    "resolve_envelope_processor",
    "resolve_envelope_processor_cfg",
]


ModelInputT = TypeVar("ModelInputT")
ProcessorContextT = TypeVar("ProcessorContextT")
PostProcessContextT = TypeVar("PostProcessContextT")

PostProcessContext: TypeAlias = (
    PostProcessContextT | list[PostProcessContextT] | None
)


@dataclass
class PipelineEnvelope(Generic[ModelInputT, ProcessorContextT]):
    """Envelope carrying model input and processor-context payload.

    :class:`PipelineEnvelope` describes structure, not granularity.
    ``model_input`` and ``processor_context`` may represent a single sample or
    an already batched payload, depending on the owner runtime. Standard
    :class:`InferencePipeline` pre-process boundaries start from single-sample
    envelopes before optional model-input collation. Other runtimes, such as
    step processors, may pass batch payloads directly.

    Args:
        model_input (Any): Model-facing payload for the current runtime unit.
            It may be a single sample or a batch.
        processor_context (Any, optional): Side-channel context aligned with
            the same runtime unit as ``model_input``. It is preserved for
            post-processing but not passed into model forward. Default is
            None.
    """

    model_input: ModelInputT
    processor_context: ProcessorContextT = None  # type: ignore[assignment]


@overload
def normalize_pipeline_envelope(
    value: PipelineEnvelope[ModelInputT, ProcessorContextT],
) -> PipelineEnvelope[ModelInputT, ProcessorContextT]: ...


@overload
def normalize_pipeline_envelope(
    value: ModelInputT,
) -> PipelineEnvelope[ModelInputT, None]: ...


def normalize_pipeline_envelope(value: Any) -> PipelineEnvelope[Any, Any]:
    """Normalize a raw value into :class:`PipelineEnvelope`.

    Args:
        value (Any): Existing envelope or a raw model-facing value.

    Returns:
        PipelineEnvelope[Any, Any]: Normalized envelope.
    """

    if isinstance(value, PipelineEnvelope):
        return value
    return PipelineEnvelope(model_input=value)


EnvelopeIOProcessorType_co = TypeVar(
    "EnvelopeIOProcessorType_co",
    bound="EnvelopeIOProcessor",
    covariant=True,
)


class EnvelopeIOProcessor(
    ClassInitFromConfigMixin,
    StateSaveLoadMixin,
    metaclass=abc.ABCMeta,
):
    """Base class for processors that consume and return envelopes.

    This family is separate from :class:`ModelIOProcessor`. It makes
    ``processor_context`` an explicit part of the runtime contract by always
    passing a :class:`PipelineEnvelope` through ``pre_process``. The payload
    granularity is owner-defined: standard inference pre-process boundaries
    usually start from single samples, while other runtimes may pass batch
    payloads directly. The standard inference runtime later collates only
    ``model_input`` and leaves ``processor_context`` in processor-owned form.
    In composed envelope chains, ``processor_context`` is passed through
    exactly as returned by child processors. The framework does not copy or
    deepcopy it. Processors that need isolation from later mutation must
    return fresh context objects themselves.

    Example:
        ``processor = MyEnvelopeProcessorCfg(...)()``
        ``envelope = processor.pre_process(PipelineEnvelope(model_input=x))``
    """

    def __init__(self, cfg: "EnvelopeIOProcessorCfg"):
        """Initialize the envelope processor from its config object."""
        self._setup(cfg)

    def _setup(self, cfg: "EnvelopeIOProcessorCfg") -> None:
        """Bind configuration onto the envelope processor instance."""
        self.cfg = cfg

    @abc.abstractmethod
    def pre_process(
        self,
        data: PipelineEnvelope[Any, Any],
    ) -> PipelineEnvelope[Any, Any]:
        """Transform an envelope before model forward.

        Args:
            data (PipelineEnvelope[Any, Any]): Owner-defined envelope
                containing model input plus optional aligned
                ``processor_context``. Depending on the runtime,
                ``data.model_input`` may describe a single sample or a batch.

        Returns:
            PipelineEnvelope[Any, Any]: Updated envelope. If the returned
                ``processor_context`` must remain isolated from later
                mutation, return a fresh context object instead of reusing and
                mutating a shared reference.
        """

    def post_process(
        self,
        model_outputs,
        *,
        model_input: Any = None,
        processor_context: PostProcessContext[Any] = None,
    ):
        """Transform raw model outputs into a user-facing representation.

        Args:
            model_outputs (Any): Raw model outputs returned by the model.
            model_input (Any, optional): Model-facing payload that was passed
                into the forward call. Default is None.
            processor_context (PostProcessContext[Any], optional): Envelope
                passthrough context preserved from pre-processing. The
                standard inference runtime never collates it by itself. Paths
                that bypass model-input collation pass the original object.
                Paths that do collate model input pass a list of per-sample
                context values, even when the list length is 1. Composed
                envelope runtimes may use structured wrappers such as
                ``ProcessorContextStack`` to encode per-child stacks without
                overloading plain lists. Entries are replayed using the exact
                references returned by the matching ``pre_process`` path; no
                copy or deepcopy is applied. Default is None.

        Returns:
            Any: Post-processed model outputs.
        """

        del model_input, processor_context
        return model_outputs

    def _get_state(self) -> State:
        """Get the processor state for serialization."""
        ret = super()._get_state()
        ret.config = ret.state.pop("cfg", None)
        return ret

    def _set_state(self, state: State) -> None:
        """Restore the processor state from serialized data."""
        state.state["cfg"] = state.config
        state.config = None
        super()._set_state(state)
        self._setup(self.cfg)

    def __add__(
        self,
        other: (EnvelopeIOProcessor | ModelIOProcessor | ComposedIOProcessor),
    ) -> ComposedEnvelopeIOProcessor:
        """Build an envelope compose chain.

        The left-hand side is not mutated.
        """

        if not isinstance(other, (EnvelopeIOProcessor, ModelIOProcessor)):
            raise TypeError(
                "Can only concatenate EnvelopeIOProcessor or "
                "ModelIOProcessor objects."
            )

        from robo_orchard_lab.processing.io_processor.compose_envelope import (
            compose_envelope,
        )

        return compose_envelope(self, other)


class EnvelopeIOProcessorCfg(ClassConfig[EnvelopeIOProcessorType_co]):
    """Base configuration class for :class:`EnvelopeIOProcessor`."""

    def __add__(
        self,
        other: (
            EnvelopeIOProcessorCfg
            | ModelIOProcessorCfg
            | ComposedIOProcessorCfg
        ),
    ) -> ComposedEnvelopeIOProcessorCfg:
        """Build an envelope compose config.

        The left-hand side is not mutated.
        """

        if not isinstance(
            other, (EnvelopeIOProcessorCfg, ModelIOProcessorCfg)
        ):
            raise TypeError(
                "Can only concatenate EnvelopeIOProcessorCfg or "
                "ModelIOProcessorCfg objects."
            )

        from robo_orchard_lab.processing.io_processor.compose_envelope import (
            compose_envelope_cfg,
        )

        return compose_envelope_cfg(self, other)


EnvelopeIOProcessorCfgType_co = TypeVar(
    "EnvelopeIOProcessorCfgType_co",
    bound=EnvelopeIOProcessorCfg,
    covariant=True,
)

LegacyProcessorT = TypeVar(
    "LegacyProcessorT",
    bound=ModelIOProcessor,
)
LegacyProcessorCfgT = TypeVar(
    "LegacyProcessorCfgT",
    bound=ModelIOProcessorCfg,
)


class ModelIOProcessorEnvelopeAdapter(
    EnvelopeIOProcessor,
    Generic[LegacyProcessorT],
):
    """Envelope-path adapter that wraps a legacy :class:`ModelIOProcessor`.

    The adapter preserves the original legacy processor contract while making
    it usable inside the canonical envelope runtime path. ``processor_context``
    is passed through unchanged and is not interpreted by the wrapped legacy
    processor. Main public APIs are :meth:`from_legacy`, the :attr:`legacy`
    property, :meth:`pre_process`, and :meth:`post_process`. Use
    :meth:`from_legacy` when you need to preserve an existing runtime instance;
    config-based construction lazily materializes a wrapped legacy processor
    from ``cfg`` instead.

    Example:
        ``legacy = MyLegacyProcessorCfg()()``
        ``envelope = ModelIOProcessorEnvelopeAdapter.from_legacy(legacy)``
    """

    cfg: "ModelIOProcessorEnvelopeAdapterCfg[ModelIOProcessorCfg]"
    _legacy: LegacyProcessorT

    def _setup(
        self, cfg: "ModelIOProcessorEnvelopeAdapterCfg[ModelIOProcessorCfg]"
    ) -> None:
        """Bind the adapter config and invalidate stale lazy legacy caches."""

        old_cfg = getattr(self, "cfg", None)
        super()._setup(cfg)
        if old_cfg is not cfg:
            self.__dict__.pop("_legacy", None)

    @property
    def legacy(self) -> LegacyProcessorT:
        """Return the wrapped legacy processor, creating it on first access."""

        if "_legacy" not in self.__dict__:
            self._legacy = cast(LegacyProcessorT, self.cfg.legacy_processor())
        return self._legacy

    @legacy.setter
    def legacy(self, legacy: LegacyProcessorT) -> None:
        """Store the wrapped legacy processor instance."""

        self._legacy = legacy

    @classmethod
    def from_legacy(
        cls,
        legacy: LegacyProcessorT,
    ) -> "ModelIOProcessorEnvelopeAdapter[LegacyProcessorT]":
        """Build an adapter while preserving the given runtime instance."""

        cfg = ModelIOProcessorEnvelopeAdapterCfg(legacy_processor=legacy.cfg)
        adapter = cls.__new__(cls)
        EnvelopeIOProcessor._setup(adapter, cfg)
        adapter.legacy = legacy
        return cast(ModelIOProcessorEnvelopeAdapter[LegacyProcessorT], adapter)

    def pre_process(
        self,
        data: PipelineEnvelope[Any, Any],
    ) -> PipelineEnvelope[Any, Any]:
        """Run the wrapped legacy ``pre_process`` on ``model_input`` only."""

        return PipelineEnvelope(
            model_input=self.legacy.pre_process(data.model_input),
            processor_context=data.processor_context,
        )

    def post_process(
        self,
        model_outputs,
        *,
        model_input: Any = None,
        processor_context: PostProcessContext[Any] = None,
    ):
        """Delegate post-processing to the wrapped legacy processor."""

        del processor_context
        return self.legacy.post_process(model_outputs, model_input)


class ModelIOProcessorEnvelopeAdapterCfg(
    EnvelopeIOProcessorCfg[ModelIOProcessorEnvelopeAdapter],
    Generic[LegacyProcessorCfgT],
):
    """Config for :class:`ModelIOProcessorEnvelopeAdapter`."""

    class_type: ClassType_co[ModelIOProcessorEnvelopeAdapter] = (
        ModelIOProcessorEnvelopeAdapter
    )
    legacy_processor: ConfigInstanceOf[LegacyProcessorCfgT]


def adapt_model_io_processor_to_envelope(
    processor: ModelIOProcessorType_co,
) -> ModelIOProcessorEnvelopeAdapter[ModelIOProcessorType_co]:
    """Wrap a legacy runtime processor for envelope-path execution.

    Args:
        processor (ModelIOProcessorType_co): Legacy runtime processor.

    Returns:
        ModelIOProcessorEnvelopeAdapter[ModelIOProcessorType_co]: Adapter that
            preserves the original runtime processor instance.
    """

    return ModelIOProcessorEnvelopeAdapter.from_legacy(processor)


def resolve_envelope_processor(
    processor: (
        EnvelopeIOProcessor | ModelIOProcessor | ComposedIOProcessor | None
    ),
) -> EnvelopeIOProcessor | None:
    """Resolve a runtime processor into the canonical envelope family.

    Legacy processors, including legacy composed processors, are adapted as
    one legacy object so the upgrade path remains reversible.

    Args:
        processor: Envelope processor, legacy processor, legacy compose, or
            None.

    Returns:
        EnvelopeIOProcessor | None: Canonical envelope runtime object.

    Raises:
        TypeError: If the value cannot be resolved into the envelope family.
    """

    if processor is None:
        return None

    if isinstance(processor, EnvelopeIOProcessor):
        return processor

    if isinstance(processor, ModelIOProcessor):
        return adapt_model_io_processor_to_envelope(processor)

    raise TypeError(
        "Expected EnvelopeIOProcessor, ModelIOProcessor, or None, "
        f"but got {type(processor).__name__}."
    )


def resolve_envelope_processor_cfg(
    cfg: (
        EnvelopeIOProcessorCfg
        | ModelIOProcessorCfg
        | ComposedIOProcessorCfg
        | None
    ),
) -> EnvelopeIOProcessorCfg | None:
    """Resolve a config into the canonical envelope processor family.

    Legacy configs, including legacy compose configs, are wrapped once by the
    adapter cfg so the original config remains directly recoverable.

    Args:
        cfg: Envelope cfg, legacy cfg, legacy compose cfg, or None.

    Returns:
        EnvelopeIOProcessorCfg | None: Canonical envelope cfg.

    Raises:
        TypeError: If the value cannot be resolved into the envelope family.
    """

    if cfg is None:
        return None

    if isinstance(cfg, EnvelopeIOProcessorCfg):
        return cfg

    if isinstance(cfg, ModelIOProcessorCfg):
        return ModelIOProcessorEnvelopeAdapterCfg(legacy_processor=cfg)

    raise TypeError(
        "Expected EnvelopeIOProcessorCfg, ModelIOProcessorCfg, or "
        f"None, but got {type(cfg).__name__}."
    )
