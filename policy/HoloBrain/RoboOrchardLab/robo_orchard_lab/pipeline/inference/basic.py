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
import warnings
from typing import (
    Any,
    Generator,
    Generic,
    Iterable,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import torch
from robo_orchard_core.utils.config import (
    CallableType,
    ClassType_co,
    ConfigInstanceOf,
)
from torch.utils.data import Dataset
from typing_extensions import deprecated

from robo_orchard_lab.dataset.collates import (
    CollatorConfig,
    collate_batch_dict,
)
from robo_orchard_lab.models.mixin import TorchModelMixin
from robo_orchard_lab.pipeline.inference.mixin import (
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
    InputType,
    OutputType,
)
from robo_orchard_lab.processing.io_processor import (
    EnvelopeIOProcessor,
    EnvelopeIOProcessorCfg,
    PipelineEnvelope,
)
from robo_orchard_lab.processing.io_processor.base import (
    ModelIOProcessor,
    ModelIOProcessorCfg,
)
from robo_orchard_lab.processing.io_processor.envelope import (
    ModelIOProcessorEnvelopeAdapter,
    PostProcessContext,
    normalize_pipeline_envelope,
    resolve_envelope_processor,
)
from robo_orchard_lab.utils.torch import to_device

__all__ = ["InferencePipeline", "InferencePipelineCfg"]


DatasetType: TypeAlias = Dataset | list | tuple | Generator
PreparedModelInputT = TypeVar("PreparedModelInputT")
PreparedProcessorContextT = TypeVar("PreparedProcessorContextT")


class InferencePipeline(
    InferencePipelineMixin[InputType, OutputType],
    Generic[InputType, OutputType],
):
    """A concrete end-to-end inference pipeline.

    Like ``Pipeline`` in Hugging Face Transformers, this class provides a
    user-friendly interface for performing inference with a model while
    handling the surrounding runtime steps such as pre-processing, batching,
    model forwarding, and post-processing.

    The canonical workflow in :meth:`__call__` is:

    1. Wrap each raw sample into a single-sample :class:`PipelineEnvelope`.
    2. Resolve the configured processor into the envelope family.
    3. If ``collate_fn`` is configured, collate only ``model_input`` into
       model-facing batched form. Dataset-like inputs collate by
       ``batch_size`` mini-batches, while single-sample inputs collate as a
       size-1 batch.
    4. Pass ``processor_context`` through post-process without collating it:
       non-collated paths keep the original object, while collated paths pass
       a per-sample list aligned with the collated batch.
    5. Post-process using the resolved envelope processor when present.
    """

    cfg: InferencePipelineCfg
    envelope_processor: EnvelopeIOProcessor | None
    collate_fn: CallableType[[list[Any]], Any] | None

    def __init__(
        self,
        cfg: InferencePipelineMixinCfg,
        model: TorchModelMixin | None = None,
    ):
        """Initialize the inference pipeline.

        Args:
            cfg (InferencePipelineMixinCfg): Configuration for the pipeline.
                Concrete configs may additionally define an optional I/O
                processor, collate function, and batch size.
            model (TorchModelMixin | None, optional): A pre-built model
                instance to bind to the pipeline. If None, the model is
                instantiated from ``cfg.model_cfg`` by the mixin. Default is
                None.
        """
        super().__init__(cfg=cfg, model=model)

    def _setup(self, cfg: InferencePipelineMixinCfg, model: TorchModelMixin):
        """Configure the pipeline runtime helpers.

        Besides storing ``cfg`` and ``model`` through the mixin, this method
        instantiates the configured processor, resolves the canonical envelope
        processor, resolves the effective collate function, and warns when a
        downstream class still defines the unsupported legacy private
        ``_model_forward_with_processor(...)`` hook name. That legacy name is
        never called by runtime and is detected only to surface migration
        warnings during setup, including after loading or state restoration.

        Args:
            cfg (InferencePipelineMixinCfg): The pipeline configuration.
            model (TorchModelMixin): The model bound to the pipeline.
        """
        super()._setup(cfg, model)
        source_processor = self.cfg.processor() if self.cfg.processor else None
        self.envelope_processor = resolve_envelope_processor(source_processor)
        if isinstance(self.cfg.collate_fn, CollatorConfig):
            self.collate_fn = self.cfg.collate_fn()
        else:
            self.collate_fn = self.cfg.collate_fn

        if (
            getattr(type(self), "_model_forward_with_processor", None)
            is not None
            and type(self)._model_forward_with_envelope
            is InferencePipeline._model_forward_with_envelope
        ):
            warnings.warn(
                f"{type(self).__name__} still provides "
                "unsupported private "
                "`_model_forward_with_processor(...)` hook name. "
                "`InferencePipeline` runtime never calls it. Override "
                "`_model_forward_with_envelope(...)` instead.",
                UserWarning,
                stacklevel=3,
            )

    def _get_ignore_save_attributes(self) -> list[str]:
        """Return runtime-only attributes that should not be serialized.

        The processor and collate function can be reconstructed from
        ``self.cfg``, so saved state only needs to persist the model and the
        configuration metadata.

        Returns:
            list[str]: Attribute names that should be excluded from saved
                state.
        """
        return super()._get_ignore_save_attributes() + [
            "envelope_processor",
            "collate_fn",
        ]

    def reset(self, **kwargs) -> None:
        """Reset standard inference-pipeline runtime state.

        The default concrete pipeline keeps no episode-local mutable state,
        so reset is a compatibility no-op. Subclasses may override this hook
        when they cache per-episode data or consume reset metadata.

        Args:
            kwargs: Optional pipeline-specific reset arguments.
        """
        pass

    @property
    @deprecated(
        "Use `envelope_processor` instead. `processor` remains a "
        "deprecated compatibility alias that unwraps legacy adapters and "
        "otherwise returns the configured envelope processor.",
        category=None,
    )  # type: ignore
    def processor(
        self,
    ) -> ModelIOProcessor | EnvelopeIOProcessor | None:
        """Backward-compatible alias for the configured processor.

        Legacy adapter-backed pipelines return the wrapped
        :class:`ModelIOProcessor`. Envelope-native pipelines return the
        configured :class:`EnvelopeIOProcessor` directly. Assignments resolve
        through :attr:`envelope_processor` so existing runtime reassignment
        code can keep using ``pipeline.processor = ...``.
        """

        if self.envelope_processor is None:
            return None
        if isinstance(
            self.envelope_processor, ModelIOProcessorEnvelopeAdapter
        ):
            return self.envelope_processor.legacy
        return self.envelope_processor

    @processor.setter
    def processor(
        self,
        value: ModelIOProcessor | EnvelopeIOProcessor | None,
    ) -> None:
        """Resolve and store the canonical envelope processor runtime."""

        self.envelope_processor = resolve_envelope_processor(value)

    @overload
    def __call__(self, data: InputType) -> OutputType: ...

    @overload
    def __call__(self, data: DatasetType) -> Iterable[OutputType]: ...

    @torch.inference_mode()
    def __call__(
        self, data: InputType | DatasetType
    ) -> OutputType | Iterable[OutputType]:
        """Execute the canonical end-to-end inference workflow.

        Args:
            data (InputType | DatasetType): Raw input data for the pipeline.
                It can be a single sample or a dataset-like iterable of
                samples, such as a generator, dataset, list, or tuple. When an
                iterable is provided, the data is processed in mini-batches of
                size ``self.cfg.batch_size`` and the method yields one result
                per batch. ``batch_size`` only controls this dataset-like
                path. When a single sample is provided and ``collate_fn`` is
                configured, runtime still collates the sample into a size-1
                model batch for shape consistency with batched inference.

        Returns:
            OutputType | Iterable[OutputType]: The post-processed inference
                result. A single input returns one output, while a dataset-like
                input returns an iterator that yields one output per batch.
        """
        if not isinstance(data, (Dataset, list, tuple, Generator)):
            return self._inference_single(data)
        return self._inference_batch_gen(data)

    def _inference_batch_gen(self, data: DatasetType) -> Iterable[OutputType]:
        """Yield batched inference results from a dataset-like input.

        Args:
            data (DatasetType): Dataset-like input containing raw samples.

        Returns:
            Iterable[OutputType]: An iterator that yields the post-processed
                result for each mini-batch assembled from ``data``.
        """
        batch = []
        for sample in data:
            if len(batch) != self.cfg.batch_size:
                batch.append(sample)
            if len(batch) == self.cfg.batch_size:
                yield self._inference_batch(batch)
                batch = []
        if len(batch) > 0:
            yield self._inference_batch(batch)

    def _inference_batch(self, batch: Iterable[InputType]) -> OutputType:
        """Execute the inference workflow for a batch of raw samples.

        Args:
            batch (Iterable[InputType]): Raw input samples for the pipeline.

        Returns:
            OutputType: Final post-processed batch result.
        """
        model_input, processor_context = self._build_runtime_input(
            batch, is_batch=True
        )
        return self._model_forward_with_envelope(
            model_input,
            processor_context=processor_context,
        )

    def _prepare_single_input(
        self, data: InputType
    ) -> PipelineEnvelope[Any, Any]:
        """Prepare one raw sample into a pipeline envelope."""

        envelope = PipelineEnvelope(model_input=data)
        if self.envelope_processor is not None:
            envelope = self.envelope_processor.pre_process(envelope)
        return normalize_pipeline_envelope(envelope)

    def _collate_model_input_keep_processor_context(
        self,
        envelope_batch: list[
            PipelineEnvelope[PreparedModelInputT, PreparedProcessorContextT]
        ],
    ) -> tuple[Any, list[PreparedProcessorContextT]]:
        """Collate model input while leaving processor context untouched.

        Args:
            envelope_batch (list[PipelineEnvelope]): Per-sample envelopes
                produced by ``pre_process``.

        Returns:
            tuple[Any, list[PreparedProcessorContextT]]: Collated model input
                plus a list of original per-sample ``processor_context``
                values. If this helper is used, ``processor_context`` is
                always returned as a list, even when only one sample was
                collated.
        """
        model_inputs = [item.model_input for item in envelope_batch]
        if self.collate_fn is None:
            warnings.warn(
                "No collate function is specified in the pipeline config for "
                "batch inference. Using default collate function "
                "`collate_batch_dict`, which assumes each data sample is a "
                "dictionary."
            )
            collated_model_input = collate_batch_dict(model_inputs)  # type: ignore[arg-type]
        else:
            collated_model_input = self.collate_fn(model_inputs)

        processor_contexts = [
            item.processor_context for item in envelope_batch
        ]
        return collated_model_input, processor_contexts

    def _build_runtime_input(
        self,
        data: InputType | Iterable[InputType],
        *,
        is_batch: bool,
    ) -> tuple[Any, PostProcessContext[Any]]:
        """Prepare model input plus passthrough processor context.

        Args:
            data (InputType | Iterable[InputType]): Raw input sample or raw
                input batch.
            is_batch (bool): Whether ``data`` is a batch of raw samples.

        Returns:
            tuple[Any, PostProcessContext[Any]]: Model-facing input and
                passthrough ``processor_context`` payload. The runtime
                collates only model input. If model input collation is used,
                ``processor_context`` is returned as a list of per-sample
                values for the processor to interpret, even when the list
                length is 1. For single-sample inference, model input is
                collated only when ``self.collate_fn`` is configured.
        """
        if is_batch:
            batch_inputs = list(cast(Iterable[InputType], data))
            envelope_batch = [
                self._prepare_single_input(sample) for sample in batch_inputs
            ]
            return self._collate_model_input_keep_processor_context(
                envelope_batch
            )

        envelope = self._prepare_single_input(cast(InputType, data))
        if self.collate_fn is None:
            return envelope.model_input, envelope.processor_context

        collated_model_input, processor_context = (
            self._collate_model_input_keep_processor_context([envelope])
        )
        return collated_model_input, processor_context

    def _model_forward_with_envelope(
        self,
        data: PreparedModelInputT,
        *,
        processor_context: PostProcessContext[
            PreparedProcessorContextT
        ] = None,
    ) -> OutputType:
        """Run the canonical forward path with envelope passthrough context.

        This is the public subclass override seam for inference runtimes that
        need direct access to envelope ``processor_context``. The standard
        pipeline runtime always dispatches here.

        Args:
            data (PreparedModelInputT): Prepared model-facing input.
            processor_context (PostProcessContext[
                PreparedProcessorContextT], optional): Envelope passthrough
                context aligned with ``data``. When model-input collation is
                used, this is typically a list of per-sample contexts or
                other structured runtime-owned batch metadata. Default is
                None.

        Returns:
            OutputType: Final inference output after optional post-process.
        """
        model_outputs = self._model_forward(data)
        if self.envelope_processor is not None:
            return self.envelope_processor.post_process(
                model_outputs,
                model_input=data,
                processor_context=processor_context,
            )
        return model_outputs

    def _inference_single(self, data: InputType) -> OutputType:
        """Execute the inference workflow for one raw sample.

        Args:
            data (InputType): Raw input sample for the pipeline.

        Returns:
            OutputType: Final post-processed result. When ``collate_fn`` is
                configured, the sample is first normalized into a size-1 model
                batch before model forward.
        """
        model_input, processor_context = self._build_runtime_input(
            data, is_batch=False
        )
        return self._model_forward_with_envelope(
            model_input,
            processor_context=processor_context,
        )

    def _model_forward(self, data: Any) -> Any:
        """Perform the model's forward pass.

        Args:
            data (Any): Model-facing input data. It is already batched when a
                collate function is configured or when dataset-like inference
                is used.

        Returns:
            Any: Raw model outputs before optional post-processing.
        """
        data = to_device(data, self.model.device)
        return self.model(data)


InferencePipelineType_co = TypeVar(
    "InferencePipelineType_co",
    bound=InferencePipeline,
    covariant=True,
)


class InferencePipelineCfg(
    InferencePipelineMixinCfg[InferencePipelineType_co]
):
    """Configuration for the concrete :class:`InferencePipeline`.

    This class extends :class:`InferencePipelineMixinCfg` with additional,
    runtime-specific settings for data handling, including the processor,
    collate function, and mini-batch size.
    """

    class_type: ClassType_co[InferencePipelineType_co] = InferencePipeline  # type: ignore # noqa: E501

    processor: (
        ConfigInstanceOf[ModelIOProcessorCfg]
        | ConfigInstanceOf[EnvelopeIOProcessorCfg]
        | None
    ) = None
    """Configuration for an I/O processor.

    New code should prefer ``EnvelopeIOProcessorCfg``.
    ``ModelIOProcessorCfg`` remains supported as a compatibility input and is
    automatically resolved into the envelope runtime.
    """

    collate_fn: (
        ConfigInstanceOf[CollatorConfig]
        | CallableType[[list[Any]], Any]
        | None
    ) = None
    """Optional function used to collate ``model_input`` into batch form.

    When configured, the pipeline always normalizes model-facing input through
    this callable, including single-sample inference. In that case, a single
    raw sample is collated as a size-1 batch so the model sees the same input
    structure as the batched inference path.

    When omitted, single-sample inference forwards the prepared
    ``model_input`` directly. Dataset-like inference still collates the batch,
    but falls back to ``collate_batch_dict`` with a warning.
    """

    batch_size: int = 1
    """Mini-batch size for dataset-like inputs only.

    This field controls how :meth:`InferencePipeline.__call__` chunks
    datasets, lists, tuples, and generators in ``_inference_batch_gen``. It
    does not disable size-1 collation on the single-sample path when
    ``collate_fn`` is configured.
    """
