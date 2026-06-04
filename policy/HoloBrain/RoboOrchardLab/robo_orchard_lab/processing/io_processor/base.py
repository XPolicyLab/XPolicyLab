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
import abc
from typing import TYPE_CHECKING, Any, TypeVar

from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,  # noqa: F401
)

from robo_orchard_lab.utils.state import State, StateSaveLoadMixin

if TYPE_CHECKING:
    from robo_orchard_lab.processing.io_processor.compose import (
        ComposedIOProcessor,
        ComposedIOProcessorCfg,
    )

__all__ = [
    "ClassType_co",
    "ModelIOProcessor",
    "ModelIOProcessorType_co",
    "ModelIOProcessorCfg",
    "ModelIOProcessorCfgType_co",
]


class ModelIOProcessor(
    ClassInitFromConfigMixin, StateSaveLoadMixin, metaclass=abc.ABCMeta
):
    """Legacy-compatible model boundary I/O processor base class.

    This class defines the standard interface for processors that encapsulate
    the pre-processing and post-processing logic required around a model. The
    primary role of a model I/O processor is to convert raw, user-provided or
    dataset-provided data into a format suitable for model consumption and to
    convert raw model outputs into a more user-friendly, structured
    representation.

    This family remains fully supported as a compatibility input surface, but
    new code should prefer :class:`EnvelopeIOProcessor` so model input and
    ``processor_context`` follow the canonical envelope runtime contract.

    In the standard inference pipeline workflow, :meth:`pre_process` is
    called once per raw sample before any collate function runs, so subclasses
    should usually treat its input as a single sample. By contrast,
    :meth:`post_process` runs after model forward and therefore commonly sees
    batched model inputs and batched model outputs produced after collation,
    even when the batch size is 1.

    Concrete processors are used directly by inference pipelines and may also
    be reused by training or evaluation step processors.
    """

    def __init__(self, cfg: "ModelIOProcessorCfg"):
        """Initialize the processor from its config object."""
        self._setup(cfg)

    def _setup(self, cfg: "ModelIOProcessorCfg") -> None:
        """Bind configuration onto the processor instance."""
        self.cfg = cfg

    @abc.abstractmethod
    def pre_process(self, data) -> Any:
        """Transform raw input data into a model-ready representation.

        Args:
            data (Any): Raw input data provided by callers or datasets.
                In the default inference flow this is one sample at a time,
                before batching or collation.

        Returns:
            Any: The transformed sample that should be consumed by the model
            directly or by a downstream collate function.
        """
        pass

    def post_process(self, model_outputs, model_input: Any = None):
        """Transform raw model outputs into a user-facing representation.

        The default implementation is an identity mapping and simply returns
        ``model_outputs`` unchanged.

        Args:
            model_outputs (Any): Raw outputs returned by the model forward
                pass. In the default inference flow this is usually batched
                model output.
            model_input (Any, optional): The transformed model input produced
                by :meth:`pre_process` and optional collation. In the default
                inference flow this is therefore usually batched as well.
                Some processors use it as additional context during
                post-processing. Default is None.

        Returns:
            Any: Post-processed model outputs.
        """
        return model_outputs

    def _get_state(self) -> State:
        """Get the state of the object for saving.

        The config object is lifted out of ``state`` into ``config`` so the
        serialized representation is easier to inspect and mirrors how runtime
        artifacts are typically organized on disk.

        Returns:
            State: Serialized processor state.
        """
        ret = super()._get_state()
        ret.config = ret.state.pop("cfg", None)
        return ret

    def _set_state(self, state: State) -> None:
        """Set the state of the object from a serialized state object.

        Args:
            state (State): Serialized processor state produced by
                :meth:`_get_state`.
        """
        state.state["cfg"] = state.config
        state.config = None
        super()._set_state(state)
        self._setup(self.cfg)

    def __add__(
        self, other: ModelIOProcessor | ComposedIOProcessor
    ) -> ComposedIOProcessor:
        """Build a composed processor by appending another processor.

        ``+`` never mutates the left-hand side. If ``self`` is already a
        composed processor, its processor list and config are shallow-copied
        before appending ``other``, so the returned processor preserves the
        original chain while exposing the extended one.

        Args:
            other: Another processor to append to this processing chain.

        Returns:
            ComposedIOProcessor: A new composed processor instance.

        Notes:
            This operator preserves the legacy compose family. New envelope-
            first code should prefer ``compose_envelope(...)`` or
            :class:`EnvelopeIOProcessor` composition helpers.
        """
        if not isinstance(other, ModelIOProcessor):
            raise TypeError(
                "Can only concatenate ModelIOProcessor or "
                "ComposedIOProcessor objects."
            )

        from robo_orchard_lab.processing.io_processor.compose import (
            ComposedIOProcessor,
            ComposedIOProcessorCfg,
        )

        new_processor = ComposedIOProcessor.__new__(ComposedIOProcessor)
        if isinstance(self, ComposedIOProcessor):
            new_processor.cfg = self.cfg.model_copy(deep=False)
            new_processor.processors = list(self.processors)
        else:
            new_processor.cfg = ComposedIOProcessorCfg(processors=[self.cfg])
            new_processor.processors = [self]

        new_processor += other
        return new_processor


ModelIOProcessorType_co = TypeVar(
    "ModelIOProcessorType_co",
    bound=ModelIOProcessor,
    covariant=True,
)


class ModelIOProcessorCfg(ClassConfig[ModelIOProcessorType_co]):
    """Base configuration class for :class:`ModelIOProcessor`.

    This Pydantic-based config stores the processor class to instantiate and
    any processor-specific fields required to build it. Subclasses are
    responsible for specifying ``class_type`` and their own runtime settings.

    The config remains valid on public pipeline surfaces, but new code should
    prefer :class:`EnvelopeIOProcessorCfg` so runtime resolution can stay in
    the canonical envelope family from the start.
    """

    def __add__(
        self, other: ModelIOProcessorCfg | ComposedIOProcessorCfg
    ) -> ComposedIOProcessorCfg:
        """Build a composed config by appending another processor config.

        ``+`` returns a new composed config and leaves the original config
        object unchanged.

        Args:
            other: Another processor config to append to this configuration.

        Returns:
            ComposedIOProcessorCfg: A new composed processor config instance.
        """
        if not isinstance(other, ModelIOProcessorCfg):
            raise TypeError(
                "Can only concatenate ModelIOProcessorCfg objects."
            )

        from robo_orchard_lab.processing.io_processor.compose import (
            ComposedIOProcessorCfg,
        )

        if isinstance(self, ComposedIOProcessorCfg):
            new_cfg = self.model_copy(deep=False)
        else:
            new_cfg = ComposedIOProcessorCfg(processors=[self])

        new_cfg += other
        return new_cfg


ModelIOProcessorCfgType_co = TypeVar(
    "ModelIOProcessorCfgType_co",
    bound=ModelIOProcessorCfg,
    covariant=True,
)
