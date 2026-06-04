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
import logging
import os
from typing import Generic, Literal, TypeAlias

import torch
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType_co,
    ConfigInstanceOf,
    load_from,
)
from typing_extensions import TypeVar

from robo_orchard_lab.models.mixin import TorchModelMixin, TorchModuleCfg
from robo_orchard_lab.models.model_ref import TorchModelRef
from robo_orchard_lab.utils.huggingface import resolve_hf_compatible_path
from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    abspath,
    in_cwd,
    is_empty_directory,
)
from robo_orchard_lab.utils.state import State, StateSaveLoadMixin

__all__ = [
    "ClassType_co",
    "InferencePipelineMixin",
    "InferencePipelineMixinCfg",
]

logger = logging.getLogger(__name__)

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")
DeviceMapValue = int | str | torch.device
DeviceMap = dict[str, DeviceMapValue]
PipelineModelCfg: TypeAlias = (
    ConfigInstanceOf[TorchModuleCfg[TorchModelMixin]]
    | ConfigInstanceOf[TorchModelRef[TorchModelMixin]]
)


class InferencePipelineMixin(
    StateSaveLoadMixin, Generic[InputType, OutputType], metaclass=abc.ABCMeta
):
    """An abstract base class for end-to-end inference pipelines.

    This generic mixin provides a common framework for orchestrating the
    inference process. It is responsible for holding the model and its
    configuration, while delegating task-specific inference behavior to
    subclasses. It also standardizes persistence helpers for saving and
    loading the full pipeline state.

    Subclasses should be specialized with ``InputType`` and ``OutputType`` and
    must implement :meth:`__call__` to define the core task-specific logic.

    Template Args:
        InputType: The type of the input data consumed by the pipeline.
        OutputType: The type of the data produced by the pipeline.
    """

    InitFromConfig: bool = True

    model: TorchModelMixin
    """The underlying model used in the pipeline."""

    cfg: InferencePipelineMixinCfg
    """The configuration bound to the pipeline instance."""

    def __init__(
        self,
        cfg: InferencePipelineMixinCfg,
        model: TorchModelMixin | None = None,
    ):
        """Initialize the inference pipeline with configuration and a model.

        This constructor either instantiates the model from
        ``cfg.model_cfg`` or binds an already constructed model instance into
        the pipeline. ``cfg.model_cfg`` remains the compatibility field name,
        but may now carry either a concrete ``TorchModuleCfg`` or a
        ``TorchModelRef`` that resolves into the model. When a model instance
        is provided, the config-side structural source of truth must still
        match ``model.cfg`` so the saved configuration stays self-consistent.

        Args:
            cfg (InferencePipelineMixinCfg): The configuration for the
                inference pipeline, including the model compatibility input
                when ``model`` is not provided.
            model (TorchModelMixin | None, optional): An optional model
                instance to use in the pipeline. If None, the model is
                resolved from ``cfg.model_cfg``. Default is None.

        Raises:
            ValueError: If no model configuration is available, or if
                ``cfg.model_cfg`` conflicts with the configuration carried by
                ``model``.
        """
        if model is None:
            model = cfg.build_model()
        else:
            cfg.validate_model(model)
        self._setup(cfg=cfg, model=model)

    def _setup(self, cfg: InferencePipelineMixinCfg, model: TorchModelMixin):
        """Configure the pipeline with the given parameters.

        This method stores the configuration and model on the pipeline
        instance. Subclasses can override it to install additional runtime
        helpers when the pipeline is initialized or restored from state.

        Args:
            cfg (InferencePipelineMixinCfg): The configuration for the
                inference pipeline.
            model (TorchModelMixin): The model to bind to the pipeline.
        """
        self.cfg = cfg
        self.model = model

    def to(self, device: str | torch.device):
        """Move the underlying model to the specified device.

        Args:
            device (str | torch.device): The target device for the model.
        """
        self.model.to(device)

    @property
    def device(self) -> torch.device:
        """The device where the model's parameters are currently located."""
        return self.model.device

    @abc.abstractmethod
    def __call__(self, data: InputType) -> OutputType:
        """Execute the end-to-end inference flow for a single input.

        This method defines the core inference logic and must be implemented by
        subclasses.

        Args:
            data (InputType): Raw input data for the pipeline.

        Returns:
            OutputType: The final processed result.
        """
        pass

    def save_pipeline(
        self,
        directory: str,
        inference_prefix: str = "inference",
        model_prefix: str = "model",
        required_empty: bool = True,
    ):
        """Save deployment-oriented pipeline artifacts to a directory.

        This interface writes the standard exported inference artifact layout:
        model weights plus the pipeline configuration file. It complements, but
        does not replace, the generic :class:`StateSaveLoadMixin` ``save`` /
        ``load`` APIs, which snapshot the in-memory pipeline object state.

        Args:
            directory (str): Target directory for the saved pipeline.
            inference_prefix (str, optional): Prefix for the pipeline config
                file. Default is ``"inference"``.
            model_prefix (str, optional): Prefix for model files passed to the
                model's save method. Default is ``"model"``.
            required_empty (bool, optional): If True, raises an error when the
                target directory is not empty. Default is True.
        """
        os.makedirs(directory, exist_ok=True)
        if required_empty and not is_empty_directory(directory):
            raise DirectoryNotEmptyError(f"{directory} is not empty!")

        self.model.save_model(
            directory=directory,
            model_prefix=model_prefix,
            required_empty=False,
        )
        with open(
            os.path.join(directory, f"{inference_prefix}.config.json"), "w"
        ) as fh:
            cfg_copy = self.cfg.model_copy()
            cfg_copy.model_cfg = None
            fh.write(cfg_copy.model_dump_json(indent=4))

    @staticmethod
    def load_pipeline(
        directory: str,
        inference_prefix: str = "inference",
        load_weights: bool = True,
        device: str | None = "cpu",
        strict: bool = True,
        device_map: str | DeviceMap | None = None,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ):
        """Load exported pipeline artifacts from disk or Hugging Face Hub.

        This method supports loading from a local path or a Hugging Face Hub
        repository. For Hub models, the supported URI format is
        ``hf://[<token>@][model/]<repo_id>[/<path>][@<revision>]``.

        This is the inverse of :meth:`save_pipeline` for exported inference
        artifacts. It is separate from the generic object-level
        :class:`StateSaveLoadMixin` ``load`` API.

        Examples:
            >>> InferencePipelineMixin.load_pipeline(
            ...     "hf://HorizonRobotics/Aux-Think"
            ... )
            >>> InferencePipelineMixin.load_pipeline(
            ...     "hf://HorizonRobotics/FineGrasp/finegrasp_pipeline"
            ... )
            >>> InferencePipelineMixin.load_pipeline(
            ...     "hf://your-name/private-repo"
            ... )

        The pipeline class is determined dynamically from the saved config,
        then model weights are optionally loaded into that instance.

        Args:
            directory (str): Local directory or Hugging Face Hub URI to load
                from.
            inference_prefix (str, optional): Prefix of the pipeline config
                file. Default is ``"inference"``.
            load_weights (bool, optional): Whether to load model weights. If
                False, the model is instantiated without loading saved weights.
                Default is True.
            device (str | None, optional): Device to map the model to. When
                None, weight loading may use ``device_map`` instead. Default is
                ``"cpu"``.
            strict (bool, optional): Whether to strictly enforce state-dict key
                matching during weight loading. Default is True.
            device_map (str | DeviceMap | None, optional): Device map to use
                when ``device`` is None. Default is None.
            model_prefix (str, optional): Prefix of the model files. Default is
                ``"model"``.
            load_impl (Literal["native", "accelerate"], optional): Backend
                used to load model weights. Default is ``"accelerate"``.

        Returns:
            InferencePipelineMixin: An initialized instance of the pipeline
                subclass defined in the saved configuration.
        """
        directory = abspath(resolve_hf_compatible_path(directory))

        with in_cwd(directory):
            cfg = load_from(
                f"{inference_prefix}.config.json",
                ensure_type=InferencePipelineMixinCfg,
            )
            cfg.update_model_cfg(f"{model_prefix}.config.json")
            if cfg.model_cfg is None:
                raise ValueError("The model configuration is missing.")
            pipeline: InferencePipelineMixin = cfg.class_type(cfg=cfg)  # type: ignore
            if load_weights:
                pipeline.model.load_weights(
                    directory=directory,
                    model_prefix=model_prefix,
                    strict=strict,
                    device=device,
                    device_map=device_map,
                    load_impl=load_impl,
                )

        return pipeline

    def reset(self, **kwargs) -> None:
        """Reset pipeline runtime state.

        This low-level mixin leaves reset semantics to concrete pipeline
        implementations. Standard orchard pipelines inherit the default
        no-op implementation from :class:`InferencePipeline`, while direct
        subclasses of this mixin must choose whether reset should be a no-op
        or should consume concrete episode-local metadata.

        Args:
            kwargs: Optional concrete pipeline-specific reset arguments.

        Raises:
            NotImplementedError: If the concrete pipeline does not provide a
                reset implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.reset() must be implemented by the "
            "concrete pipeline class."
        )

    def _get_state(self) -> State:
        """Get the state of the object for saving.

        The configuration is separated from the raw instance state for better
        readability of serialized pipeline artifacts.

        Returns:
            State: Serialized pipeline state.
        """
        ret = super()._get_state()
        ret.config = ret.state.pop("cfg", None)
        return ret

    def _set_state(self, state: State) -> None:
        """Set the state of the object from a serialized state object.

        Args:
            state (State): Serialized pipeline state produced by
                :meth:`_get_state`.
        """
        state.state["cfg"] = state.config
        state.config = None
        super()._set_state(state)
        self._setup(cfg=self.cfg, model=self.model)


InferencePipelineMixinType_co = TypeVar(
    "InferencePipelineMixinType_co",
    bound=InferencePipelineMixin,
    covariant=True,
)


class InferencePipelineMixinCfg(ClassConfig[InferencePipelineMixinType_co]):
    """Configuration class for an inference pipeline.

    This Pydantic-based config stores the pipeline class to instantiate and
    the model compatibility input required to construct or reload the
    pipeline.
    """

    model_cfg: PipelineModelCfg | None = None
    """Compatibility input for the model used in the pipeline.

    The field name is intentionally retained for caller compatibility. It may
    carry either a concrete :class:`TorchModuleCfg` or a
    :class:`~robo_orchard_lab.models.model_ref.TorchModelRef`.
    """

    def _get_structural_model_cfg(
        self,
    ) -> TorchModuleCfg[TorchModelMixin] | None:
        if self.model_cfg is None:
            return None
        if isinstance(self.model_cfg, TorchModelRef):
            return self.model_cfg.cfg
        structural_cfg: TorchModuleCfg[TorchModelMixin] = self.model_cfg
        return structural_cfg

    def build_model(self) -> TorchModelMixin:
        """Build or load the pipeline model from ``model_cfg``."""

        if self.model_cfg is None:
            raise ValueError("The model configuration is missing.")
        model: TorchModelMixin = self.model_cfg()
        return model

    def validate_model(self, model: TorchModelMixin) -> None:
        """Validate ``model_cfg`` against a runtime model.

        Runtime model binding keeps structural validation anchored on concrete
        comparable configs. A load-from-only ref has no comparable structural
        config, so this method validates its explicit runtime constraints but
        preserves the ref itself as the reconstruction source of truth.
        """

        if self.model_cfg is None:
            self.model_cfg = model.cfg
            return

        if isinstance(self.model_cfg, TorchModelRef):
            self.model_cfg.validate_runtime_model(model)

        structural_cfg = self._get_structural_model_cfg()
        if structural_cfg is None:
            # A load-from-only ref has no comparable structural config, but
            # it still carries the checkpoint source needed for reconstruction.
            return

        if structural_cfg is not model.cfg and structural_cfg != model.cfg:
            raise ValueError(
                "The provided model configuration in the pipeline "
                "differs from the configuration of the given model "
                "instance. You should set cfg.model_cfg to None or provide "
                "a compatible structural config/ref when model is provided."
            )

    def update_model_cfg(self, model_cfg_path: str):
        """Update the model configuration from a saved config file.

        If both the pipeline config and the standalone model config contain
        model metadata, the standalone model config takes precedence so the
        loaded pipeline matches the saved artifact layout.

        Args:
            model_cfg_path (str): Path to the saved model config file.
        """
        if self.model_cfg is not None and os.path.exists(model_cfg_path):
            logger.warning(
                f"Both the pipeline config and {model_cfg_path} "
                "contain a model configuration. The latter will be used."
            )
            self.model_cfg = load_from(
                model_cfg_path, ensure_type=TorchModuleCfg
            )
        if self.model_cfg is None:
            self.model_cfg = load_from(
                model_cfg_path, ensure_type=TorchModuleCfg
            )
