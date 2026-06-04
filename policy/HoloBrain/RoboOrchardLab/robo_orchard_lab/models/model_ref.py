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
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generic, Literal, TypeVar, cast

import torch
import transformers
from pydantic import (
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)
from robo_orchard_core.utils.config import (
    ClassType,
    Config,
    ConfigInstanceOf,
)
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from typing_extensions import Self

from robo_orchard_lab.models.torch_model import TorchModelMixin, TorchModuleCfg
from robo_orchard_lab.utils.huggingface import resolve_hf_compatible_path
from robo_orchard_lab.utils.path import abspath, in_cwd

__all__ = [
    "HFPretrainedModelRef",
    "TorchModelLoadConfig",
    "TorchModelRef",
]


ModelT = TypeVar("ModelT", covariant=True)
TorchModelT = TypeVar("TorchModelT", bound=TorchModelMixin, covariant=True)
HFModelT = TypeVar("HFModelT", bound=PreTrainedModel, covariant=True)
DeviceMapValue = int | str
DeviceMapInputValue = int | str | torch.device
BOOL_ADAPTER = TypeAdapter(bool)


def _normalize_ref_json_value(value: Any) -> Any:
    """Normalize supported ref values into JSON-compatible data."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_ref_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_ref_json_value(item) for item in value]
    raise TypeError(
        "Model refs only support JSON-serializable kwargs plus common "
        f"torch/path values. Unsupported value type: {type(value).__name__}."
    )


def _runtime_hf_dtype_kwarg_name() -> Literal["dtype", "torch_dtype"]:
    """Return the dtype kwarg name expected by the installed transformers."""

    version_tokens = transformers.__version__.split(".")
    parsed_tokens: list[int] = []
    for token in version_tokens[:2]:
        numeric_prefix = ""
        for char in token:
            if char.isdigit():
                numeric_prefix += char
            else:
                break
        if not numeric_prefix:
            return "torch_dtype"
        parsed_tokens.append(int(numeric_prefix))

    if len(parsed_tokens) < 2:
        return "torch_dtype"

    return "dtype" if tuple(parsed_tokens[:2]) >= (4, 56) else "torch_dtype"


def _normalize_hf_dtype_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``dtype`` / ``torch_dtype`` aliases for transformers."""

    normalized_kwargs = dict(kwargs)
    dtype = normalized_kwargs.pop("dtype", None)
    torch_dtype = normalized_kwargs.pop("torch_dtype", None)

    if dtype is None and torch_dtype is None:
        return normalized_kwargs

    if dtype is not None and torch_dtype is not None and dtype != torch_dtype:
        raise ValueError(
            "`dtype` and `torch_dtype` must match when both are provided."
        )

    normalized_kwargs[_runtime_hf_dtype_kwarg_name()] = (
        dtype if dtype is not None else torch_dtype
    )
    return normalized_kwargs


class LoadableModelRef(Config, Generic[ModelT], ABC):
    """Abstract base class for serializable model references."""

    @abstractmethod
    def resolve(self) -> ModelT:
        """Resolve the reference into a concrete model instance."""

    def __call__(self) -> ModelT:
        """Keep the existing config-call surface for caller compatibility."""

        return self.resolve()


class TorchModelLoadConfig(Config):
    """Checkpoint-loading parameters for :class:`TorchModelMixin` models."""

    directory: str
    load_weights: bool = True
    strict: bool = True
    device: str | None = "cpu"
    device_map: str | dict[str, DeviceMapValue] | None = None
    model_prefix: str = "model"
    load_impl: Literal["native", "accelerate"] = "accelerate"

    @field_validator("directory")
    @classmethod
    def _validate_directory(cls, directory: str) -> str:
        directory = directory.strip()
        if not directory:
            raise ValueError("`directory` must not be empty.")
        return directory

    @field_validator("device_map", mode="before")
    @classmethod
    def _normalize_device_map(
        cls,
        device_map: str | dict[str, DeviceMapInputValue] | None,
    ) -> str | dict[str, DeviceMapValue] | None:
        if not isinstance(device_map, dict):
            return device_map
        return {
            module_name: (
                str(device) if isinstance(device, torch.device) else device
            )
            for module_name, device in device_map.items()
        }

    def runtime_device_map(
        self,
    ) -> str | dict[str, int | str | torch.device] | None:
        """Return ``device_map`` in the widened loader-compatible shape."""

        if isinstance(self.device_map, dict):
            return cast(
                dict[str, int | str | torch.device],
                dict(self.device_map),
            )
        return self.device_map

    def resolve_directory(self) -> str:
        """Resolve ``directory`` into a local absolute checkpoint path."""

        return abspath(resolve_hf_compatible_path(self.directory))

    @contextmanager
    def resolved_directory_context(self):
        """Yield the resolved directory, entering it when it is local.

        This is useful for config-driven model construction that relies on
        files stored relative to the resolved checkpoint directory. Because
        this temporarily changes the process cwd, keep the context narrowly
        scoped to immediate model construction and do not treat it as
        thread-safe ambient state.
        """

        directory = self.resolve_directory()
        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Checkpoint directory {directory} does not exist."
            )
        if not os.path.isdir(directory):
            raise NotADirectoryError(
                f"Checkpoint path {directory} is not a directory."
            )
        with in_cwd(directory):
            yield directory


class TorchModelRef(LoadableModelRef[TorchModelT], Generic[TorchModelT]):
    """Serializable reference for building or loading Torch models.

    This ref supports cfg-only, load-only, and cfg-plus-load construction.
    Call :meth:`resolve` for the explicit API, or use :meth:`__call__` to
    preserve config-call compatibility. In cfg-plus-load mode, config
    construction runs with the process cwd temporarily set to
    ``load_from.directory`` so relative assets resolve from the checkpoint
    directory. cfg-only mode uses the ambient cwd instead.

    Example::

        ref = TorchModelRef(
            cfg=model_cfg,
            load_from=TorchModelLoadConfig(directory="/tmp/exported_model"),
        )
        model = ref.resolve()
        same_model = ref()
    """

    cfg: ConfigInstanceOf[TorchModuleCfg[TorchModelT]] | None = None
    load_from: TorchModelLoadConfig | None = None
    ensure_type: ClassType[TorchModelT] | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.cfg is None and self.load_from is None:
            raise ValueError(
                "At least one of `cfg` or `load_from` must be set."
            )
        return self

    def _validate_ensure_type(self, model: TorchModelMixin) -> None:
        if self.ensure_type is not None and not isinstance(
            model, self.ensure_type
        ):
            raise TypeError(
                "The resolved model does not match `ensure_type`. "
                f"Expected {self.ensure_type.__name__}, "
                f"got {type(model).__name__}."
            )

    def validate_runtime_model(self, model: TorchModelMixin) -> None:
        """Validate a runtime model against ref-side type constraints.

        Args:
            model (TorchModelMixin): Runtime model instance to validate.
        """

        self._validate_ensure_type(model)

    def resolve(self) -> TorchModelT:
        """Build or load the referenced model.

        When both ``cfg`` and ``load_from`` are set, this method builds the
        model from ``cfg`` first and then loads weights from
        ``load_from.directory``. When only ``load_from`` is set, it delegates
        to :meth:`TorchModelMixin.load_model`. When ``ensure_type`` is set, the
        resolved model must match that runtime type in either path.

        Returns:
            TorchModelT: The resolved model instance.
        """
        if self.cfg is not None:
            if self.load_from is not None:
                with self.load_from.resolved_directory_context() as directory:
                    model = self.cfg()
            else:
                directory = None
                model = self.cfg()

            if self.load_from is not None and self.load_from.load_weights:
                assert directory is not None
                model.load_weights(
                    directory=directory,
                    strict=self.load_from.strict,
                    device=self.load_from.device,
                    device_map=self.load_from.runtime_device_map(),
                    model_prefix=self.load_from.model_prefix,
                    load_impl=self.load_from.load_impl,
                )
            self.validate_runtime_model(model)
            return model

        assert self.load_from is not None
        directory = self.load_from.resolve_directory()
        model: TorchModelMixin = TorchModelMixin.load_model(
            directory=directory,
            load_weights=self.load_from.load_weights,
            strict=self.load_from.strict,
            device=self.load_from.device,
            device_map=self.load_from.runtime_device_map(),
            model_prefix=self.load_from.model_prefix,
            load_impl=self.load_from.load_impl,
        )
        self.validate_runtime_model(model)
        return cast(TorchModelT, model)


class HFPretrainedModelRef(LoadableModelRef[HFModelT], Generic[HFModelT]):
    """Serializable reference for Hugging Face pretrained models.

    ``path`` may be a local pretrained directory, a standard Hugging Face
    model id, or the repository's ``hf://...`` URI form. `load_kwargs` apply
    only to the weights-loading branch, and `build_kwargs` apply only to the
    config-only construction branch. Both kwargs surfaces accept either
    ``dtype`` or ``torch_dtype``; the ref normalizes that alias to match the
    installed transformers runtime before dispatch.

    Example::

        ref = HFPretrainedModelRef(
            class_type=SomeHFModel,
            path="/tmp/local_hf_model",
            load_weights=True,
        )
        model = ref.resolve()
    """

    class_type: ClassType[HFModelT]
    path: str
    load_weights: bool = True
    config_kwargs: dict[str, Any] = Field(default_factory=dict)
    load_kwargs: dict[str, Any] = Field(default_factory=dict)
    build_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_kwargs_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "model_kwargs" not in data:
            return data

        if "load_kwargs" in data or "build_kwargs" in data:
            raise ValueError(
                "`model_kwargs` can not be combined with `load_kwargs` or "
                "`build_kwargs`."
            )

        data = dict(data)
        legacy_model_kwargs = data.pop("model_kwargs")
        try:
            load_weights = BOOL_ADAPTER.validate_python(
                data.get("load_weights", True)
            )
        except ValidationError:
            load_weights = True
        target_key = "load_kwargs" if load_weights else "build_kwargs"
        data[target_key] = legacy_model_kwargs
        return data

    @field_validator("path")
    @classmethod
    def _validate_path(cls, path: str) -> str:
        path = path.strip()
        if not path:
            raise ValueError("`path` must not be empty.")
        return path

    @field_validator(
        "config_kwargs",
        "load_kwargs",
        "build_kwargs",
        mode="before",
    )
    @classmethod
    def _normalize_kwargs(
        cls, kwargs: dict[str, Any] | None
    ) -> dict[str, Any]:
        if kwargs is None:
            return {}
        return {
            str(key): _normalize_ref_json_value(value)
            for key, value in kwargs.items()
        }

    @model_validator(mode="after")
    def _validate_active_branch_kwargs(self) -> Self:
        if self.load_weights:
            if self.build_kwargs:
                raise ValueError(
                    "`build_kwargs` requires `load_weights=False`."
                )
        elif self.load_kwargs:
            raise ValueError("`load_kwargs` requires `load_weights=True`.")
        _normalize_hf_dtype_kwargs(self.load_kwargs)
        _normalize_hf_dtype_kwargs(self.build_kwargs)
        return self

    def resolve_path(self) -> str:
        """Resolve a local pretrained directory or ``hf://`` source.

        Returns an absolute local path for existing local resources and
        resolved ``hf://`` URIs. Otherwise returns the original string
        unchanged so standard Hugging Face model ids continue to flow through.
        Non-existing local-looking strings are therefore treated the same as
        model ids and may fail later inside Hugging Face loaders.
        """

        if self.path.startswith("hf://"):
            return abspath(resolve_hf_compatible_path(self.path))

        if os.path.exists(self.path):
            return abspath(self.path)

        return self.path

    def _load_config(self, resolved_path: str) -> PretrainedConfig:
        config_class = getattr(self.class_type, "config_class", None)
        from_pretrained = getattr(config_class, "from_pretrained", None)
        if callable(from_pretrained):
            return cast(
                PretrainedConfig,
                from_pretrained(resolved_path, **self.config_kwargs),
            )
        return AutoConfig.from_pretrained(
            resolved_path,
            **self.config_kwargs,
        )

    def _build_from_config(self, config: Any) -> HFModelT:
        build_kwargs = _normalize_hf_dtype_kwargs(self.build_kwargs)

        from_config = getattr(self.class_type, "from_config", None)
        if callable(from_config):
            return cast(
                HFModelT,
                from_config(config, **build_kwargs),
            )

        from_private_config = getattr(self.class_type, "_from_config", None)
        if callable(from_private_config):
            return cast(
                HFModelT,
                from_private_config(config, **build_kwargs),
            )

        raise TypeError(
            f"{self.class_type.__name__} does not support config-only "
            "construction via `from_config` or `_from_config`."
        )

    def resolve(self) -> HFModelT:
        """Build or load the referenced Hugging Face pretrained model.

        When ``load_weights`` is True, this method delegates to
        ``class_type.from_pretrained`` and forwards ``load_kwargs`` only to
        that branch. When ``config_kwargs`` is set, the config is loaded
        explicitly first so those overrides also apply to the weights-loading
        path. When ``load_weights`` is False, it loads config only and
        instantiates a fresh model with ``class_type.from_config`` when
        available, otherwise falling back to ``class_type._from_config`` and
        forwarding only ``build_kwargs``.

        Returns:
            HFModelT: The resolved Hugging Face model instance.
        """

        resolved_path = self.resolve_path()
        config = None
        if self.config_kwargs or not self.load_weights:
            config = self._load_config(resolved_path)

        if self.load_weights:
            load_kwargs = _normalize_hf_dtype_kwargs(self.load_kwargs)
            if config is not None:
                load_kwargs["config"] = config
            return self.class_type.from_pretrained(
                resolved_path,
                **load_kwargs,
            )

        assert config is not None
        return self._build_from_config(config)
