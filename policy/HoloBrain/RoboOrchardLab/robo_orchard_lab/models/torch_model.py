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
import logging
import os
from typing import Any, Literal, TypeVar, overload

import torch
from accelerate import Accelerator
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassInitFromConfigMixin,
    ClassType_co,  # noqa: F401
    load_config_class,
)
from safetensors.torch import (
    load_model as safetensors_load_model,
    save_model as safetensors_save_model,
)
from transformers.modeling_utils import (
    get_parameter_device,
    get_parameter_dtype,
)
from typing_extensions import Self, deprecated

from robo_orchard_lab.utils.huggingface import resolve_hf_compatible_path
from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    abspath,
    in_cwd,
    is_empty_directory,
)
from robo_orchard_lab.utils.state import CustomizedSaveLoadMixin

__all__ = [
    "TorchModuleCfg",
    "ModelMixin",
    "TorchModelMixin",
    "ClassType_co",
    "TorchModuleCfgType_co",
]

logger = logging.getLogger(__name__)

TorchNNModuleTypeCo = TypeVar(
    "TorchNNModuleTypeCo", bound=torch.nn.Module, covariant=True
)


class TorchModuleCfg(ClassConfig[TorchNNModuleTypeCo]):
    """Configuration class for PyTorch nn.Module.

    This class extends `ClassConfig` to specifically handle configurations
    for PyTorch modules, ensuring type safety for the module class.
    """

    pass


TorchModuleCfgType_co = TypeVar(
    "TorchModuleCfgType_co", bound=TorchModuleCfg, covariant=True
)


def _set_nested_attr(obj: Any, names: list[str], value: Any):
    """Sets a nested attribute on an object.

    Args:
        obj: The target object.
        names: A list of attribute names representing the path.
        value: The value to set on the final attribute.

    Example:
        >>> _set_nested_attr(model, "encoder.layer.weight".split("."), tensor)
        equivalent to `model.encoder.layer.weight = tensor`.

    """
    for name in names[:-1]:
        obj = getattr(obj, name)
    setattr(obj, names[-1], value)


class TorchModelMixin(
    torch.nn.Module, CustomizedSaveLoadMixin, ClassInitFromConfigMixin
):
    """A mixin class for PyTorch `nn.Module` providing model saving and loading utilities.

    This mixin standardizes how models are configured, saved, and loaded,
    integrating with a configuration system and supporting `safetensors`
    for model weights.
    """  # noqa: E501

    def __init__(self, cfg: TorchModuleCfg):
        """Initializes the ModelMixin.

        Args:
            cfg (TorchModuleCfg): The configuration object for the model.
        """
        super().__init__()
        self.cfg = cfg

        self._accelerate_model_id: int = -1

    @property
    def device(self) -> torch.device:
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        return get_parameter_dtype(self)  # type: ignore

    @property
    def accelerate_model_id(self) -> int:
        """The model index of the prepared model in the 'accelerator' instance.

        This should be set when the model is prepared with
        `accelerator.prepare()`.

        If not available, it returns -1.

        """
        return self._accelerate_model_id

    @accelerate_model_id.setter
    def accelerate_model_id(self, value: int):
        self._accelerate_model_id = value

    def accelerator_register_all_hooks(
        self,
        accelerator: Accelerator,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Register all necessary hooks to the given Hugging Face Accelerator.

        Args:
            accelerator (Accelerator): The Hugging Face Accelerator instance.
        """

        if accelerator.is_main_process is False:
            logger.info("Not the main process, skip registering hooks.")
            return []

        model_id = self.accelerate_model_id
        if model_id < 0:
            raise ValueError(
                "Model's accelerate_model_id is not set. "
                "Ensure accelerate_model_id is set when preparing the model "
                "with `accelerator.prepare()`."
            )

        for hook in accelerator._save_model_state_pre_hook.keys():
            if hook == self.accelerator_save_state_pre_hook:
                logger.warning(
                    f"accelerator_save_state_pre_hook of {self}"
                    " is already registered. Skip registering again."
                )
                return []
        return [
            accelerator.register_save_state_pre_hook(
                self.accelerator_save_state_pre_hook
            )
        ]

    def accelerator_save_state_pre_hook(
        self,
        models: list[torch.nn.Module],
        weights: list[dict[str, torch.Tensor]],
        output_dir: str,
    ):
        """Pre-hook for Hugging Face Accelerate to save model configurations.

        This static method is designed to be registered with Accelerate's
        `save_state` mechanism. It iterates through the provided models and,
        if a model is an instance of `ModelMixin`, saves its configuration
        to a JSON file in the `output_dir`.

        The configuration is saved as `model_{id}.config.json` where `{id}`
        corresponds to the model's `accelerate_model_id`. If there is only one
        model, it is saved as `model.config.json`.

        Note:
            This hook only saves the configuration. The model weights (`state_dict`)
            are expected to be saved by the Accelerate framework itself.

        Args:
            models: A list of `torch.nn.Module` instances to process.
            weights: A list of model state dictionaries. This argument is part
                of the Accelerate hook signature but is not directly used by
                this method as Accelerate handles weight saving.
            output_dir: The directory where the configuration files will be saved.
        """  # noqa: E501

        model_id = self.accelerate_model_id

        if len(models) == 1:
            filename = "model.config.json"
        else:
            filename = f"model_{model_id}.config.json"
        logger.info(f"Saving model config to {filename}.")
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    @overload
    def save_model(
        self,
        directory: str,
        model_prefix: str = "model",
        allow_shared_tensor: None = None,
        required_empty: bool = True,
        accelerator: Accelerator | None = None,
    ): ...

    @overload
    @deprecated(
        "The `allow_shared_tensor` argument is deprecated and will be "
        "removed in future versions. The shared tensor handling is "
        "always enabled."
    )
    def save_model(
        self,
        directory: str,
        model_prefix: str = "model",
        allow_shared_tensor: bool = False,
        required_empty: bool = True,
        accelerator: Accelerator | None = None,
    ): ...

    def save_model(
        self,
        directory: str,
        model_prefix: str = "model",
        allow_shared_tensor: bool | None = None,
        required_empty: bool = True,
        accelerator: Accelerator | None = None,
    ):
        """Saves the model's config and weights to a directory.

        This method saves the model's configuration to `{model_prefix}.config.json`
        and its weights to `{model_prefix}.safetensors`.

        If an `accelerator` instance is provided, it uses the accelerator's
        `save_model` method to save the model, which is useful for distributed
        training scenarios. In this case, `model_prefix` is ignored.

        Args:
            directory: The path to the directory for saving the model.
            model_prefix: The file prefix for the config and weights files.
                If None and `accelerator` is not provided, defaults to "model".
                Ignored if `accelerator` is provided.
            allow_shared_tensor: If True, enables the logic to handle
                tied weights by saving a de-duplicated state dict and a
                key-sharing map. This field is deprecated and will be
                removed in future versions. The shared tensor handling
                is always enabled.
            required_empty (bool): If True, raises an error if the target
                directory is not empty. Defaults to True.
            accerator: An optional `Accelerator` instance from Hugging Face
                Accelerate. If provided, the model will be saved using the
                accelerator's `save_model` method.

        Raises:
            DirectoryNotEmptyError: If the target directory already exists and
                is not empty.
        """  # noqa: E501

        os.makedirs(directory, exist_ok=True)
        if required_empty and not is_empty_directory(directory):
            raise DirectoryNotEmptyError(f"{directory} is not empty!")

        if allow_shared_tensor is not None:
            logger.warning(
                "The `allow_shared_tensor` argument is deprecated and will be "
                "removed in future versions. The shared tensor handling is "
                "always enabled."
            )

        if accelerator is not None:
            # save model will be:
            #   directory/model.safetensors
            #   or directory/model-00001-of-00005.safetensors ...
            if model_prefix is not None and model_prefix != "model":
                logger.warning(
                    "model_prefix is ignored when saving with accelerator. "
                    "When using accelerator, the model prefix is "
                    "always 'model'."
                )
            accelerator.save_model(self, save_directory=directory)
        else:
            assert model_prefix is not None
            weights_path = os.path.join(
                directory, f"{model_prefix}.safetensors"
            )
            safetensors_save_model(
                self, weights_path, metadata={"format": "pt"}
            )

        config_path = os.path.join(directory, f"{model_prefix}.config.json")

        with open(config_path, "w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    @classmethod
    def load_model(
        cls: type[Self],
        directory: str,
        load_weights: bool = True,
        strict: bool = True,
        device: str | None = "cpu",
        device_map: str | dict[str, int | str | torch.device] | None = None,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
        overwrite_cfg_class_type: bool = False,
    ) -> Self | TorchModelMixin:
        """Loads a model from a local directory or the Hugging Face Hub.

        This class method services like `load_pretrained` methods in huggingface
        transformers or other similar libraries.

        It supports loading from a local path or a Hugging Face Hub repository.
        For Hub models, a URI format is used:
        ``hf://[<token>@][model/]<repo_id>[/<path>][@<revision>]``

        .. code-block:: text

            Public model: `hf://HorizonRobotics/BIP3D_Tiny_Det`

            Public model directory: `hf://HorizonRobotics/FineGrasp/finegrasp_pipeline`

            Private model: `hf://your-name/private-repo`

        This method first loads the model's configuration from a JSON file
        (e.g., "model.config.json") found in the given directory. It then
        instantiates the model using this configuration. The instantiation
        occurs within a context where the current working directory is set to
        the specified directory.

        If `load_weights` is True, the method proceeds to load the model's
        weights (state dictionary) from a ".safetensors" file (e.g.,
        "model.safetensors") located in the same directory.

        Args:
            directory (str): The path to the directory containing the model's
                configuration file and state dictionary file.
            device (str|None, optional): The device (e.g., "cpu", "cuda:0")
                onto which the model's state dictionary should be loaded.
                Passed to `load_file` for loading the safetensors file.
                Defaults to "cpu".
            device_map (str | dict[str, int | str | torch.device] | None, optional):
                A device map to specify how to distribute the model's
                layers across multiple devices. This parameter is only used
                when `load_impl` is "accelerate". `device` and `device_map`
                cannot be both specified. Defaults to None.
            strict (bool, optional): Whether to strictly enforce that the keys
                in `state_dict` match the keys returned by this module's
                `state_dict()` function. Passed to `model.load_state_dict()`.
                Defaults to True.
            model_prefix (str, optional): The prefix for the configuration
                and state dictionary files. For example, if "model", files
                will be sought as "model.config.json" and "model.safetensors".
                Defaults to "model".
            load_impl (Literal["native", "accelerate"], optional): The
                implementation to use for loading the model weights:

                  - "native": Uses the native PyTorch loading mechanism via
                    `safetensors.torch.load_file`.
                  - "accelerate": Uses Hugging Face Accelerate's
                    `load_checkpoint_and_dispatch`, which supports
                    advanced features like device mapping and offloading.

                Defaults to "accelerate".
            overwrite_cfg_class_type (bool, optional): If True, overwrites the
                `cfg.class_type` loaded from the configuration file with
                `cls`, even if they differ. This is useful when you want to
                load a model configuration that was originally saved with
                a different class type. Defaults to False.

        Returns:
            torch.nn.Module: An instance of the model (typed as "ModelMixin" or a subclass),
                initialized from the configuration and optionally with weights loaded.
                The returned model is of the type specified by `cls` if
                `overwrite_cfg_class_type` is True. Otherwise, it is of the type
                specified in the configuration file.

        Raises:
            FileNotFoundError: If the specified `directory` does not exist,
                or if the configuration file (`{model_prefix}.config.json`)
                or the state dictionary file (`{model_prefix}.safetensors}`,
                when `load_weights` is True) is not found in the directory.
            ValueError: If the Hugging Face Hub URI is invalid.
        """  # noqa: E501

        directory = abspath(resolve_hf_compatible_path(directory))

        if not os.path.exists(directory):
            raise FileNotFoundError(f"checkpoint {directory} does not exists!")

        config_file = os.path.join(directory, f"{model_prefix}.config.json")

        with open(config_file, "r") as f:
            cfg: TorchModuleCfg = load_config_class(f.read())  # type: ignore

        if cfg.class_type != cls:
            if overwrite_cfg_class_type:
                logger.warning(
                    f"Overwriting cfg.class_type from {cfg.class_type} "
                    f"to {cls} because overwrite_cfg_class_type=True. "
                    "Make sure this is intended. "
                )
                cfg.class_type = cls
            else:
                if cls != TorchModelMixin:
                    raise ValueError(
                        f"cfg.class_type {cfg.class_type} does not match "
                        f"the requested class type {cls}. Only when cls is "
                        f"`TorchModelMixin`, cfg.class_type is allowed to be "
                        f"different."
                    )

        with in_cwd(directory):
            model: Self = cfg()

        if load_weights:
            model.load_weights(
                directory=directory,
                strict=strict,
                device=device,
                device_map=device_map,
                model_prefix=model_prefix,
                load_impl=load_impl,
            )
        return model

    def load_weights(
        self,
        directory: str,
        device: str | None = "cpu",
        device_map: str | dict[str, int | str | torch.device] | None = None,
        strict: bool = True,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ):
        """Loads pretrained weights from a directory.

        Unlike `load_model`, this method only loads the model's weights
        from a specified directory into the existing model instance. It does
        not modify the model's architecture or configuration.

        Args:
            directory (str): The path to the directory containing the model's
                configuration file and pretrained weights file.
            device (str|None, optional): The device (e.g., "cpu", "cuda:0")
                onto which the model's state dictionary should be loaded.
                Passed to `load_file` for loading the safetensors file.
                Defaults to "cpu".
            device_map (str | dict[str, int | str | torch.device] | None, optional):
                A device map to specify how to distribute the model's
                layers across multiple devices. This parameter is only used
                when `load_impl` is "accelerate". `device` and `device_map`
                cannot be both specified. Defaults to None.
            strict (bool, optional): Whether to strictly enforce that the keys
                in `state_dict` match the keys returned by this module's
                `state_dict()` function. Passed to `model.load_state_dict()`.
                Defaults to True.
            model_prefix (str, optional): The prefix for the configuration
                and weights files. For example, if prefix is "model", files
                will be sought as "model.config.json" and "model.safetensors".
                Defaults to "model".
            load_impl (Literal["native", "accelerate"], optional): The
                implementation to use for loading the model weights:

                  - "native": Uses the native PyTorch loading mechanism via
                    `safetensors.torch.load_file`.
                  - "accelerate": Uses Hugging Face Accelerate's
                    `load_checkpoint_and_dispatch`, which supports
                    advanced features like device mapping and offloading.

                Defaults to "accelerate".

        """  # noqa: E501

        if load_impl == "native":
            if device_map is not None:
                raise ValueError(
                    "device_map is not supported when load_impl is 'native'."
                )
            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"{ckpt_path} does not exists!")
            missing, unexpected = safetensors_load_model(
                self, filename=ckpt_path, strict=strict
            )
            if len(missing) > 0:
                logger.warning(
                    f"Some weights are missing when loading state_dict from "
                    f"{ckpt_path}: {missing}"
                )
            if len(unexpected) > 0:
                logger.warning(
                    f"Some unexpected weights are found when "
                    f"loading state_dict from {ckpt_path}: {unexpected}"
                )
        elif load_impl == "accelerate":
            from accelerate import load_checkpoint_and_dispatch

            if device is not None and device_map is not None:
                raise ValueError(
                    "Only one of device or device_map can be specified."
                )
            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if os.path.exists(ckpt_path):
                # if the safetensors exists, `load_checkpoint_and_dispatch`
                # is compatible with the native implementation.
                load_checkpoint_and_dispatch(
                    self, ckpt_path, strict=strict, device_map=device_map
                )
            else:
                load_checkpoint_and_dispatch(
                    self, directory, strict=strict, device_map=device_map
                )

        else:
            raise ValueError(
                f"Invalid load_impl: {load_impl}, "
                f"expected 'native' or 'accelerate'."
            )
        # safe to call this because for accelerate, either device or
        # device_map can be None.
        if device is not None:
            self.to(device=device)

    def _save_impl(self, path: str, protocol: Any):
        """Serializes the model state to the given path.

        This method implements `save` from `StateSaveLoadMixin` by
        calling `save_model`.
        """
        self.save_model(directory=path, model_prefix="model")

    @classmethod
    def load(cls, path: str) -> Self | TorchModelMixin:
        """Deserializes the model state from the given path.

        This method implements `load` from `StateSaveLoadMixin` by
        calling `load_model`.
        """
        return cls.load_model(directory=path, model_prefix="model")


ModelMixin = TorchModelMixin
