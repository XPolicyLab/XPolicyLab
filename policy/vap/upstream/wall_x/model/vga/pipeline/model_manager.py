"""
ModelManager - Local implementation for loading and managing Wan Video models
Completely removes dependency on diffsynth
"""

import os
import torch
from typing import List, Optional, Tuple, Dict, Any

from wall_x.utils.io import (
    load_state_dict,
    hash_state_dict_keys,
    init_weights_on_device,
)
from wall_x.model.vga.modeling.wan_dit import WanModel
from wall_x.model.vga.modeling.wan_text_encoder import WanTextEncoder
from wall_x.model.vga.modeling.wan_image_encoder import WanImageEncoder
from wall_x.model.vga.modeling.wan_vae import WanVideoVAE, WanVideoVAE38


# Wan Video model configurations
# Format: (keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource)
WAN_VIDEO_MODEL_CONFIGS = [
    # WanModel (DiT) - multiple configurations
    (
        None,
        "9269f8db9040a9d860eaca435be61814",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "aafcfd9672c3a2456dc46e1cb6e52c70",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "6bfcfb3b342cb286ce886889d519a77e",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "6d6ccde6845b95ad9114ab993d917893",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "349723183fc063b2bfc10bb2835cf677",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "efa44cddf936c70abd0ea28b6cbe946c",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "3ef3b1f8e1dab83d5b71fd7b617f859f",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "70ddad9d3a133785da5ea371aae09504",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "26bde73488a92e64cc20b0a7485b9e5b",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "ac6a5aa74f4a0aab6f64eb9a72f19901",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "b61c605c2adbd23124d152ed28e049ae",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "1f5ab7703c6fc803fdded85ff040c316",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "5b013604280dd715f8457c6ed6d6a626",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "2267d489f0ceb9f21836532952852ee5",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "47dbeab5e560db3180adf51dc0232fb1",
        ["wan_video_dit"],
        [WanModel],
        "civitai",
    ),
    (
        None,
        "cb104773c6c2cb6df4f9529ad5c60d0b",
        ["wan_video_dit"],
        [WanModel],
        "diffusers",
    ),
    # WanTextEncoder
    (
        None,
        "9c8818c2cbea55eca56c7b447df170da",
        ["wan_video_text_encoder"],
        [WanTextEncoder],
        "civitai",
    ),
    # Qwen3.5: load via WanPrompter/qwen35_path (Qwen35TextEncoderAdapter), not ModelManager.fetch_model.
    # WanImageEncoder
    (
        None,
        "5941c53e207d62f20f9025686193c40b",
        ["wan_video_image_encoder"],
        [WanImageEncoder],
        "civitai",
    ),
    # WanVideoVAE
    (
        None,
        "1378ea763357eea97acdef78e65d6d96",
        ["wan_video_vae"],
        [WanVideoVAE],
        "civitai",
    ),
    (
        None,
        "ccc42284ea13e1ad04693284c7a09be6",
        ["wan_video_vae"],
        [WanVideoVAE],
        "civitai",
    ),
    # WanVideoVAE38 (Wan2.2-TI2V-5B VAE, 48 latent dims)
    (
        None,
        "e1de6c02cdac79f8b739f4d3698cd216",
        ["wan_video_vae"],
        [WanVideoVAE38],
        "civitai",
    ),
]


def load_model_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    model_names: List[str],
    model_classes: List[type],
    model_resource: str,
    torch_dtype: torch.dtype,
    device: str,
    model_init_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], List[torch.nn.Module]]:
    """
    Load model from state_dict

    Args:
        state_dict: Model weight dictionary
        model_names: List of model names
        model_classes: List of model classes
        model_resource: Model source ("civitai" or "diffusers")
        torch_dtype: Data type
        device: Target device
        model_init_kwargs: Model initialization parameters

    Returns:
        (loaded_model_names, loaded_models): Loaded model names and model instances
    """
    loaded_model_names, loaded_models = [], []

    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")

        # Get state_dict converter
        state_dict_converter = model_class.state_dict_converter()

        # Convert state_dict based on source
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        else:
            state_dict_results = state_dict

        # Process return value
        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(
                f"        This model is initialized with extra kwargs: {extra_kwargs}"
            )
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}

        # Merge externally passed initialization parameters
        if model_init_kwargs and model_name in model_init_kwargs:
            extra_kwargs.update(model_init_kwargs[model_name])
            print(f"        Merged external kwargs: {model_init_kwargs[model_name]}")

        # Process data type
        final_dtype = (
            torch.float32
            if extra_kwargs.get("upcast_to_float32", False)
            else torch_dtype
        )

        # Create model instance
        with init_weights_on_device():
            model = model_class(**extra_kwargs)

        if hasattr(model, "eval"):
            model = model.eval()

        # Special handling: WanModel depth_head
        if (
            model_class.__name__ == "WanModel"
            and hasattr(model, "depth_head")
            and model.depth_head is not None
        ):
            has_depth_head_weights = any(
                k.startswith("depth_head.") for k in model_state_dict.keys()
            )
            if not has_depth_head_weights:
                print(
                    "        depth_head not found in checkpoint, will initialize from head weights"
                )
                model = model.to_empty(device=device)
                model.load_state_dict(model_state_dict, strict=False)
                model = model.to(dtype=final_dtype)
                # Copy weights from head to depth_head
                model.depth_head.modulation.data.copy_(model.head.modulation.data)
                model.depth_head.head.weight.data.copy_(model.head.head.weight.data)
                model.depth_head.head.bias.data.copy_(model.head.head.bias.data)
            else:
                model.load_state_dict(model_state_dict, assign=True)
                model = model.to(dtype=final_dtype, device=device)
        else:
            model.load_state_dict(model_state_dict, assign=True)
            model = model.to(dtype=final_dtype, device=device)

        loaded_model_names.append(model_name)
        loaded_models.append(model)

    return loaded_model_names, loaded_models


class ModelDetector:
    """Model detector - identifies model type based on state_dict key hash"""

    def __init__(self, model_configs: List[tuple]):
        self.keys_hash_with_shape_dict = {}
        self.keys_hash_dict = {}

        for config in model_configs:
            (
                keys_hash,
                keys_hash_with_shape,
                model_names,
                model_classes,
                model_resource,
            ) = config
            self.keys_hash_with_shape_dict[keys_hash_with_shape] = (
                model_names,
                model_classes,
                model_resource,
            )
            if keys_hash is not None:
                self.keys_hash_dict[keys_hash] = (
                    model_names,
                    model_classes,
                    model_resource,
                )

    def match(self, file_path: str, state_dict: Dict[str, torch.Tensor]) -> bool:
        """Check if it matches a known model configuration"""
        if isinstance(file_path, str) and os.path.isdir(file_path):
            print(f"    {file_path} is a directory, skipping model detection.")
            return False

        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True

        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            return True

        return False

    def load(
        self,
        file_path: str,
        state_dict: Dict[str, torch.Tensor],
        device: str,
        torch_dtype: torch.dtype,
        model_init_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> Tuple[List[str], List[torch.nn.Module]]:
        """Load model"""
        if len(state_dict) == 0:
            state_dict = load_state_dict(file_path)

        # Prefer hash matching with shape
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[
                keys_hash_with_shape
            ]
            return load_model_from_state_dict(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
                model_init_kwargs,
            )

        # Fallback to hash matching without shape
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            return load_model_from_state_dict(
                state_dict,
                model_names,
                model_classes,
                model_resource,
                torch_dtype,
                device,
                model_init_kwargs,
            )

        return [], []


class ModelManager:
    """
    Model manager - loads and manages Wan Video models

    Local implementation, completely removes dependency on diffsynth.
    Supports the following model types:
    - wan_video_dit (WanModel)
    - wan_video_text_encoder (WanTextEncoder)
    - wan_video_image_encoder (WanImageEncoder)
    - wan_video_vae (WanVideoVAE)
    """

    def __init__(
        self,
        torch_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model: List[torch.nn.Module] = []
        self.model_path: List[str] = []
        self.model_name: List[str] = []

        # Initialize model detector
        self.model_detector = ModelDetector(WAN_VIDEO_MODEL_CONFIGS)

    def load_model(
        self,
        file_path: str,
        model_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        model_init_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Load model

        Args:
            file_path: Model file path
            model_names: Model names to load (optional)
            device: Target device
            torch_dtype: Data type
            model_init_kwargs: Model initialization parameters
        """
        print(f"Loading models from: {file_path}")

        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype

        # Load state_dict
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            print(f"    File not found or is a directory: {file_path}")
            return

        # Try to detect and load model
        if self.model_detector.match(file_path, state_dict):
            loaded_names, loaded_models = self.model_detector.load(
                file_path,
                state_dict,
                device=device,
                torch_dtype=torch_dtype,
                model_init_kwargs=model_init_kwargs,
            )

            for name, model in zip(loaded_names, loaded_models):
                self.model.append(model)
                self.model_path.append(file_path)
                self.model_name.append(name)

            print(f"    The following models are loaded: {loaded_names}.")
        else:
            # Calculate hash for debugging
            keys_hash = hash_state_dict_keys(state_dict, with_shape=True)
            print(f"    Cannot detect model type. Hash: {keys_hash}")
            print("    No models loaded.")
            raise ValueError(f"Unrecognized model format for file: {file_path}")

    def load_models(
        self,
        file_path_list: List[str],
        model_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Batch load models"""
        for file_path in file_path_list:
            self.load_model(
                file_path, model_names, device=device, torch_dtype=torch_dtype
            )

    def fetch_model(
        self,
        model_name: str,
        file_path: Optional[str] = None,
        require_model_path: bool = False,
        index: Optional[int] = None,
    ):
        """
        Get loaded model

        Args:
            model_name: Model name
            file_path: Specify model file path (optional)
            require_model_path: Whether to also return model path
            index: When multiple models with same name exist, return first N

        Returns:
            model or (model, path) or None
        """
        fetched_models = []
        fetched_model_paths = []

        for model, model_path, name in zip(
            self.model, self.model_path, self.model_name
        ):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == name:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)

        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None

        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        else:
            if index is None:
                model = fetched_models[0]
                path = fetched_model_paths[0]
                print(
                    f"More than one {model_name} models loaded: {fetched_model_paths}. Using from {fetched_model_paths[0]}."
                )
            elif isinstance(index, int):
                model = fetched_models[:index]
                path = fetched_model_paths[:index]
                print(
                    f"More than one {model_name} models loaded: {fetched_model_paths}. Using first {index}."
                )
            else:
                model = fetched_models
                path = fetched_model_paths
                print(
                    f"More than one {model_name} models loaded: {fetched_model_paths}. Using all."
                )

        if require_model_path:
            return model, path
        else:
            return model

    def to(self, device: str) -> None:
        """Move all models to specified device"""
        for model in self.model:
            model.to(device)


__all__ = ["ModelManager"]
