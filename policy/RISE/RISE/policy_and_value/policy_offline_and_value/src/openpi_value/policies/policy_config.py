import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi_value.models.model as _model
import openpi_value.policies.policy as _policy
import openpi_value.shared.download as download
from openpi_value.training import checkpoints as _checkpoints
from openpi_value.training import config as _config
import openpi_value.transforms as transforms
import copy

def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """

    repack_transforms = train_config.data.repack_transforms

    # * ------------ Remove actions if present
    new_transform_dict = copy.deepcopy(repack_transforms.inputs[0].structure)

    if "actions" in new_transform_dict:
        del new_transform_dict["actions"]

    repack_transforms = transforms.Group(
        inputs=[transforms.RepackTransform(new_transform_dict)],
    )
    # * ------------ action removed
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    # weight_path2 = pathlib.Path(checkpoint_dir) / "model.pt"
    weight_path2 = os.path.join(checkpoint_dir, "model.pt")
    
    # weight_path = weight_path if os.path.exists(weight_path) else str(weight_path2)

    # is_pytorch = os.path.exists(weight_path)
    weight_path = weight_path if os.path.exists(weight_path) else weight_path2
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
             
        try:
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir /  "assets", data_config.asset_id
            )
        except:
            norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir, data_config.asset_id
            )


    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
        
        # * Custom
        policy_seed=train_config.seed,  # * used for JAX rng initialization
    )
