# Copyright (c) 2026 Awomo-WAM Team. Apache-2.0.
"""Save the final training EMA in the adapter's inference checkpoint format."""

from pathlib import Path
import json
import inspect
import os
import shutil
import tempfile

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from awomo.awomo.factory import canonical_lora, file_sha256, read_checkpoint_config, validate_checkpoint_payload


def inference_config(config):
    data = config.data.dataloader.dataset
    model = config.action_patch
    vlm = model.vlm
    if data.norm_default_mode != "z-score":
        raise ValueError("The inference adapter requires z-score absolute joint normalization")
    return dict(
        model_type="awomo-0.5i-robodojo-sft", format_version=1, artifact_kind="sft",
        weights="ema", parameter_dtype="float32", action_dim=data.action_dim,
        state_dim=data.state_dim, action_horizon=data.num_frames - 1,
        action_indices=list(data.action_output_indices), action_scale=model.action_patch_scale,
        normalization="zscore_absolute_joint", gripper_source=data.action_gripper_source_semantics,
        gripper_model=data.action_gripper_target_semantics, extra_padding48="zero_before_transformer",
        inactive80="sigma_initial_noise_zero_clean", image_color_order=data.image_color_order,
        image_layout=data.image_layout, video_size=list(data.video_size),
        vl_image_size=list(vlm.image_size), vl_text_layout="prefix_v2",
        history_slots=vlm.history_num_slots, condition_layers=list(vlm.qwen_layers),
        summary_layers=list(vlm.summary_layers), vl_lora=dict(rank=vlm.lora.r, alpha=vlm.lora.alpha),
        scheduler=dict(shift=model.scheduler.shift, num_inference_steps=config.inference.num_inference_steps),
    )


def load_ema_snapshot(checkpoint, expected_state):
    """Read globally shaped FP32 EMA tensors on CPU without gathering GPU tensors."""
    reader = dcp.FileSystemReader(Path(checkpoint) / "ema")
    metadata = reader.read_metadata().state_dict_metadata
    if set(metadata) != set(expected_state):
        raise ValueError("EMA checkpoint keys do not match the training model")
    state = {}
    for name, item in metadata.items():
        if not isinstance(item, TensorStorageMetadata) or item.properties.dtype != torch.float32:
            raise ValueError(f"Expected an FP32 EMA tensor: {name}")
        if tuple(item.size) != tuple(expected_state[name].shape):
            raise ValueError(f"EMA tensor shape mismatch: {name}")
        state[name] = torch.empty(item.size, dtype=torch.float32, device="cpu")
    dcp.load(state, storage_reader=reader, no_dist=True)
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("EMA checkpoint contains non-finite parameters")
    return state


def save_inference_checkpoint(destination, model_state, dit_config, vl_encoder, config, *, allow_existing=False):
    """Atomically publish the final EMA, frozen VLM adapters and normalization."""
    destination = Path(destination)
    if destination.exists() and not allow_existing:
        raise FileExistsError(f"Refusing to overwrite inference checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    adapters = {canonical_lora(name): parameter.detach().cpu().clone()
                for name, parameter in vl_encoder.named_parameters() if "lora_" in name}
    if not adapters or any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                           for value in adapters.values()):
        raise ValueError("Expected finite FP32 VLM adapters")
    from diffusers import Flux2Transformer2DModel
    architecture_keys = inspect.signature(Flux2Transformer2DModel.__init__).parameters
    architecture = {key: value for key, value in dict(dit_config).items() if key in architecture_keys and key != "self"}
    if destination.exists():
        if read_checkpoint_config(destination) != inference_config(config):
            raise ValueError("Existing inference configuration does not match this run")
        if (destination / "dataset_stats.json").read_bytes() != Path(config.data.dataloader.dataset.pretrained_norm_stats).read_bytes():
            raise ValueError("Existing inference normalization does not match this run")
        payload = torch.load(destination / "model.pt", map_location="cpu", weights_only=True, mmap=True)
        validate_checkpoint_payload(payload)
        if payload["dit_config"] != architecture:
            raise ValueError("Existing inference architecture does not match this model")
        for key, expected in (("dit", model_state), ("qwen_lora", adapters)):
            saved = payload[key]
            if set(saved) != set(expected) or any(
                saved[name].dtype != value.dtype or saved[name].shape != value.shape
                or not torch.equal(saved[name], value) for name, value in expected.items()
            ):
                raise ValueError(f"Existing inference {key} tensors do not match this checkpoint")
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".inference-", dir=destination.parent))
    try:
        torch.save(dict(dit=model_state, dit_config=architecture, qwen_lora=adapters), staging / "model.pt")
        (staging / "awomo_config.json").write_text(json.dumps(inference_config(config), indent=2) + "\n")
        shutil.copyfile(config.data.dataloader.dataset.pretrained_norm_stats, staging / "dataset_stats.json")
        (staging / "SHA256SUMS").write_text("".join(
            f"{file_sha256(staging / name)}  {name}\n"
            for name in ("model.pt", "awomo_config.json", "dataset_stats.json")
        ))
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
