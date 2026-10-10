"""Checkpoint loading for Focus-VLWA weights."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open

if TYPE_CHECKING:
    from focus_vlwa.model.focus_vlwa import FocusVLWA


_CHECKPOINT_PREFIXES = (
    ("paligemma_with_expert.paligemma.", "joint_experts.vision_language_model."),
    ("paligemma_with_expert.gemma_expert.", "joint_experts.action_expert."),
    ("paligemma_with_expert.gemma_wm_expert.", "joint_experts.world_model_expert."),
    ("action_in_proj.", "action_input_projection."),
    ("action_out_proj.", "action_output_projection."),
    ("wm_in_proj.", "world_model_input_projection."),
    ("wm_out_proj.", "world_model_output_projection."),
    ("time_mlp_in.", "time_mlp_input."),
    ("time_mlp_out.", "time_mlp_output."),
)


def _target_key(source_key: str) -> str:
    for old, new in _CHECKPOINT_PREFIXES:
        if source_key.startswith(old):
            return new + source_key[len(old):]
    return source_key


def load_checkpoint_weights(
    model: FocusVLWA, checkpoint: str | Path, *, init_world_model_from_action: bool = False
) -> None:
    """Load weights one tensor at a time to avoid a second full checkpoint copy in memory."""
    if init_world_model_from_action and not model.use_world_model:
        raise ValueError("Action-to-WM initialization requires use_world_model=True")
    checkpoint = Path(checkpoint)
    path = checkpoint / "model.safetensors" if checkpoint.is_dir() else checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    target_state = model.state_dict()
    loaded: set[str] = set()
    unexpected: list[str] = []
    device = str(next(model.parameters()).device)
    with safe_open(path, framework="pt", device=device) as handle, torch.no_grad():
        aliases = handle.metadata() or {}
        for source_key in handle.keys():
            target_key = _target_key(source_key)
            target = target_state.get(target_key)
            if target is None:
                unexpected.append(source_key)
                continue
            if target_key in loaded:
                raise ValueError(f"Duplicate checkpoint parameter: {target_key}")
            source = handle.get_tensor(source_key)
            if source.shape != target.shape:
                raise ValueError(
                    f"Checkpoint shape mismatch for {source_key}: source={tuple(source.shape)}, "
                    f"target={tuple(target.shape)}"
                )
            target.copy_(source.to(dtype=target.dtype))
            loaded.add(target_key)
            for alias_key, stored_key in aliases.items():
                if stored_key != source_key:
                    continue
                alias_target_key = _target_key(alias_key)
                alias_target = target_state.get(alias_target_key)
                if alias_target is None or alias_target_key in loaded:
                    continue
                if source.shape != alias_target.shape:
                    raise ValueError(f"Checkpoint alias shape mismatch: {alias_key}")
                alias_target.copy_(source.to(dtype=alias_target.dtype))
                loaded.add(alias_target_key)

    loaded_storage = {target_state[key].data_ptr() for key in loaded}
    missing = [
        key for key, value in target_state.items() if key not in loaded and value.data_ptr() not in loaded_storage
    ]
    if init_world_model_from_action:
        model.copy_action_expert_into_world_model()
        missing = [
            key
            for key in missing
            if not key.startswith("joint_experts.world_model_expert.")
            and not key.startswith("world_model_input_projection.")
            and not key.startswith("world_model_output_projection.")
        ]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    logging.info("Loaded Focus-VLWA weights from %s", path)
