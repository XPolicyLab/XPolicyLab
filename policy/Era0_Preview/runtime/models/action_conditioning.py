"""Shared validity-mask handling for action-transformer conditions."""

from __future__ import annotations

import os

import torch


def should_reapply_preprocessed_condition_masks(vlm_backend: str) -> bool:
    """Keep Florence's legacy mask timing unless a run explicitly opts in."""
    raw = str(
        os.environ.get(
            "XVLA_REAPPLY_PREPROCESSED_CONDITION_MASKS",
            "auto",
        )
    ).strip().lower()
    if raw in {"", "auto"}:
        return str(vlm_backend).strip().lower() == "qwen3_5"
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "XVLA_REAPPLY_PREPROCESSED_CONDITION_MASKS must be auto, 0, or 1; "
        f"got {raw!r}."
    )


def normalize_optional_mask(
    mask: torch.Tensor | None,
    *,
    ref: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    """Validate a raw-space mask against its source tensor."""
    if mask is None:
        return None
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)
    mask = mask.to(device=ref.device, dtype=ref.dtype)
    if mask.shape != ref.shape:
        raise ValueError(f"{name} shape mismatch: got={tuple(mask.shape)} expected={tuple(ref.shape)}")
    return mask


def expand_optional_mask_for_preprocessed_tensor(
    mask: torch.Tensor | None,
    *,
    ref: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    """Align a raw-space validity mask with a right-padded model-space tensor."""
    if mask is None:
        return None
    mask = mask.to(device=ref.device, dtype=ref.dtype)
    if mask.shape == ref.shape:
        return mask
    if (
        mask.dim() == ref.dim()
        and mask.shape[:-1] == ref.shape[:-1]
        and mask.shape[-1] < ref.shape[-1]
    ):
        expanded = ref.new_zeros(ref.shape)
        expanded[..., :mask.shape[-1]] = mask
        return expanded
    raise ValueError(
        f"{name} cannot align with preprocessed tensor: "
        f"mask={tuple(mask.shape)} ref={tuple(ref.shape)}"
    )


def apply_preprocessed_condition_masks(
    *,
    proprio: torch.Tensor,
    action: torch.Tensor,
    proprio_valid_mask: torch.Tensor | None,
    action_valid_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Reapply raw validity after action-space preprocessing.

    Some action spaces normalize valid channels. A raw zero in a masked channel
    is generally not zero after `(x - mean) / std`, so masking only before
    preprocessing makes a missing condition visible to the action transformer.
    """
    proprio_model_mask = expand_optional_mask_for_preprocessed_tensor(
        proprio_valid_mask,
        ref=proprio,
        name="proprio_valid_mask",
    )
    action_model_mask = expand_optional_mask_for_preprocessed_tensor(
        action_valid_mask,
        ref=action,
        name="action_valid_mask",
    )
    if proprio_model_mask is not None:
        proprio = proprio * proprio_model_mask
    if action_model_mask is not None:
        action = action * action_model_mask
    return proprio, action, proprio_model_mask, action_model_mask
