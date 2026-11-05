"""Typed tensors passed to the Focus-VLWA model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FocusVLWAObservation:
    """A batched model observation."""

    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    state: torch.Tensor
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None
    action_mask: torch.Tensor | None = None
    world_state: torch.Tensor | None = None
    world_state_mask: torch.Tensor | None = None
    event_action: torch.Tensor | None = None
    event_action_mask: torch.Tensor | None = None
    history_images: torch.Tensor | None = None
    history_mask: torch.Tensor | None = None
