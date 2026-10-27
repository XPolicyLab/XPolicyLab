"""Temporal padding helpers shared by MachEmbodiedUnifiedModel data and loss code."""

from __future__ import annotations

import torch


class TemporalMaskError(ValueError):
    """Raised when dataset camera/time masks violate a pipeline invariant."""


def future_latent_token_valid_mask(
    frame_is_pad: torch.Tensor,
    temporal_factor: int,
    latent_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Map frame padding to flattened future-latent token validity.

    Wan's causal temporal layout keeps frame zero as the conditioning latent and
    compresses subsequent frames in groups of ``temporal_factor``. As in ME-U0,
    a latent timestep is padding only when every frame in its group is padding.
    """

    frame_is_pad = torch.as_tensor(frame_is_pad, dtype=torch.bool)
    if frame_is_pad.ndim != 1 or frame_is_pad.numel() < 1:
        raise ValueError("video_is_pad must be a non-empty 1-D tensor")
    if temporal_factor <= 0:
        raise ValueError(f"temporal_factor must be positive, got {temporal_factor}")

    tail_is_pad = frame_is_pad[1:]
    if tail_is_pad.numel() % temporal_factor != 0:
        raise ValueError(
            "video_is_pad cannot align with VAE temporal factor: "
            f"frames={frame_is_pad.numel()}, temporal_factor={temporal_factor}"
        )

    latent_is_pad = tail_is_pad.view(-1, temporal_factor).all(dim=1)
    if latent_is_pad.numel() != latent_frames:
        raise ValueError(
            "video mask and future latent length differ: "
            f"mask={latent_is_pad.numel()}, latents={latent_frames}"
        )
    return (~latent_is_pad).repeat_interleave(height * width).unsqueeze(1)


def masked_token_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    token_valid: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error over valid tokens and all prediction channels."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    token_valid = torch.as_tensor(token_valid, device=prediction.device)
    if token_valid.ndim == prediction.ndim - 1:
        token_valid = token_valid.unsqueeze(-1)
    if token_valid.shape[:-1] != prediction.shape[:-1] or token_valid.shape[-1] not in (
        1,
        prediction.shape[-1],
    ):
        raise ValueError(
            f"token mask shape {token_valid.shape} cannot mask prediction {prediction.shape}"
        )

    valid = token_valid.to(dtype=prediction.dtype)
    squared_error = (prediction - target) ** 2
    valid_elements = valid.sum() * (
        prediction.shape[-1] if valid.shape[-1] == 1 else 1
    )
    return (squared_error * valid).sum() / valid_elements.clamp(min=1.0)
