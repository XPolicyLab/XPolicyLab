from typing import Any, Dict, Optional, Tuple

import torch


def _as_batch_scale(
    scales: Any,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scales = torch.as_tensor(scales, device=device, dtype=dtype)
    if scales.ndim == 0:
        return scales.view(1, 1, 1, 1, 1).expand(batch_size, -1, -1, -1, -1)
    if scales.ndim == 1:
        if scales.shape[0] != batch_size:
            raise ValueError(f"scales batch dimension {scales.shape[0]} must be {batch_size}")
        return scales.view(batch_size, 1, 1, 1, 1)
    if scales.shape[0] != batch_size:
        raise ValueError(f"scales batch dimension {scales.shape[0]} must be {batch_size}")
    return scales


def apply_future_frame_noise_augmentation(
    latents: torch.Tensor,
    *,
    enabled: bool,
    probability: float = 0.5,
    min_scale: float = 0.5,
    max_scale: float = 1.0,
    future_start_index: int = 1,
    noise: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply LingBot-style partial noise interpolation to future latent frames.

    For selected samples, future frames are replaced by
    ``(1 - s_aug) * eps + s_aug * z``. Frames before ``future_start_index``
    are always preserved.
    """
    if latents.ndim != 5:
        raise ValueError(f"latents must be [B, C, F, H, W], got shape {tuple(latents.shape)}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if not 0.0 <= min_scale <= max_scale <= 1.0:
        raise ValueError(
            f"scale range must satisfy 0 <= min_scale <= max_scale <= 1, got {min_scale}, {max_scale}"
        )
    if future_start_index < 1:
        raise ValueError(
            f"future_start_index must be >= 1 so the condition frame is not augmented, got {future_start_index}"
        )

    batch_size = latents.shape[0]
    applied = torch.zeros(batch_size, device=latents.device, dtype=torch.bool)
    scale_info = torch.zeros(batch_size, 1, 1, 1, 1, device=latents.device, dtype=latents.dtype)
    info = {"applied": applied, "scales": scale_info}

    if not enabled or probability == 0.0 or future_start_index >= latents.shape[2]:
        return latents, info

    if probability == 1.0:
        applied = torch.ones_like(applied)
    else:
        applied = torch.rand(batch_size, device=latents.device) < probability

    if not applied.any():
        info["applied"] = applied
        return latents, info

    future = latents[:, :, future_start_index:]
    if noise is None:
        noise = torch.randn(future.shape, device=latents.device, dtype=latents.dtype)
    else:
        noise = noise.to(device=latents.device, dtype=latents.dtype)
        if noise.shape != future.shape:
            raise ValueError(f"noise shape {tuple(noise.shape)} must match future shape {tuple(future.shape)}")

    if scales is None:
        scales = torch.empty(batch_size, 1, 1, 1, 1, device=latents.device, dtype=latents.dtype)
        scales.uniform_(min_scale, max_scale)
    else:
        scales = _as_batch_scale(scales, batch_size, device=latents.device, dtype=latents.dtype)

    augmented = latents.clone()
    augmented_future = (1.0 - scales) * noise + scales * future
    applied_mask = applied.view(batch_size, 1, 1, 1, 1)
    augmented[:, :, future_start_index:] = torch.where(applied_mask, augmented_future, future)

    info["applied"] = applied
    info["scales"] = scales
    return augmented, info
