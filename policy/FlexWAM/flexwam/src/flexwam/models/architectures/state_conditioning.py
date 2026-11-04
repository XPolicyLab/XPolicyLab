"""Validation and masking for optional, current-state conditioning."""

from typing import Any

import torch


def conditioning_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Model-wide state routing; missing mode preserves legacy checkpoints."""
    result = {"mode": "separate", "drop_prob": 0.0, **dict(config or {})}
    if set(result) - {"mode", "drop_prob"} or result["mode"] not in ("separate", "shared"):
        raise ValueError("state_conditioning accepts mode='separate'/'shared' and drop_prob.")
    if not 0.0 <= float(result["drop_prob"]) <= 1.0:
        raise ValueError("state_conditioning.drop_prob must be in [0,1].")
    return result


def state_config(config: dict[str, Any] | None, *, video: bool = False) -> dict[str, Any]:
    config = {} if config is None else dict(config)
    defaults = {"enabled": False, "mode": "prefix" if video else "late_concat"}
    defaults.update({"num_tokens": 1} if video else {"hidden_dim": 256})
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown state conditioning options: {sorted(unknown)}")
    result = {**defaults, **config}
    if not isinstance(result["enabled"], bool):
        raise ValueError("state_conditioning.enabled must be a boolean.")
    if result["mode"] != defaults["mode"]:
        raise ValueError(f"state_conditioning.mode must be {defaults['mode']!r}.")
    if video and result["num_tokens"] != 1:
        raise ValueError("Video state conditioning supports exactly one token.")
    if not video and (not isinstance(result["hidden_dim"], int) or result["hidden_dim"] <= 0):
        raise ValueError("state_conditioning.hidden_dim must be a positive integer.")
    return result


def encode_state(encoder, proprio, *, batch_size: int, proprio_dim: int, reference,
                 keep_mask: torch.Tensor | None = None) -> torch.Tensor:
    if proprio is None or proprio.shape != (batch_size, proprio_dim):
        shape = None if proprio is None else tuple(proprio.shape)
        raise ValueError(f"proprio must be [{batch_size},{proprio_dim}], got {shape}.")
    result = encoder(proprio.to(device=reference.device, dtype=reference.dtype))
    if keep_mask is not None:
        if keep_mask.shape != (batch_size, 1) or keep_mask.dtype != torch.bool:
            raise ValueError("state_keep_mask must be boolean [B,1].")
        result = result * keep_mask.to(device=result.device, dtype=result.dtype)
    return result
