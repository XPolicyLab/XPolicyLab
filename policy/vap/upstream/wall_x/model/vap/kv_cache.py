"""Inference-only exact prefix KV cache containers for VAP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rope import RoPE4DPhaseCache


@dataclass
class VAPAttentionKVCache:
    """Post-RoPE keys and values in ``[B,H,S,D]`` layout.

    Self-attention caches may reserve an active tail.  Their immutable prefix
    is copied once into the combined buffer; every denoise evaluation updates
    only the tail.  ``key``/``value`` remain prefix-only views so callers cannot
    accidentally treat a previous noisy Action tail as reusable context.
    """

    key: torch.Tensor
    value: torch.Tensor
    active_capacity: int = 0

    def __post_init__(self) -> None:
        if self.key.ndim != 4 or self.key.shape != self.value.shape:
            raise ValueError(
                "cached key/value must share [B,H,S,D], got "
                f"{tuple(self.key.shape)} and {tuple(self.value.shape)}"
            )
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("cached key/value must share device and dtype")
        if self.active_capacity < 0:
            raise ValueError("active_capacity must be non-negative")
        self._prefix_tokens = int(self.key.shape[-2])
        self._combined_key = None
        self._combined_value = None
        if self.active_capacity:
            combined_shape = (
                *self.key.shape[:-2],
                self._prefix_tokens + self.active_capacity,
                self.key.shape[-1],
            )
            combined_key = torch.empty(
                combined_shape, device=self.key.device, dtype=self.key.dtype
            )
            combined_value = torch.empty_like(combined_key)
            combined_key[..., : self._prefix_tokens, :].copy_(self.key)
            combined_value[..., : self._prefix_tokens, :].copy_(self.value)
            self._combined_key = combined_key
            self._combined_value = combined_value
            # Keep the public tensors as prefix-only views backed by the same
            # storage, avoiding a second permanent prefix allocation.
            self.key = combined_key[..., : self._prefix_tokens, :]
            self.value = combined_value[..., : self._prefix_tokens, :]

    def update_active(
        self,
        active_key: torch.Tensor,
        active_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update only the reserved noisy-Action tail and return combined K/V."""

        if not self.active_capacity:
            raise RuntimeError("this KV cache has no active tail capacity")
        expected = (
            *self.key.shape[:-2],
            self.active_capacity,
            self.key.shape[-1],
        )
        if (
            tuple(active_key.shape) != expected
            or active_key.shape != active_value.shape
        ):
            raise ValueError(
                "active key/value must share reserved [B,H,S,D] shape "
                f"{expected}, got {tuple(active_key.shape)} and "
                f"{tuple(active_value.shape)}"
            )
        if (
            active_key.device != self.key.device
            or active_key.dtype != self.key.dtype
            or active_value.device != self.key.device
            or active_value.dtype != self.key.dtype
        ):
            raise ValueError("active and prefix KV must share device and dtype")
        if self._combined_key is None or self._combined_value is None:
            raise AssertionError("active KV buffers were not allocated")
        self._combined_key[..., self._prefix_tokens :, :].copy_(active_key)
        self._combined_value[..., self._prefix_tokens :, :].copy_(active_value)
        return self._combined_key, self._combined_value


@dataclass(frozen=True)
class VAPLayerKVCache:
    """Self-attention prefix plus optional static Text/Anchor K/V."""

    self_attention: VAPAttentionKVCache
    text_attention: VAPAttentionKVCache | None
    first_frame_memory: VAPAttentionKVCache | None = None


@dataclass(frozen=True)
class VAPActionPrefixKVCache:
    """Immutable per-layer prefix plus reusable active scratch for one stage."""

    layers: tuple[VAPLayerKVCache, ...]
    active_rank: int
    active_tokens: int
    batch_size: int
    active_action_rope: RoPE4DPhaseCache

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("prefix KV cache must contain at least one layer")
        if self.active_rank not in (1, 3):
            raise ValueError(
                f"Action active rank must be 1 or 3, got {self.active_rank}"
            )
        if self.active_tokens <= 0 or self.batch_size <= 0:
            raise ValueError("active_tokens and batch_size must be positive")
        if tuple(self.active_action_rope.cos.shape[:2]) != (
            self.batch_size,
            self.active_tokens,
        ):
            raise ValueError("active Action RoPE cache shape does not match KV cache")
        for index, layer in enumerate(self.layers):
            if layer.self_attention.key.shape[0] != self.batch_size:
                raise ValueError(f"layer {index} cache batch size mismatch")
            if layer.self_attention.key.shape[-1] != self.active_action_rope.head_dim:
                raise ValueError(f"layer {index} cache head size does not match RoPE")
            if (
                layer.self_attention.key.device != self.active_action_rope.cos.device
                or layer.self_attention.key.dtype != self.active_action_rope.cos.dtype
            ):
                raise ValueError(
                    f"layer {index} KV and Action RoPE must share device and dtype"
                )


__all__ = [
    "VAPActionPrefixKVCache",
    "VAPAttentionKVCache",
    "VAPLayerKVCache",
]
