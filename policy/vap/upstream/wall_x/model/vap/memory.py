"""Independent first-frame visual Anchor Bank for VAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .attention import RMSNorm
from .initialization import copy_linear, copy_norm
from .kv_cache import VAPAttentionKVCache
from .rope import RoPE4DPhaseCache, apply_rope_4d


@dataclass(frozen=True)
class FirstFrameMemoryTokens:
    """Compressed Anchor Bank tokens independent from the IVAVA layout."""

    tokens: torch.Tensor
    position_ids: torch.Tensor
    camera_ids: torch.Tensor
    grid: tuple[int, int]
    num_views: int
    valid: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("first-frame memory tokens must be [B,M,D]")
        if self.valid is not None and (self.valid.dtype != torch.bool or self.valid.shape != self.tokens.shape[:2]):
            raise ValueError("Memory token valid must be bool [B,M]")
        if tuple(self.position_ids.shape) != (*self.tokens.shape[:2], 4):
            raise ValueError("first-frame memory position_ids must be [B,M,4]")
        if tuple(self.camera_ids.shape) != tuple(self.tokens.shape[:2]):
            raise ValueError("first-frame memory camera_ids must be [B,M]")


class FirstFrameMemoryProjector(nn.Module):
    """Project one shared Anchor Bank to post-RoPE K/V exactly once.

    Video/Action queries and output projections stay owned by each selected
    VAP block.  The global bank therefore adds no duplicate Q/O parameters.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        rope_axes_dim: Sequence[int],
        eps: float = 1e-6,
        wan_attention: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if num_heads * head_dim != dim:
            raise ValueError("first-frame memory K/V width must equal video dim")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_axes_dim = tuple(int(value) for value in rope_axes_dim)
        self.context_k = nn.Linear(dim, dim)
        self.context_v = nn.Linear(dim, dim)
        self.context_norm_k = RMSNorm(dim, eps=eps)
        # Separate projection already identifies the modality.  A zero-init
        # type vector gives it optional capacity without perturbing warm start.
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        if wan_attention is not None:
            self.initialize_from_wan_cross_attention(wan_attention)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unflatten(-1, (self.num_heads, self.head_dim))

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        rope_cache: RoPE4DPhaseCache | None = None,
    ) -> VAPAttentionKVCache:
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim:
            raise ValueError(
                f"first-frame memory tokens must be [B,M,{self.dim}], "
                f"got {tuple(tokens.shape)}"
            )
        if tuple(position_ids.shape) != (*tokens.shape[:2], 4):
            raise ValueError("first-frame memory position_ids must be [B,M,4]")
        typed = tokens + self.type_embedding.to(tokens)
        key = self._heads(self.context_norm_k(self.context_k(typed)))
        value = self._heads(self.context_v(typed))
        key = apply_rope_4d(
            key,
            key,
            position_ids,
            axes_dim=self.rope_axes_dim,
            phase_cache=rope_cache,
        )[0]
        return VAPAttentionKVCache(
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
        )

    @torch.no_grad()
    def initialize_from_wan_cross_attention(self, wan_attention: nn.Module) -> None:
        copy_linear(self.context_k, wan_attention.k, allow_resize=False)
        copy_linear(self.context_v, wan_attention.v, allow_resize=False)
        copy_norm(self.context_norm_k, wan_attention.norm_k, allow_resize=False)


__all__ = ["FirstFrameMemoryProjector", "FirstFrameMemoryTokens"]
