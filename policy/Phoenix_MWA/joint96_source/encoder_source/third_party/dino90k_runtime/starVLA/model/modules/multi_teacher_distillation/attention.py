"""Self-contained Transformer blocks used by the deployable video adapter."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_pairs(value: torch.Tensor) -> torch.Tensor:
    paired = value.unflatten(-1, (-1, 2))
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _apply_axis_rope(value: torch.Tensor, positions: torch.Tensor, axis_dim: int, base: float) -> torch.Tensor:
    if axis_dim == 0:
        return value
    if axis_dim % 2:
        raise ValueError(f"RoPE axis dimension must be even, got {axis_dim}")
    frequencies = torch.arange(0, axis_dim, 2, device=value.device, dtype=torch.float32) / axis_dim
    inverse = base ** (-frequencies)
    angles = positions.float().unsqueeze(-1) * inverse
    sine = angles.sin().repeat_interleave(2, dim=-1).unsqueeze(1)
    cosine = angles.cos().repeat_interleave(2, dim=-1).unsqueeze(1)
    rotated = value.float() * cosine + _rotate_pairs(value.float()) * sine
    return rotated.to(dtype=value.dtype)


def apply_multiaxis_rope(
    value: torch.Tensor,
    positions: torch.Tensor,
    axis_dims: tuple[int, ...],
    base: float,
) -> torch.Tensor:
    """Rotate selected head channels independently for each position axis.

    ``value`` is ``[B, heads, tokens, head_dim]`` and ``positions`` is
    ``[B, tokens, axes]``. Any unused remainder in ``head_dim`` is unchanged.
    """

    if positions.shape[-1] != len(axis_dims):
        raise ValueError(f"Expected {len(axis_dims)} position axes, got {positions.shape[-1]}")
    if sum(axis_dims) > value.shape[-1]:
        raise ValueError(f"RoPE dimensions {axis_dims} exceed head dimension {value.shape[-1]}")
    parts = []
    start = 0
    for axis, axis_dim in enumerate(axis_dims):
        end = start + axis_dim
        parts.append(_apply_axis_rope(value[..., start:end], positions[..., axis], axis_dim, base))
        start = end
    if start < value.shape[-1]:
        parts.append(value[..., start:])
    return torch.cat(parts, dim=-1)


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        axis_dims: tuple[int, ...],
        rope_base: float,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.axis_dims = tuple(int(value) for value in axis_dims)
        self.rope_base = float(rope_base)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = tokens.shape
        if positions.shape[:2] != (batch_size, num_tokens):
            raise ValueError(f"Position shape {tuple(positions.shape)} does not match tokens {tuple(tokens.shape)}")
        qkv = self.qkv(tokens).view(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        query = apply_multiaxis_rope(query, positions, self.axis_dims, self.rope_base)
        key = apply_multiaxis_rope(key, positions, self.axis_dims, self.rope_base)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch_size, num_tokens, self.dim))


class LayerScale(nn.Module):
    """Per-channel residual scaling initialized near zero."""

    def __init__(self, dim: int, init_value: float) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_value)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.gamma


class TransformerBlock(nn.Module):
    """Pre-LN attention and MLP branches with optional LayerScale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        axis_dims: tuple[int, ...],
        rope_base: float,
        mlp_ratio: float = 4.0,
        layer_scale_init: float | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = RotarySelfAttention(dim, num_heads, axis_dims, rope_base)
        self.layer_scale1 = LayerScale(dim, layer_scale_init) if layer_scale_init is not None else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.layer_scale2 = LayerScale(dim, layer_scale_init) if layer_scale_init is not None else nn.Identity()

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.layer_scale1(self.attention(self.norm1(tokens), positions))
        return tokens + self.layer_scale2(self.mlp(self.norm2(tokens)))
