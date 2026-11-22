"""Window-local 4D ``(chunk_start, local_time, height, width)`` MRoPE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


# Wan head_dim=128 originally allocates 44 channels to time and 42/42 to H/W.
# Split only the temporal budget into a compact planning-window chunk axis and
# a richer local-time axis, preserving both pretrained spatial subspaces.
HIERARCHICAL_ROPE_AXES_DIM = (8, 36, 42, 42)


@dataclass(frozen=True)
class RoPE4DPhaseCache:
    """Materialized 4D RoPE phases for one immutable token layout.

    Only the position-dependent cosine/sine tensors are cached. Rotated Q/K
    are deliberately excluded because they change across layers and diffusion
    timesteps. Inference callers own the association between this cache and the
    position IDs from which it was built.
    """

    cos: torch.Tensor
    sin: torch.Tensor
    axes_dim: tuple[int, int, int, int]
    theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.cos.ndim != 3 or self.cos.shape != self.sin.shape:
            raise ValueError(
                "cached RoPE cos/sin must share [B,S,D/2], got "
                f"{tuple(self.cos.shape)} and {tuple(self.sin.shape)}"
            )
        if self.cos.device != self.sin.device or self.cos.dtype != self.sin.dtype:
            raise ValueError("cached RoPE cos/sin must share device and dtype")
        if not self.cos.is_floating_point():
            raise ValueError("cached RoPE cos/sin must use a floating dtype")
        if self.cos.shape[0] <= 0 or self.cos.shape[1] <= 0:
            raise ValueError("cached RoPE batch and sequence dimensions must be positive")
        axes = _validate_axes(2 * self.cos.shape[-1], self.axes_dim)
        object.__setattr__(self, "axes_dim", axes)
        if not math.isfinite(float(self.theta)) or float(self.theta) <= 0:
            raise ValueError("RoPE theta must be finite and positive")

    @property
    def head_dim(self) -> int:
        return 2 * int(self.cos.shape[-1])

    def validate_for(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        axes_dim: Sequence[int],
        theta: float,
    ) -> None:
        if q.shape != k.shape or q.ndim != 4:
            raise ValueError(f"q/k must share [B,S,H,D], got {q.shape} and {k.shape}")
        if tuple(q.shape[:2]) != tuple(self.cos.shape[:2]):
            raise ValueError(
                "cached RoPE batch/sequence shape does not match q/k: "
                f"{tuple(self.cos.shape[:2])} != {tuple(q.shape[:2])}"
            )
        if q.shape[-1] != self.head_dim:
            raise ValueError(
                f"cached RoPE head_dim {self.head_dim} != q/k head_dim {q.shape[-1]}"
            )
        if self.cos.device != q.device or self.cos.dtype != q.dtype:
            raise ValueError("cached RoPE phase must share q/k device and dtype")
        if tuple(int(value) for value in axes_dim) != self.axes_dim:
            raise ValueError("cached RoPE axes do not match the attention axes")
        if float(theta) != float(self.theta):
            raise ValueError("cached RoPE theta does not match the attention theta")
        if position_ids is not None and tuple(position_ids.shape) != (
            q.shape[0],
            q.shape[1],
            4,
        ):
            raise ValueError("position_ids must align with cached q/k as [B,S,4]")

    def token_slice(self, start: int, end: int) -> "RoPE4DPhaseCache":
        """Return a zero-copy token view for one cached causal stage."""

        start, end = int(start), int(end)
        if not 0 <= start < end <= self.cos.shape[1]:
            raise ValueError(
                f"invalid RoPE token slice [{start}:{end}] for S={self.cos.shape[1]}"
            )
        return RoPE4DPhaseCache(
            cos=self.cos[:, start:end],
            sin=self.sin[:, start:end],
            axes_dim=self.axes_dim,
            theta=self.theta,
        )


def _validate_axes(head_dim: int, axes_dim: Sequence[int]) -> tuple[int, ...]:
    axes = tuple(int(value) for value in axes_dim)
    if len(axes) != 4:
        raise ValueError(f"4D MRoPE requires (g,tau,h,w), got {axes}")
    if sum(axes) != head_dim:
        raise ValueError(f"sum(axes_dim)={sum(axes)} must equal head_dim={head_dim}")
    if any(value <= 0 or value % 2 for value in axes):
        raise ValueError(f"each MRoPE axis width must be positive and even, got {axes}")
    return axes


def rope_cos_sin_4d(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    axes_dim: Sequence[int],
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-axis cos/sin in fp32 for ``[...,S,4]`` positions."""

    axes = _validate_axes(head_dim, axes_dim)
    if position_ids.shape[-1] != 4:
        raise ValueError(
            "position_ids must end in (chunk_start,local_time,height,width), "
            f"got {tuple(position_ids.shape)}"
        )
    positions = position_ids.to(dtype=torch.float32)
    cos_parts = []
    sin_parts = []
    for axis, width in enumerate(axes):
        base = torch.arange(0, width, 2, device=positions.device, dtype=torch.float32)
        inverse_frequency = theta ** (-base / width)
        phase = positions[..., axis : axis + 1] * inverse_frequency
        cos_parts.append(torch.cos(phase))
        sin_parts.append(torch.sin(phase))
    return torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1)


def build_rope_4d_cache(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    axes_dim: Sequence[int],
    theta: float = 10000.0,
    dtype: torch.dtype,
) -> RoPE4DPhaseCache:
    """Build one inference phase cache using the legacy fp32 phase math.

    Casting happens once after fp32 trig evaluation, matching the uncached
    implementation's rounding while avoiding a dtype conversion in every
    attention layer.
    """

    if position_ids.ndim != 3 or position_ids.shape[-1] != 4:
        raise ValueError(
            f"position_ids must be [B,S,4], got {tuple(position_ids.shape)}"
        )
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("RoPE cache dtype must be floating point")
    axes = _validate_axes(int(head_dim), axes_dim)
    cos, sin = rope_cos_sin_4d(
        position_ids,
        head_dim=int(head_dim),
        axes_dim=axes,
        theta=float(theta),
    )
    return RoPE4DPhaseCache(
        cos=cos.to(dtype=dtype).detach(),
        sin=sin.to(dtype=dtype).detach(),
        axes_dim=axes,
        theta=float(theta),
    )


def apply_rope_4d(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor | None,
    *,
    axes_dim: Sequence[int] = HIERARCHICAL_ROPE_AXES_DIM,
    theta: float = 10000.0,
    phase_cache: RoPE4DPhaseCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply hierarchical 4D MRoPE to ``[B,S,H,D]`` Q/K tensors."""

    if q.shape != k.shape or q.ndim != 4:
        raise ValueError(f"q/k must share [B,S,H,D], got {q.shape} and {k.shape}")
    axes = _validate_axes(q.shape[-1], axes_dim)
    if phase_cache is None:
        if position_ids is None or position_ids.shape[:2] != q.shape[:2]:
            raise ValueError(
                "position_ids [B,S,4] must align with q/k when no cache is given"
            )
        cos, sin = rope_cos_sin_4d(
            position_ids, head_dim=q.shape[-1], axes_dim=axes, theta=theta
        )
    else:
        phase_cache.validate_for(
            q,
            k,
            position_ids=position_ids,
            axes_dim=axes,
            theta=theta,
        )
        cos, sin = phase_cache.cos, phase_cache.sin

    def _rotate_by_axis(x: torch.Tensor) -> torch.Tensor:
        output = []
        channel_offset = 0
        frequency_offset = 0
        for width in axes:
            axis = x[..., channel_offset : channel_offset + width]
            first = axis[..., 0::2]
            second = axis[..., 1::2]
            axis_cos = cos[
                ..., frequency_offset : frequency_offset + width // 2
            ][:, :, None, :].to(dtype=x.dtype)
            axis_sin = sin[
                ..., frequency_offset : frequency_offset + width // 2
            ][:, :, None, :].to(dtype=x.dtype)
            output.append(
                torch.stack(
                    (
                        first * axis_cos - second * axis_sin,
                        second * axis_cos + first * axis_sin,
                    ),
                    dim=-1,
                ).flatten(-2)
            )
            channel_offset += width
            frequency_offset += width // 2
        return torch.cat(output, dim=-1)

    return _rotate_by_axis(q), _rotate_by_axis(k)


def action_position_ids(
    batch_size: int,
    num_chunks: int,
    tokens_per_chunk: int,
    *,
    chunk_stride: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create window-local Action coordinates ``(chunk_start,tau,0,0)``."""

    if batch_size <= 0 or num_chunks <= 0 or tokens_per_chunk <= 0:
        raise ValueError("batch_size, num_chunks and tokens_per_chunk must be positive")
    chunk_stride = tokens_per_chunk if chunk_stride is None else int(chunk_stride)
    if chunk_stride <= 0:
        raise ValueError("chunk_stride must be positive")
    chunk_start = torch.arange(num_chunks, device=device, dtype=torch.float32)
    chunk_start = (chunk_start * chunk_stride).repeat_interleave(tokens_per_chunk)
    local_time = torch.arange(
        1, tokens_per_chunk + 1, device=device, dtype=torch.float32
    ).repeat(num_chunks)
    zeros = torch.zeros_like(chunk_start)
    ids = torch.stack((chunk_start, local_time, zeros, zeros), dim=-1)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def text_position_ids(
    batch_size: int,
    sequence_length: int,
    *,
    window_origin: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Place one global instruction at the current planning-window origin."""

    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    order = torch.arange(sequence_length, device=device, dtype=torch.float32)
    origin = torch.full_like(order, float(window_origin))
    ids = torch.stack((origin, order, order, order), dim=-1)
    return ids.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def video_position_ids_from_grid(
    *,
    batch_size: int,
    num_chunks: int,
    num_views: int,
    latent_frames: int,
    height_tokens: int,
    width_tokens: int,
    chunk_stride: int,
    temporal_stride: int = 1,
    time_offset: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build window-local Video coordinates ``(chunk_start,tau,h,w)``."""

    values = (
        batch_size,
        num_chunks,
        num_views,
        latent_frames,
        height_tokens,
        width_tokens,
        chunk_stride,
        temporal_stride,
    )
    if any(value <= 0 for value in values) or time_offset < 0:
        raise ValueError("grid sizes/strides must be positive and offset non-negative")
    local_time = (
        torch.arange(latent_frames, device=device, dtype=torch.float32) + time_offset
    ) * temporal_stride
    height = torch.arange(height_tokens, device=device, dtype=torch.float32)
    width = torch.arange(width_tokens, device=device, dtype=torch.float32)
    coordinates = []
    for chunk in range(num_chunks):
        chunk_start = torch.full(
            (batch_size, num_views, latent_frames, height_tokens, width_tokens),
            float(chunk * chunk_stride),
            device=device,
            dtype=torch.float32,
        )
        tau = local_time[None, None, :, None, None].expand_as(chunk_start)
        h = height[None, None, None, :, None].expand_as(chunk_start)
        w = width[None, None, None, None, :].expand_as(chunk_start)
        coordinates.append(torch.stack((chunk_start, tau, h, w), dim=-1))
    return torch.stack(coordinates, dim=1).flatten(1, 5).contiguous()


__all__ = [
    "HIERARCHICAL_ROPE_AXES_DIM",
    "RoPE4DPhaseCache",
    "action_position_ids",
    "apply_rope_4d",
    "build_rope_4d_cache",
    "rope_cos_sin_4d",
    "text_position_ids",
    "video_position_ids_from_grid",
]
