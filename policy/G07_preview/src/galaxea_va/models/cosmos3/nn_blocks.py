# Copyright 2025 HuggingFace Inc.
# Copyright 2026 The HuggingFace Team. All rights reserved.
# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
# Modified by Galaxea AI, 2026, for inference-only use in G07-preview.
# Licensed under the Apache License, Version 2.0; see LICENSE and THIRD_PARTY_NOTICES.md
# in the G07_preview directory.
"""Vendored ``nn.Module`` building blocks for the Cosmos3 backbone.

Everything here is copied (behaviour-preserving) from NVIDIA/HuggingFace Cosmos3
diffusers source so fastwam does not need a ``diffusers`` install. Submodule and
parameter names are kept identical to the published Cosmos3-Nano checkpoint
(diffusers layout) so ``state_dict`` loads without renaming.

Sources (in the cosmos venv):
  diffusers/models/normalization.py         -> RMSNorm
  diffusers/models/embeddings.py            -> get_timestep_embedding / TimestepEmbedding / Timesteps
  diffusers/models/transformers/transformer_cosmos3.py
                                            -> _rotate_half / Cosmos3VLTextRotaryEmbedding / Cosmos3VLTextMLP
"""

from __future__ import annotations

import math
import numbers

import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm with fp32 variance compute (matches diffusers ``models.normalization.RMSNorm``)."""

    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if isinstance(dim, numbers.Integral):
            dim = (dim,)
        self.dim = torch.Size(dim)
        self.weight = None
        self.bias = None
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            if bias:
                self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        if self.weight is not None:
            if self.weight.dtype in (torch.float16, torch.bfloat16):
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
            if self.bias is not None:
                hidden_states = hidden_states + self.bias
        else:
            hidden_states = hidden_states.to(input_dtype)
        return hidden_states


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (matches diffusers ``get_timestep_embedding``)."""
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Timestep MLP (matches diffusers ``TimestepEmbedding``; keys ``linear_1`` / ``linear_2``)."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        assert act_fn == "silu", f"only silu vendored, got {act_fn!r}"
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.post_act = None

    def forward(self, sample, condition=None):
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class FP32RotaryEmbedding(nn.Module):
    """Keep deterministic RoPE frequencies in FP32 across module conversions."""

    def __init__(self, dim: int, theta: float):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.register_buffer("inv_freq", torch.empty(dim // 2, dtype=torch.float32), persistent=False)
        self._reset_inv_freq()

    def _reset_inv_freq(self):
        # Compute on CPU so dtype/device moves use the same FP32 values as loading.
        exponent = torch.arange(0, self.dim, 2, dtype=torch.float32, device="cpu") / self.dim
        self.inv_freq = (1.0 / (self.theta ** exponent)).to(device=self.inv_freq.device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # DeepSpeed casts floating buffers with module.bfloat16(); to_empty also
        # discards their contents. Rebuild from scalars instead of upcasting a
        # rounded buffer or reusing uninitialized storage.
        self._reset_inv_freq()
        return self


class Cosmos3VLTextRotaryEmbedding(FP32RotaryEmbedding):
    """Interleaved 3D mRoPE frequency generator (copied from transformer_cosmos3.py)."""

    def __init__(self, head_dim: int, rope_theta: float, rope_axes_dim):
        super().__init__(head_dim, rope_theta)
        self.rope_axes_dim = rope_axes_dim

    def apply_interleaved_mrope(self, freqs, rope_axes_dim):
        """Reorganize chunked [TTT...HHH...WWW] frequency layout into interleaved [THTHW...]."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = rope_axes_dim[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    def forward(self, position_ids, device, dtype):
        # position_ids: [3, N] (mrope) -> expand to [3, 1, N]
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(1)  # [3, 1, N]
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3, B, 1, N]
        # The generator's modality offset is 15000. A matmul can use TF32 even
        # with autocast disabled, rounding nearby positions together. This is
        # an outer product with K=1; pointwise FP32 multiplication preserves IDs.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            freqs = inv_freq_expanded * position_ids_expanded
        freqs = freqs.transpose(2, 3)  # [3, B, N, head_dim//2]
        freqs = self.apply_interleaved_mrope(freqs, self.rope_axes_dim)  # [B, N, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, N, head_dim]
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


class Cosmos3VLTextMLP(nn.Module):
    """SwiGLU MLP (copied from transformer_cosmos3.py); keys gate_proj/up_proj/down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
