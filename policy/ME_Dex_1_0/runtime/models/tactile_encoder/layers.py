from __future__ import annotations

import torch
import torch.nn as nn

from .base_layers import FP32LayerNorm
from .surface_layers import (
    fourier_coordinates,
    window_partition,
    window_reverse,
)


class MaskedWindowAttention2D(nn.Module):
    """Swin attention with 2-D relative position and external support masks."""

    def __init__(self, hidden: int, heads: int, window_height: int, window_width: int):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output = nn.Linear(hidden, hidden)
        table_size = (2 * window_height - 1) * (2 * window_width - 1)
        self.relative_bias = nn.Parameter(torch.zeros(table_size, heads))
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

        y, x = torch.meshgrid(
            torch.arange(window_height), torch.arange(window_width), indexing="ij"
        )
        coordinates = torch.stack((y, x)).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_height - 1
        relative[:, :, 1] += window_width - 1
        relative[:, :, 0] *= 2 * window_width - 1
        self.register_buffer(
            "relative_index", relative.sum(-1).to(torch.long), persistent=True
        )

    def forward(
        self,
        windows: torch.Tensor,
        *,
        key_valid: torch.Tensor,
        shifted_mask: torch.Tensor | None,
        windows_per_sample: int,
    ) -> torch.Tensor:
        window_batch, tokens, _ = windows.shape
        qkv = self.qkv(windows).reshape(
            window_batch, tokens, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        relative = self.relative_bias[self.relative_index.reshape(-1)]
        relative = relative.reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        logits = logits + relative[None].to(logits.dtype)

        if shifted_mask is not None:
            batch = window_batch // windows_per_sample
            logits = logits.view(batch, windows_per_sample, self.heads, tokens, tokens)
            logits = logits + shifted_mask[None, :, None].to(logits.dtype)
            logits = logits.reshape(window_batch, self.heads, tokens, tokens)

        invalid_keys = ~key_valid[:, None, None, :]
        logits = logits.masked_fill(invalid_keys, -10000.0)
        attention = logits.float().softmax(dim=-1).to(value.dtype)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(window_batch, tokens, self.hidden)
        return self.output(output) * key_valid[..., None].to(output.dtype)


class MaskedSwinSurfaceBlock(nn.Module):
    """W-MSA or shifted W-MSA on one masked 10x14 tactile surface."""

    def __init__(
        self,
        hidden: int,
        heads: int,
        *,
        full_height: int,
        full_width: int,
        window_height: int,
        window_width: int,
        shift_height: int = 0,
        shift_width: int = 0,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        self.full_height = full_height
        self.full_width = full_width
        self.window_height = window_height
        self.window_width = window_width
        self.shift_height = shift_height
        self.shift_width = shift_width
        self.windows_per_sample = (
            full_height // window_height
        ) * (full_width // window_width)
        self.attention_norm = FP32LayerNorm(hidden)
        self.attention = MaskedWindowAttention2D(
            hidden, heads, window_height, window_width
        )
        self.ffn_norm = FP32LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_multiplier),
            nn.GELU(),
            nn.Linear(hidden * ffn_multiplier, hidden),
        )
        self.register_buffer(
            "shifted_attention_mask", self._build_shifted_mask(), persistent=True
        )

    def _build_shifted_mask(self) -> torch.Tensor | None:
        if self.shift_height == 0 and self.shift_width == 0:
            return None
        labels = torch.zeros((1, self.full_height, self.full_width, 1))
        height_slices = (
            slice(0, -self.window_height),
            slice(-self.window_height, -self.shift_height),
            slice(-self.shift_height, None),
        )
        width_slices = (
            slice(0, -self.window_width),
            slice(-self.window_width, -self.shift_width),
            slice(-self.shift_width, None),
        )
        label = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                labels[:, height_slice, width_slice] = label
                label += 1
        windows = window_partition(labels, self.window_height, self.window_width)
        difference = windows[:, :, None, 0] - windows[:, None, :, 0]
        return difference.ne(0).to(torch.float32) * -10000.0

    def forward(self, values: torch.Tensor, support_mask: torch.Tensor) -> torch.Tensor:
        if values.shape[1:3] != (self.full_height, self.full_width):
            raise ValueError("Swin surface block received an unexpected grid size")
        if support_mask.shape != (*values.shape[:3], 1):
            raise ValueError("support mask must have shape [B,H,W,1]")
        support = support_mask.to(dtype=values.dtype)
        residual = values * support
        normalized = self.attention_norm(residual) * support
        valid = support_mask
        if self.shift_height or self.shift_width:
            normalized = torch.roll(
                normalized,
                shifts=(-self.shift_height, -self.shift_width),
                dims=(1, 2),
            )
            valid = torch.roll(
                valid,
                shifts=(-self.shift_height, -self.shift_width),
                dims=(1, 2),
            )
        windows = window_partition(normalized, self.window_height, self.window_width)
        window_valid = window_partition(
            valid, self.window_height, self.window_width
        )[..., 0].bool()
        windows = self.attention(
            windows,
            key_valid=window_valid,
            shifted_mask=self.shifted_attention_mask,
            windows_per_sample=self.windows_per_sample,
        )
        update = window_reverse(
            windows,
            batch=values.shape[0],
            full_height=self.full_height,
            full_width=self.full_width,
            window_height=self.window_height,
            window_width=self.window_width,
        )
        if self.shift_height or self.shift_width:
            update = torch.roll(
                update,
                shifts=(self.shift_height, self.shift_width),
                dims=(1, 2),
            )
        values = (residual + update) * support
        return (values + self.ffn(self.ffn_norm(values)) * support) * support


class PlainRelationBlock(nn.Module):
    """Plain self-attention; token identity is supplied outside this block."""

    def __init__(self, hidden: int, heads: int, ffn: int, scale_init: float):
        super().__init__()
        self.attention_norm = FP32LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn_norm = FP32LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden)
        )
        self.attention_scale = nn.Parameter(torch.full((hidden,), scale_init))
        self.ffn_scale = nn.Parameter(torch.full((hidden,), scale_init))

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(values)
        # A formal sample always has at least one physical surface.  Avoid a
        # tensor-to-host validity branch here because this runs every layer and
        # every step on PPU.
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = values + self.attention_scale.to(update.dtype) * update
        update = self.ffn(self.ffn_norm(values))
        values = values + self.ffn_scale.to(update.dtype) * update
        return values * valid[..., None].to(values.dtype)


__all__ = [
    "MaskedSwinSurfaceBlock",
    "PlainRelationBlock",
    "fourier_coordinates",
]
