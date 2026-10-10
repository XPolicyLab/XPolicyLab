from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base_layers import FP32LayerNorm


def window_partition(values: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Partition [B,H,W,C] into exact non-overlapping windows."""

    batch, full_height, full_width, channels = values.shape
    if full_height % height or full_width % width:
        raise ValueError("feature map is not divisible by the requested window")
    values = values.view(
        batch, full_height // height, height, full_width // width, width, channels
    )
    return values.permute(0, 1, 3, 2, 4, 5).reshape(-1, height * width, channels)


def window_reverse(
    windows: torch.Tensor,
    *,
    batch: int,
    full_height: int,
    full_width: int,
    window_height: int,
    window_width: int,
) -> torch.Tensor:
    channels = windows.shape[-1]
    values = windows.view(
        batch,
        full_height // window_height,
        full_width // window_width,
        window_height,
        window_width,
        channels,
    )
    return values.permute(0, 1, 3, 2, 4, 5).reshape(
        batch, full_height, full_width, channels
    )


class WindowAttention2D(nn.Module):
    """Small Swin attention with learned 2-D relative-position bias."""

    def __init__(self, hidden: int, heads: int, window_height: int, window_width: int):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim**-0.5
        self.window_height = window_height
        self.window_width = window_width
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
        attention_mask: torch.Tensor | None = None,
        windows_per_sample: int,
    ) -> torch.Tensor:
        window_batch, tokens, _ = windows.shape
        qkv = self.qkv(windows).reshape(
            window_batch, tokens, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        bias = self.relative_bias[self.relative_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        logits = logits + bias[None].to(logits.dtype)
        if attention_mask is not None:
            batch = window_batch // windows_per_sample
            logits = logits.view(
                batch, windows_per_sample, self.heads, tokens, tokens
            )
            logits = logits + attention_mask[None, :, None].to(logits.dtype)
            logits = logits.reshape(window_batch, self.heads, tokens, tokens)
        # Explicit FP32 softmax avoids BF16 overflow on PPU and CUDA.
        attention = logits.float().softmax(dim=-1).to(value.dtype)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(window_batch, tokens, self.hidden)
        return self.output(output)


class SwinSurfaceBlock(nn.Module):
    """One W-MSA or shifted W-MSA block on the native 10x14 taxel grid."""

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
        self.attention = WindowAttention2D(
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

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1:3] != (self.full_height, self.full_width):
            raise ValueError("Swin surface block received an unexpected grid size")
        residual = values
        normalized = self.attention_norm(values)
        if self.shift_height or self.shift_width:
            normalized = torch.roll(
                normalized,
                shifts=(-self.shift_height, -self.shift_width),
                dims=(1, 2),
            )
        windows = window_partition(
            normalized, self.window_height, self.window_width
        )
        windows = self.attention(
            windows,
            attention_mask=self.shifted_attention_mask,
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
        values = residual + update
        return values + self.ffn(self.ffn_norm(values))


class GeometryAwareRelationBlock(nn.Module):
    """Relation attention over four surfaces x four spatial tokens.

    The bias distinguishes same-token, same-surface, same-hand/opposite-jaw,
    cross-hand/same-jaw, and cross-hand/opposite-jaw relations.  This adds the
    gripper topology without pretending the four surfaces form one 2-D image.
    """

    def __init__(self, hidden: int, heads: int, ffn: int, scale_init: float):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim**-0.5
        self.attention_norm = FP32LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.output = nn.Linear(hidden, hidden)
        self.relation_bias = nn.Embedding(5, heads)
        self.quadrant_pair_bias = nn.Embedding(16, heads)
        self.ffn_norm = FP32LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden)
        )
        self.attention_scale = nn.Parameter(torch.full((hidden,), scale_init))
        self.ffn_scale = nn.Parameter(torch.full((hidden,), scale_init))
        relation, quadrant_pair = self._relation_indices()
        self.register_buffer("relation_index", relation, persistent=True)
        self.register_buffer("quadrant_pair_index", quadrant_pair, persistent=True)
        nn.init.zeros_(self.relation_bias.weight)
        nn.init.zeros_(self.quadrant_pair_bias.weight)

    @staticmethod
    def _relation_indices() -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.arange(16)
        regions = token_ids // 4
        quadrants = token_ids % 4
        hands = regions // 2
        jaws = regions % 2
        relation = torch.empty(16, 16, dtype=torch.long)
        for query in range(16):
            for key in range(16):
                if query == key:
                    kind = 0
                elif regions[query] == regions[key]:
                    kind = 1
                elif hands[query] == hands[key]:
                    kind = 2
                elif jaws[query] == jaws[key]:
                    kind = 3
                else:
                    kind = 4
                relation[query, key] = kind
        quadrant_pair = quadrants[:, None] * 4 + quadrants[None]
        return relation, quadrant_pair

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = values.shape
        if tokens != 16:
            raise ValueError("geometry relation block requires exactly 16 tokens")
        normalized = self.attention_norm(values)
        qkv = self.qkv(normalized).reshape(
            batch, tokens, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        bias = self.relation_bias(self.relation_index)
        bias = bias + self.quadrant_pair_bias(self.quadrant_pair_index)
        logits = logits + bias.permute(2, 0, 1)[None].to(logits.dtype)
        attention = logits.float().softmax(dim=-1).to(value.dtype)
        update = torch.matmul(attention, value)
        update = self.output(update.transpose(1, 2).reshape(batch, tokens, self.hidden))
        values = values + self.attention_scale.to(update.dtype) * update
        update = self.ffn(self.ffn_norm(values))
        return values + self.ffn_scale.to(update.dtype) * update


def fourier_coordinates(coordinates: torch.Tensor) -> torch.Tensor:
    """Encode canonical decoder coordinates; never used as encoder input."""

    frequencies = coordinates.new_tensor([1.0, 2.0, 4.0, 8.0]) * math.pi
    angles = coordinates.unsqueeze(-1) * frequencies
    return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)
