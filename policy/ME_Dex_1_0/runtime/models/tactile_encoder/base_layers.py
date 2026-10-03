from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm with an explicit FP32 compute path for PPU BF16 stability."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = F.layer_norm(
            values.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.to(values.dtype)


class PreNormSelfAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int):
        super().__init__()
        self.attention_norm = FP32LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ffn_norm = FP32LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden))

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(values)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = values + update
        values = values + self.ffn(self.ffn_norm(values))
        return values * valid[..., None].to(values.dtype)


class LayerScaledSelfAttention(nn.Module):
    """Inter-anatomy attention with a small learnable residual at initialization."""

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
        # Keep all-missing sensor samples finite on SDPA backends that do not
        # define an all-masked softmax. The final multiplication still zeros all
        # missing slots; observed samples use exactly the original attention.
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~valid.any(-1)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )
        values = values + self.attention_scale.to(update.dtype) * update
        update = self.ffn(self.ffn_norm(values))
        values = values + self.ffn_scale.to(update.dtype) * update
        return values * valid[..., None].to(values.dtype)


class ResidualMLP(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.norm = FP32LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(hidden, ffn), nn.GELU(), nn.Linear(ffn, hidden))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(self.norm(values))
