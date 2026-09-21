"""Inference encoder for SigLIP history tokens."""

import math

from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = next(n for n in (32, 16, 8, 4, 2, 1) if channels % n == 0)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, inputs):
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        return inputs + self.conv2(F.silu(self.norm2(hidden)))


class HistoryEncoder(nn.Module):
    def __init__(self, grid_size=4, hidden_dim=512):
        super().__init__()
        if grid_size not in (1, 2, 4, 8, 16) or hidden_dim <= 0:
            raise ValueError("Invalid history encoder dimensions")
        self.grid_size = grid_size
        self.input_norm = nn.LayerNorm(2048)
        self.input_projection = nn.Conv2d(2048, hidden_dim, 1)
        self.stages = nn.ModuleList(
            nn.Sequential(
                ResidualBlock(hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            )
            for _ in range(int(math.log2(16 // grid_size)))
        )
        self.bottleneck = ResidualBlock(hidden_dim)
        self.output_projection = nn.Conv2d(hidden_dim, 2048, 1)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (256, 2048):
            raise ValueError("History encoder expects [batch, 256, 2048] SigLIP tokens")
        hidden = self.input_norm(tokens.to(self.input_norm.weight.dtype))
        hidden = hidden.reshape(tokens.shape[0], 16, 16, 2048).permute(0, 3, 1, 2)
        hidden = self.input_projection(hidden)
        for stage in self.stages:
            hidden = stage(hidden)
        hidden = self.output_projection(self.bottleneck(hidden))
        return hidden.permute(0, 2, 3, 1).reshape(
            tokens.shape[0], self.grid_size * self.grid_size, 2048
        )
