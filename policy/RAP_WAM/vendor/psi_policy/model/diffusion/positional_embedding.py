from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim 必须为正整数")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError(f"SinusoidalPosEmb 输入必须为 1 维，实际 {tuple(x.shape)}")

        half_dim = self.dim // 2
        if half_dim == 0:
            return torch.zeros((x.shape[0], self.dim), device=x.device, dtype=torch.float32)

        emb = math.log(10000) / max(1, half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


__all__ = ["SinusoidalPosEmb"]
