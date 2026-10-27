"""CLIP ViT-H/14 image encoder of Wan2.1 I2V.

Loads the ``visual`` tower of ``models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth``
from Wan-AI/Wan2.1-I2V-14B-480P and returns the 257 tokens of its penultimate
block, as Wan2.1 does.

Adapted from Wan2.1 (https://github.com/Wan-Video/Wan2.1), Apache-2.0.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        weight = None if self.weight is None else self.weight.float()
        bias = None if self.bias is None else self.bias.float()
        return F.layer_norm(
            x.float(), self.normalized_shape, weight, bias, self.eps
        ).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, c = x.shape
        q, k, v = self.to_qkv(x).view(b, s, 3, self.num_heads, self.head_dim).unbind(2)
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        )
        return self.proj(x.transpose(1, 2).reshape(b, s, c))


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, num_heads: int, eps: float):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=eps)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, eps=eps)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        mlp_ratio: float = 4,
        out_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 32,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.image_size = image_size
        num_patches = (image_size // patch_size) ** 2
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(1, num_patches + 1, dim))
        self.pre_norm = LayerNorm(dim, eps=eps)
        self.transformer = nn.Sequential(
            *[AttentionBlock(dim, mlp_ratio, num_heads, eps) for _ in range(num_layers)]
        )
        # Unused by the penultimate-block features; kept to load the file strictly.
        self.post_norm = LayerNorm(dim, eps=eps)
        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(x.size(0), -1, -1), x], dim=1)
        x = self.pre_norm(x + self.pos_embedding)
        return self.transformer[:-1](x)


class CLIPImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer()
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))
        self.normalize = T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

    def forward(self, videos: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode the first frame of each [3,T,H,W] video in [-1, 1]."""
        size = (self.visual.image_size,) * 2
        frames = torch.cat(
            [
                F.interpolate(
                    u[:, :1].transpose(0, 1),
                    size=size,
                    mode="bicubic",
                    align_corners=False,
                )
                for u in videos
            ],
            dim=0,
        )
        frames = self.normalize(frames.mul(0.5).add(0.5))
        return self.visual(frames)
