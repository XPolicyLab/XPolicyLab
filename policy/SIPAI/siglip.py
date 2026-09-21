"""Inference layers; parameters are populated by strict checkpoint loading."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SigLIPConfig:
    """Configuration for a SigLIP vision tower and output projection."""

    image_size: int
    patch_size: int
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    output_dim: int

    def __post_init__(self) -> None:
        if (
            min(
                self.image_size,
                self.patch_size,
                self.width,
                self.depth,
                self.mlp_dim,
                self.num_heads,
                self.output_dim,
            )
            <= 0
        ):
            raise ValueError("SigLIP dimensions must be positive.")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


SIGLIP_SO400M_PATCH14_224 = SigLIPConfig(
    image_size=224,
    patch_size=14,
    width=1152,
    depth=27,
    mlp_dim=4304,
    num_heads=16,
    output_dim=2048,
)


class SigLIPMLP(nn.Module):
    def __init__(
        self,
        width: int,
        mlp_dim: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, mlp_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(mlp_dim, width, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states), approximate="tanh"))


class SigLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        config: SigLIPConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = nn.LayerNorm(config.width, eps=1e-6, dtype=dtype, device=device)
        self.attn = nn.MultiheadAttention(
            config.width,
            config.num_heads,
            batch_first=True,
            dtype=dtype,
            device=device,
        )
        self.mlp = SigLIPMLP(config.width, config.mlp_dim, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_dtype = self.norm1.weight.dtype
        normalized = self.norm1(hidden_states.to(norm_dtype))
        # Attention weights are never consumed. Disabling them lets PyTorch use
        # memory-efficient SDPA instead of materializing [B,H,N,N] weights.
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        hidden_states = hidden_states + attended
        return hidden_states + self.mlp(self.norm2(hidden_states.to(norm_dtype)))


class SigLIPEncoder(nn.Module):
    def __init__(
        self,
        config: SigLIPConfig,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [SigLIPEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.width, eps=1e-6, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states.to(self.norm.weight.dtype))


class SigLIPVisionEncoder(nn.Module):
    """Parameterizable SigLIP vision tower with a token-wise output projection."""

    def __init__(
        self,
        config: SigLIPConfig = SIGLIP_SO400M_PATCH14_224,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.stem = nn.Conv2d(
            3,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            padding=0,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, config.num_patches, config.width, dtype=dtype, device=device))
        self.encoder = SigLIPEncoder(config, dtype=dtype, device=device)
        self.head = nn.Linear(config.width, config.output_dim, dtype=dtype, device=device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        config = self.config
        if images.ndim != 4 or images.shape[1:3] != (config.image_size, config.image_size) or images.shape[-1] != 3:
            raise ValueError(
                f"Expected SigLIP RGB images with spatial size {config.image_size}; got {tuple(images.shape)}."
            )
        with torch.autocast(device_type=images.device.type, enabled=False):
            image_chw = images.permute(0, 3, 1, 2).float()
            hidden_states = F.conv2d(
                image_chw,
                self.stem.weight.float(),
                self.stem.bias.float() if self.stem.bias is not None else None,
                stride=self.stem.stride,
                padding=self.stem.padding,
            )
            batch_size, width, height, patch_width = hidden_states.shape
            hidden_states = hidden_states.reshape(batch_size, width, height * patch_width).permute(0, 2, 1)
            hidden_states = hidden_states + self.pos_embedding.float()
        hidden_states = self.encoder(hidden_states.to(self.encoder.norm.weight.dtype))
        return self.head(hidden_states)
