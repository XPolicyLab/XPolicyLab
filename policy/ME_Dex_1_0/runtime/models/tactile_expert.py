"""Unified tactile expert for the ME-Dex video-action-tactile model."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from wan.modules.model import WanLayerNorm, WanRMSNorm


@contextmanager
def _fork_cpu_seed(seed: int):
    state = torch.get_rng_state()
    torch.default_generator.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(state)


def sinusoidal_embedding_1d(dim: int, positions: torch.Tensor) -> torch.Tensor:
    positions = positions.float().reshape(-1, 1)
    omega = torch.arange(dim // 2, device=positions.device, dtype=torch.float32)
    omega = torch.pow(10000.0, -omega / (dim // 2))
    angles = positions * omega.reshape(1, -1)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


@dataclass(frozen=True)
class TactileExpertConfig:
    latent_dim: int = 48
    latent_slices: int = 18
    queries_per_slice: int = 16
    condition_slices: int = 2
    hidden_size: int = 1024
    ffn_multiplier: int = 4
    num_layers: int = 30
    wan_num_heads: int = 24
    wan_head_dim: int = 128
    norm_eps: float = 1.0e-6
    freq_dim: int = 256
    initialization_seed: int = 42042
    token_packing: str = "robotwin_left_right"

    def __post_init__(self) -> None:
        contract = (
            self.latent_dim,
            self.latent_slices,
            self.queries_per_slice,
            self.condition_slices,
            self.hidden_size,
            self.token_packing,
        )
        expected = (48, 18, 16, 2, 1024, "robotwin_left_right")
        if contract != expected:
            raise ValueError(f"Unsupported tactile contract: {contract}")

    @classmethod
    def from_mapping(cls, values: Dict[str, Any] | None, *, num_layers: int):
        values = dict(values or {})
        values["num_layers"] = int(num_layers)
        return cls(**values)

    @property
    def future_slices(self) -> int:
        return self.latent_slices - self.condition_slices

    @property
    def tokens_per_slice(self) -> int:
        return 2

    @property
    def packed_latent_dim(self) -> int:
        return self.queries_per_slice * self.latent_dim // self.tokens_per_slice

    @property
    def sequence_length(self) -> int:
        return self.latent_slices * self.tokens_per_slice

    @property
    def wan_dim(self) -> int:
        return self.wan_num_heads * self.wan_head_dim

    def pack_latent(self, latent: torch.Tensor) -> torch.Tensor:
        expected = (self.latent_slices, self.queries_per_slice, self.latent_dim)
        if tuple(latent.shape[1:]) != expected:
            raise ValueError(f"Expected latent [B,{expected}], got {tuple(latent.shape)}")
        return latent.reshape(*latent.shape[:-2], self.tokens_per_slice, self.packed_latent_dim)

    def unpack_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.reshape(*latent.shape[:-2], self.queries_per_slice, self.latent_dim)


class TactileTokenizer(nn.Module):
    def __init__(self, config: TactileExpertConfig):
        super().__init__()
        self.config = config
        with _fork_cpu_seed(config.initialization_seed):
            self.input_projection = nn.Linear(config.packed_latent_dim, config.hidden_size)
            self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
            self.type_embedding = nn.Parameter(torch.randn(2, config.hidden_size) * 0.02)
            self.slice_embedding = nn.Parameter(torch.randn(config.latent_slices, config.hidden_size) * 0.02)
            self.query_embedding = nn.Parameter(torch.randn(config.tokens_per_slice, config.hidden_size) * 0.02)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        projected = self.input_projection(cfg.pack_latent(latent))
        tokens = F.layer_norm(
            projected.float(),
            (cfg.hidden_size,),
            self.input_norm.weight.float(),
            self.input_norm.bias.float(),
            self.input_norm.eps,
        ).to(projected.dtype)
        type_ids = torch.cat(
            (
                torch.zeros(cfg.condition_slices, dtype=torch.long, device=latent.device),
                torch.ones(cfg.future_slices, dtype=torch.long, device=latent.device),
            )
        )
        tokens = (
            tokens
            + self.type_embedding[type_ids][None, :, None]
            + self.slice_embedding[None, :, None]
            + self.query_embedding[None, None]
        )
        return tokens.reshape(latent.shape[0], cfg.sequence_length, cfg.hidden_size)


class TactileExpertBlock(nn.Module):
    def __init__(self, config: TactileExpertConfig):
        super().__init__()
        self.norm1 = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.norm2 = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.wan_tactile_qkv = nn.Parameter(
            torch.randn(3, config.wan_num_heads, config.hidden_size, config.wan_head_dim)
            / (config.hidden_size * config.wan_head_dim) ** 0.5
        )
        self.wan_tactile_norm_q = WanRMSNorm(config.wan_dim, eps=config.norm_eps)
        self.wan_tactile_norm_k = WanRMSNorm(config.wan_dim, eps=config.norm_eps)
        self.wan_tactile_o = nn.Linear(config.wan_dim, config.hidden_size, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * config.ffn_multiplier),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.hidden_size * config.ffn_multiplier, config.hidden_size),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, config.hidden_size) / config.hidden_size**0.5
        )


class TactileOutputHead(nn.Module):
    def __init__(self, config: TactileExpertConfig):
        super().__init__()
        self.config = config
        self.norm = WanLayerNorm(config.hidden_size, eps=config.norm_eps)
        self.modulation = nn.Parameter(
            torch.randn(1, 2, config.hidden_size) / config.hidden_size**0.5
        )
        self.projection = nn.Linear(config.hidden_size, config.packed_latent_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        shift, scale = (
            self.modulation.unsqueeze(0) + time_embedding.unsqueeze(2)
        ).chunk(2, dim=2)
        values = self.projection(
            self.norm(tokens) * (1 + scale.squeeze(2)) + shift.squeeze(2)
        )
        values = values.reshape(
            tokens.shape[0], cfg.latent_slices, cfg.tokens_per_slice, cfg.packed_latent_dim
        )
        values = cfg.unpack_latent(values)
        return torch.cat(
            (torch.zeros_like(values[:, : cfg.condition_slices]), values[:, cfg.condition_slices :]),
            dim=1,
        )


class TactileExpert(nn.Module):
    def __init__(self, config: TactileExpertConfig):
        super().__init__()
        self.config = config
        self.tokenizer = TactileTokenizer(config)
        with _fork_cpu_seed(config.initialization_seed + 2):
            self.time_embedding = nn.Sequential(
                nn.Linear(config.freq_dim, config.hidden_size),
                nn.SiLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.time_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(config.hidden_size, config.hidden_size * 6)
            )
            self.blocks = nn.ModuleList(
                [TactileExpertBlock(config) for _ in range(config.num_layers)]
            )
            self.output_head = TactileOutputHead(config)

    def get_time_embeddings(self, timestep: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        slice_t = torch.cat(
            (
                torch.zeros(timestep.shape[0], cfg.condition_slices, device=timestep.device),
                timestep.float()[:, None].expand(-1, cfg.future_slices),
            ),
            dim=1,
        )
        token_t = slice_t.repeat_interleave(cfg.tokens_per_slice, dim=1)
        features = sinusoidal_embedding_1d(cfg.freq_dim, token_t).reshape(
            timestep.shape[0], cfg.sequence_length, cfg.freq_dim
        )
        features = features.to(self.time_embedding[0].weight.dtype)
        embedded = self.time_embedding(features)
        projected = self.time_projection(embedded).reshape(
            timestep.shape[0], cfg.sequence_length, 6, cfg.hidden_size
        )
        return embedded, projected

    @staticmethod
    def modulation(block: TactileExpertBlock, projected: torch.Tensor):
        return (block.modulation.unsqueeze(0) + projected).chunk(6, dim=2)

    @staticmethod
    def apply_ffn(tokens: torch.Tensor, block: TactileExpertBlock, modulation):
        values = block.norm2(tokens).float() * (
            1 + modulation[4].squeeze(2)
        ) + modulation[3].squeeze(2)
        return tokens + block.ffn(values).to(tokens.dtype) * modulation[5].squeeze(2)
