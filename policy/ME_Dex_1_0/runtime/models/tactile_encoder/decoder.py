from __future__ import annotations

import torch
import torch.nn as nn

from .base_layers import FP32LayerNorm, ResidualMLP

from .config import RobotwinTactileAEV41Config
from .layers import fourier_coordinates


class MaskedTaxelCrossAttention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim**-0.5
        self.query_norm = FP32LayerNorm(hidden)
        self.memory_norm = FP32LayerNorm(hidden)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)

    def forward(
        self, taxels: torch.Tensor, tokens: torch.Tensor, memory_valid: torch.Tensor
    ) -> torch.Tensor:
        batch, queries, _ = taxels.shape
        memory = tokens.shape[1]
        query = self.query(self.query_norm(taxels)).reshape(
            batch, queries, self.heads, self.head_dim
        ).transpose(1, 2)
        normalized_memory = self.memory_norm(tokens)
        key = self.key(normalized_memory).reshape(
            batch, memory, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(normalized_memory).reshape(
            batch, memory, self.heads, self.head_dim
        ).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        logits = logits.masked_fill(~memory_valid[:, None, None, :], -10000.0)
        attention = logits.float().softmax(dim=-1).to(value.dtype)
        update = torch.matmul(attention, value)
        update = update.transpose(1, 2).reshape(batch, queries, self.hidden)
        return taxels + self.output(update)


class SpatialRefinementBlock(nn.Module):
    """One residual 5x5-effective-field refinement block (two 3x3 convs)."""

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        values = values * support
        return (values + self.net(values) * support) * support


class SingleForceDecoderV41(nn.Module):
    """The sole decoder: taxel queries, one spatial refinement, and gated force."""

    def __init__(self, config: RobotwinTactileAEV41Config):
        super().__init__()
        self.region_embedding = nn.Embedding(4, config.latent_dim)
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(16, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.cross_attention = MaskedTaxelCrossAttention(
            config.latent_dim, config.relation_attention_heads
        )
        self.layers = nn.ModuleList(
            ResidualMLP(config.latent_dim, config.ffn_dim)
            for _ in range(config.decoder_layers)
        )
        self.output_norm = FP32LayerNorm(config.latent_dim)
        self.refinement = SpatialRefinementBlock(
            config.latent_dim, config.decoder_refinement_hidden
        )
        self.force_head = nn.Conv2d(config.latent_dim, 3, 1)
        self.contact_head = nn.Conv2d(config.latent_dim, 1, 1)
        nn.init.trunc_normal_(self.region_embedding.weight, std=0.02)
        nn.init.normal_(self.force_head.weight, std=0.01)
        with torch.no_grad():
            self.force_head.bias.copy_(torch.tensor((-4.0, 0.0, 0.0)))
        nn.init.zeros_(self.contact_head.weight)
        nn.init.zeros_(self.contact_head.bias)
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, config.height),
            torch.linspace(-1.0, 1.0, config.width),
            indexing="ij",
        )
        self.register_buffer(
            "taxel_coordinates", torch.stack((x, y), dim=-1).reshape(-1, 2),
            persistent=True,
        )
        self.config = config

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_valid: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if frame_tokens.ndim != 3 or frame_tokens.shape[1:] != (
            self.config.frame_tokens,
            self.config.latent_dim,
        ):
            raise ValueError(f"frame tokens must have shape [B,16,{self.config.latent_dim}]")
        batch = frame_tokens.shape[0]
        if token_valid.shape != (batch, 16):
            raise ValueError("token_valid must have shape [B,16]")
        if support_mask.shape != (batch, 4, 1, 10, 14):
            raise ValueError("support_mask must have shape [B,4,1,10,14]")
        memory = frame_tokens.reshape(batch, 4, 4, self.config.latent_dim).flatten(0, 1)
        memory_valid = token_valid.reshape(batch, 4, 4).flatten(0, 1)
        coordinates = fourier_coordinates(self.taxel_coordinates)
        taxels = self.coordinate_embedding(coordinates)
        region_ids = torch.arange(4, device=frame_tokens.device)
        taxels = taxels[None, None] + self.region_embedding(region_ids)[:, None]
        taxels = taxels.expand(batch, -1, -1, -1).flatten(0, 1)
        hidden = self.cross_attention(taxels, memory, memory_valid)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.output_norm(hidden)
        hidden = hidden.transpose(1, 2).reshape(batch * 4, self.config.latent_dim, 10, 14)
        support = support_mask.flatten(0, 1).to(hidden.dtype)
        hidden = self.refinement(hidden, support)
        force_logits = self.force_head(hidden)
        raw_force = torch.cat(
            (force_logits[:, :1].sigmoid(), force_logits[:, 1:].tanh()), dim=1
        ) * support
        contact_logits = self.contact_head(hidden)
        contact_probability = contact_logits.sigmoid() * support
        reconstructed = raw_force * contact_probability
        def unflatten(values: torch.Tensor) -> torch.Tensor:
            return values.reshape(batch, 4, *values.shape[1:])
        return {
            "raw_reconstructed_force": unflatten(raw_force),
            "reconstructed_force": unflatten(reconstructed),
            "contact_logits": unflatten(contact_logits),
            "contact_probability": unflatten(contact_probability),
        }
