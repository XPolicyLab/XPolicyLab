from __future__ import annotations

import torch
import torch.nn as nn

from .residual_layers import ResidualBlock2d
from .base_layers import FP32LayerNorm

from .config import RobotwinTactileAEV41Config
from .layers import MaskedSwinSurfaceBlock


class FullResolutionForceEncoderV41(nn.Module):
    """Encode one force surface while keeping its 10x14 taxel grid."""

    def __init__(self, config: RobotwinTactileAEV41Config):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResidualBlock2d(64),
            nn.Conv2d(64, config.surface_hidden_dim, 1),
            nn.GroupNorm(8, config.surface_hidden_dim),
            nn.SiLU(),
            ResidualBlock2d(config.surface_hidden_dim),
        )
        common = dict(
            hidden=config.surface_hidden_dim,
            heads=config.surface_attention_heads,
            full_height=config.height,
            full_width=config.width,
            window_height=config.window_height,
            window_width=config.window_width,
        )
        self.window_attention = MaskedSwinSurfaceBlock(**common)
        self.shifted_window_attention = MaskedSwinSurfaceBlock(
            **common,
            shift_height=config.shift_height,
            shift_width=config.shift_width,
        )
        self.output_norm = FP32LayerNorm(config.surface_hidden_dim)

    def forward(
        self, force: torch.Tensor, support_mask: torch.Tensor
    ) -> torch.Tensor:
        if force.ndim != 4 or force.shape[1:] != (3, 10, 14):
            raise ValueError("surface force must have shape [N,3,10,14]")
        if support_mask.shape != (force.shape[0], 1, 10, 14):
            raise ValueError("support mask must have shape [N,1,10,14]")
        support = support_mask.to(force.dtype)
        features = self.stem(force * support) * support
        features = features.permute(0, 2, 3, 1)
        nhwc_support = support.permute(0, 2, 3, 1).bool()
        features = self.window_attention(features, nhwc_support)
        features = self.shifted_window_attention(features, nhwc_support)
        features = self.output_norm(features) * nhwc_support.to(features.dtype)
        return features.permute(0, 3, 1, 2).contiguous()


class MaskedQuadrantLearnedPoolingV41(nn.Module):
    """Produce one token per fixed quadrant without reading invalid taxels."""

    def __init__(self, config: RobotwinTactileAEV41Config):
        super().__init__()
        hidden = config.surface_hidden_dim
        self.queries = nn.Parameter(torch.empty(4, hidden))
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.output = nn.Linear(hidden, config.latent_dim)
        self.output_norm = FP32LayerNorm(config.latent_dim)
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.scale = hidden**-0.5

    @staticmethod
    def _quadrants(values: torch.Tensor) -> torch.Tensor:
        quadrants = (
            values[:, :5, :7],
            values[:, :5, 7:],
            values[:, 5:, :7],
            values[:, 5:, 7:],
        )
        return torch.stack(
            [item.reshape(item.shape[0], 35, -1) for item in quadrants], dim=1
        )

    def forward(
        self, features: torch.Tensor, support_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if support_mask.shape != (features.shape[0], 1, 10, 14):
            raise ValueError("support mask must have shape [N,1,10,14]")
        quadrants = self._quadrants(features.permute(0, 2, 3, 1))
        valid = self._quadrants(
            support_mask.permute(0, 2, 3, 1).bool()
        )[..., 0]
        token_valid = valid.any(dim=-1)
        key = self.key(quadrants)
        value = self.value(quadrants)
        logits = torch.einsum("nqsd,qd->nqs", key, self.queries) * self.scale
        logits = logits.masked_fill(~valid, -10000.0)
        weights = logits.float().softmax(dim=-1).to(value.dtype)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        pooled = torch.einsum("nqs,nqsd->nqd", weights, value)
        tokens = self.output_norm(self.output(pooled))
        tokens = tokens * token_valid[..., None].to(tokens.dtype)
        return (
            tokens,
            weights.reshape(features.shape[0], 4, 5, 7),
            token_valid,
        )


class SurfaceTokenizerV41(nn.Module):
    def __init__(self, config: RobotwinTactileAEV41Config):
        super().__init__()
        self.encoder = FullResolutionForceEncoderV41(config)
        self.pool = MaskedQuadrantLearnedPoolingV41(config)

    def forward(
        self, force: torch.Tensor, support_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        feature_map = self.encoder(force, support_mask)
        tokens, pooling_weights, token_valid = self.pool(feature_map, support_mask)
        return {
            "surface_feature_map": feature_map,
            "surface_tokens": tokens,
            "quadrant_pooling_weights": pooling_weights,
            "surface_token_valid": token_valid,
        }
