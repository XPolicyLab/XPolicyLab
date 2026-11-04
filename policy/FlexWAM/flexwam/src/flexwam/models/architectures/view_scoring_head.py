"""Shared first-frame view scoring head for router and critic."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from flexwam.models.architectures.state_conditioning import encode_state, state_config

CANONICAL_HEAD_ARCH = "spatial_query_v1"
HEAD_ARCH_ALIASES = {
    "spatial_query_v1": CANONICAL_HEAD_ARCH,
    "l18_spatial_query_v1": CANONICAL_HEAD_ARCH,
    "l18": CANONICAL_HEAD_ARCH,
}
HEAD_NETWORK_KEYS = (
    "arch",
    "feature_layer_index",
    "feature_dim",
    "pool_grid",
    "num_queries",
    "num_heads",
    "spatial_hidden_dim",
    "branch_output_dim",
    "head_hidden_dim",
    "state_conditioning",
)


def normalize_head_arch(arch: str | None) -> str:
    key = CANONICAL_HEAD_ARCH if arch in (None, "") else str(arch)
    if key not in HEAD_ARCH_ALIASES:
        raise ValueError(
            f"Unsupported camera head arch {arch!r}. Expected one of {sorted(HEAD_ARCH_ALIASES)}."
        )
    return HEAD_ARCH_ALIASES[key]


def arches_compatible(left: str | None, right: str | None) -> bool:
    try:
        return normalize_head_arch(left) == normalize_head_arch(right)
    except ValueError:
        return False


def _build_2d_sincos_pos_embed(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [H*W, dim] 2D sine-cosine positional encodings."""
    y = torch.arange(height, device=device, dtype=torch.float32)
    x = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega_dim = dim // 4
    omega = 1.0 / (10000.0 ** (torch.arange(omega_dim, device=device, dtype=torch.float32) / float(omega_dim)))
    y_embed = yy.reshape(-1, 1) * omega.unsqueeze(0)
    x_embed = xx.reshape(-1, 1) * omega.unsqueeze(0)
    pos = torch.cat([y_embed.sin(), y_embed.cos(), x_embed.sin(), x_embed.cos()], dim=-1)
    return pos.to(dtype=dtype)


class SpatialQueryHead(nn.Module):
    """Joint multi-camera head over first-frame visual tokens.

    Spatial branch keeps a coarse pooled layout; query branch attends over all
    spatial tokens. The two summaries are concatenated per camera, then a
    shared MLP emits V logits.
    """

    arch_name = CANONICAL_HEAD_ARCH

    def __init__(
        self,
        token_dim: int,
        num_views: int = 3,
        feature_dim: int = 256,
        pool_grid: Sequence[int] = (4, 4),
        num_queries: int = 4,
        num_heads: int = 4,
        spatial_hidden_dim: int = 1024,
        branch_output_dim: int = 512,
        head_hidden_dim: int = 512,
        proprio_dim: int | None = None,
        state_conditioning: dict | None = None,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_views = int(num_views)
        self.feature_dim = int(feature_dim)
        self.pool_grid = (int(pool_grid[0]), int(pool_grid[1]))
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.spatial_hidden_dim = int(spatial_hidden_dim)
        self.branch_output_dim = int(branch_output_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        if self.num_views <= 1:
            raise ValueError(f"`num_views` must be > 1, got {self.num_views}.")
        if self.feature_dim % self.num_heads != 0:
            raise ValueError(
                f"`feature_dim` must be divisible by `num_heads`, got {self.feature_dim} and {self.num_heads}."
            )
        if self.feature_dim % 4 != 0:
            raise ValueError(f"`feature_dim` must be divisible by 4 for 2D sincos PE, got {self.feature_dim}.")

        self.input_norm = nn.LayerNorm(self.token_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(self.token_dim, self.feature_dim),
            nn.GELU(),
        )
        pool_tokens = self.pool_grid[0] * self.pool_grid[1]
        spatial_in = pool_tokens * self.feature_dim
        self.spatial_pool = nn.AdaptiveAvgPool2d(self.pool_grid)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(spatial_in, self.spatial_hidden_dim),
            nn.GELU(),
            nn.Linear(self.spatial_hidden_dim, self.branch_output_dim),
            nn.GELU(),
        )
        self.queries = nn.Parameter(torch.empty(self.num_queries, self.feature_dim))
        nn.init.xavier_uniform_(self.queries)
        self.query_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=self.num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(self.num_queries * self.feature_dim, self.branch_output_dim),
            nn.GELU(),
        )
        joint_dim = self.num_views * (2 * self.branch_output_dim)
        self.state_conditioning = state_config(state_conditioning)
        self.proprio_dim = int(proprio_dim or 0)
        self.state_encoder = None
        if self.state_conditioning["enabled"]:
            state_dim = self.state_conditioning["hidden_dim"]
            self.state_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, state_dim), nn.GELU(), nn.Linear(state_dim, state_dim),
            )
            joint_dim += state_dim
        self.head = nn.Sequential(
            nn.Linear(joint_dim, self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, self.num_views),
        )
        final = self.head[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def network_config(self) -> dict:
        return {
            "arch": str(self.arch_name),
            "feature_dim": int(self.feature_dim),
            "pool_grid": [int(self.pool_grid[0]), int(self.pool_grid[1])],
            "num_queries": int(self.num_queries),
            "num_heads": int(self.num_heads),
            "spatial_hidden_dim": int(self.spatial_hidden_dim),
            "branch_output_dim": int(self.branch_output_dim),
            "head_hidden_dim": int(self.head_hidden_dim),
            "state_conditioning": dict(self.state_conditioning),
            "proprio_dim": self.proprio_dim if self.state_conditioning["enabled"] else None,
        }

    def forward(
        self,
        features: torch.Tensor,
        grid_hw: tuple[int, int] | Sequence[int],
        return_logits: bool = False,
        proprio: torch.Tensor | None = None,
        state_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, num_tokens, token_dim = features.shape
        if num_views != self.num_views:
            raise ValueError(f"Head expects {self.num_views} views, got {num_views}.")
        if token_dim != self.token_dim:
            raise ValueError(f"Head token dim mismatch: expected {self.token_dim}, got {token_dim}.")
        # FlexWAM validates the first-frame spatial grid before dispatching to either head.
        height, width = int(grid_hw[0]), int(grid_hw[1])

        tokens = features.reshape(batch_size * num_views, num_tokens, token_dim)
        projected = self.input_proj(self.input_norm(tokens))
        spatial = projected.transpose(1, 2).reshape(
            batch_size * num_views,
            self.feature_dim,
            height,
            width,
        )
        pooled = self.spatial_pool(spatial)
        spatial_flat = pooled.permute(0, 2, 3, 1).reshape(batch_size * num_views, -1)
        spatial_summary = self.spatial_mlp(spatial_flat)

        pos = _build_2d_sincos_pos_embed(
            height,
            width,
            self.feature_dim,
            device=projected.device,
            dtype=projected.dtype,
        )
        query = self.queries.to(device=projected.device, dtype=projected.dtype)
        query = query.unsqueeze(0).expand(batch_size * num_views, -1, -1)
        key = projected + pos.unsqueeze(0)
        attended, _ = self.query_attn(query, key, projected, need_weights=False)
        query_summary = self.query_mlp(attended.reshape(batch_size * num_views, -1))

        per_view = torch.cat([spatial_summary, query_summary], dim=-1)
        joint = per_view.reshape(batch_size, num_views * per_view.shape[-1])
        if self.state_encoder is not None:
            state_features = encode_state(
                self.state_encoder, proprio, batch_size=batch_size, proprio_dim=self.proprio_dim,
                reference=joint, keep_mask=state_keep_mask,
            )
            joint = torch.cat([joint, state_features], dim=-1)
        logits = self.head(joint)
        if return_logits:
            return logits
        weights = torch.softmax(logits.float(), dim=-1).to(dtype=features.dtype)
        value_scales = weights * float(num_views)
        return weights, value_scales
