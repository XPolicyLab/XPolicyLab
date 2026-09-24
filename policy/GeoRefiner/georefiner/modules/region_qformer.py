"""Language-guided region QFormer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from georefiner.config import GeoRefinerConfig
from georefiner.modules.common import FeedForward, sanitize_padding_mask


@dataclass(slots=True)
class RegionQFormerOutput:
    """Language-guided region tokens produced from learnable queries."""

    region_tokens: Tensor


class RegionQFormerBlock(nn.Module):
    """One lightweight region-query block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.geometry_norm = nn.LayerNorm(hidden_dim)
        self.language_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.geometry_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.language_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, dropout=dropout)

    def forward(
        self,
        queries: Tensor,
        geometry_tokens: Tensor,
        language_tokens: Tensor,
        geometry_padding_mask: Tensor | None = None,
        language_padding_mask: Tensor | None = None,
    ) -> Tensor:
        normalized_queries = self.query_norm(queries)
        self_out, _ = self.self_attn(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + self_out

        geom_out, _ = self.geometry_cross_attn(
            self.query_norm(queries),
            self.geometry_norm(geometry_tokens),
            self.geometry_norm(geometry_tokens),
            key_padding_mask=geometry_padding_mask,
            need_weights=False,
        )
        queries = queries + geom_out

        lang_out, _ = self.language_cross_attn(
            self.query_norm(queries),
            self.language_norm(language_tokens),
            self.language_norm(language_tokens),
            key_padding_mask=language_padding_mask,
            need_weights=False,
        )
        queries = queries + lang_out
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries


class RegionQFormer(nn.Module):
    """Learnable region queries conditioned on geometry and language tokens."""

    def __init__(self, config: GeoRefinerConfig) -> None:
        super().__init__()
        self.config = config
        self.region_queries = nn.Parameter(
            torch.empty(config.num_region_queries, config.hidden_dim)
        )
        nn.init.normal_(self.region_queries, std=0.02)
        self.layers = nn.ModuleList(
            [
                RegionQFormerBlock(config.hidden_dim, config.num_heads, config.dropout)
                for _ in range(config.region_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        geometry_tokens: Tensor,
        language_tokens: Tensor,
        geometry_padding_mask: Tensor | None = None,
        language_padding_mask: Tensor | None = None,
    ) -> RegionQFormerOutput:
        self._validate_tokens("geometry_tokens", geometry_tokens)
        self._validate_tokens("language_tokens", language_tokens)
        if geometry_tokens.shape[0] != language_tokens.shape[0]:
            raise ValueError(
                "geometry_tokens and language_tokens batch sizes must match; "
                f"got {geometry_tokens.shape[0]} and {language_tokens.shape[0]}."
            )

        geometry_padding_mask = sanitize_padding_mask(geometry_padding_mask)
        language_padding_mask = sanitize_padding_mask(language_padding_mask)
        batch_size = geometry_tokens.shape[0]
        queries = self.region_queries.unsqueeze(0).expand(batch_size, -1, -1)
        for layer in self.layers:
            queries = layer(
                queries,
                geometry_tokens,
                language_tokens,
                geometry_padding_mask=geometry_padding_mask,
                language_padding_mask=language_padding_mask,
            )
        return RegionQFormerOutput(region_tokens=self.output_norm(queries))

    def _validate_tokens(self, name: str, tokens: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(f"{name} must have shape [B, N, D]; got {tuple(tokens.shape)}.")
        if tokens.shape[-1] != self.config.hidden_dim:
            raise ValueError(
                f"{name} last dimension must be hidden_dim={self.config.hidden_dim}; "
                f"got {tuple(tokens.shape)}."
            )
