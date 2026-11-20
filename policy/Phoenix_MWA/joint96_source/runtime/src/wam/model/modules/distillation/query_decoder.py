"""Track-query decoder used by the Track4World distillation branch."""

from __future__ import annotations

import torch
from torch import nn


class TrackQueryDecoderLayer(nn.Module):
    """Pre-norm query self-attention, cross-attention, and FFN block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}."
            )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_norm = self.query_norm(queries)
        queries = queries + self.self_attn(
            query_norm, query_norm, query_norm, need_weights=False
        )[0]
        key_padding_mask = None
        if memory_valid is not None:
            if tuple(memory_valid.shape) != tuple(memory.shape[:2]):
                raise ValueError(
                    "memory_valid must match memory [B,S], got "
                    f"{tuple(memory_valid.shape)} vs {tuple(memory.shape[:2])}."
                )
            key_padding_mask = ~memory_valid.to(device=memory.device, dtype=torch.bool)
        normalized_memory = self.memory_norm(memory)
        queries = queries + self.cross_attn(
            self.cross_query_norm(queries),
            normalized_memory,
            normalized_memory,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return queries + self.ffn(self.ffn_norm(queries))


class TrackQueryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 16,
        ffn_dim: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        self.layers = nn.ModuleList(
            [
                TrackQueryDecoderLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        *,
        memory_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            queries = layer(queries, memory, memory_valid=memory_valid)
        return self.output_norm(queries)
