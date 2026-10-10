from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.net(values)


class PreNormSelfAttentionLayer(nn.Module):
    def __init__(self, hidden: int = 256, heads: int = 8, ffn: int = 1024):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.GELU(),
            nn.Linear(ffn, hidden),
        )

    def forward(
        self,
        values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(values)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        values = values + update
        return values + self.ffn(self.ffn_norm(values))


class PreNormCrossAttentionLayer(nn.Module):
    def __init__(self, hidden: int = 256, heads: int = 8, ffn: int = 1024):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden)
        self.memory_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn),
            nn.GELU(),
            nn.Linear(ffn, hidden),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_memory = self.memory_norm(memory)
        update, _ = self.attention(
            self.query_norm(query),
            normalized_memory,
            normalized_memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        query = query + update
        return query + self.ffn(self.ffn_norm(query))
