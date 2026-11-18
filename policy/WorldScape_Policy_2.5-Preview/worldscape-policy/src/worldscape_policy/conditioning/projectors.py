"""Condition projection modules shared by native conditioners."""

import torch
from torch import nn

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        encoder_hidden_state: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.norm1(query)
        mask = (
            encoder_attention_mask.unsqueeze(1).to(dtype=torch.bool)
            if encoder_attention_mask is not None
            else None
        )
        output, _ = self.cross_attn(
            q,
            encoder_hidden_state,
            encoder_hidden_state,
            key_padding_mask=mask,
        )
        query = query + output
        return query + self.dropout(self.mlp(self.norm2(query)))


class LayerwiseQFormer(nn.Module):
    def __init__(
        self,
        input_hidden_dim: int = 2048,
        output_hidden_dim: int = 768,
        num_query_tokens: int = 64,
        num_layers: int = 37,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.input_hidden_dim = input_hidden_dim
        self.output_hidden_dim = output_hidden_dim
        self.num_query_tokens = num_query_tokens
        self.num_layers = num_layers
        self.proj = nn.Linear(input_hidden_dim, output_hidden_dim)
        self.query_tokens = nn.Parameter(
            torch.randn(num_query_tokens, output_hidden_dim)
        )
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(output_hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(
        self,
        hidden_states_list: list[torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(hidden_states_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {len(hidden_states_list)}"
            )
        batch = hidden_states_list[0].size(0)
        hidden_states_list = list(
            self.proj(torch.stack(hidden_states_list, dim=1)).unbind(dim=1)
        )
        query = self.query_tokens.unsqueeze(0).expand(batch, -1, -1)
        for index, layer in enumerate(self.layers):
            query = layer(query, hidden_states_list[index], encoder_attention_mask)
        return query


__all__ = ["CrossAttentionBlock", "LayerwiseQFormer"]
