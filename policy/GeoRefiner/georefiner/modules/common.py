"""Common neural network blocks for GeoRefiner encoders."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class TransformerIdentity(nn.Module):
    """Identity module with a TransformerEncoder-compatible forward signature."""

    def forward(self, inputs: Tensor, *args: object, **kwargs: object) -> Tensor:
        return inputs


class FeedForward(nn.Module):
    """Small transformer-style feed-forward network."""

    def __init__(self, hidden_dim: int, dropout: float = 0.0, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


def make_transformer_encoder(
    hidden_dim: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """Build a pre-norm TransformerEncoder, or identity when depth is zero."""

    if num_layers == 0:
        return TransformerIdentity()
    layer = nn.TransformerEncoderLayer(
        d_model=hidden_dim,
        nhead=num_heads,
        dim_feedforward=hidden_dim * 4,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


def sinusoidal_position_embedding(length: int, hidden_dim: int, device: torch.device) -> Tensor:
    """Return sinusoidal position embeddings with shape `[1, length, hidden_dim]`."""

    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / hidden_dim)
    )
    embedding = torch.zeros(length, hidden_dim, device=device)
    embedding[:, 0::2] = torch.sin(position * div_term)
    if hidden_dim > 1:
        embedding[:, 1::2] = torch.cos(position * div_term[: embedding[:, 1::2].shape[1]])
    return embedding.unsqueeze(0)


def sanitize_padding_mask(mask: Tensor | None) -> Tensor | None:
    """Convert a padding mask to bool and avoid all-masked rows.

    PyTorch attention can produce NaNs when every key in a row is masked. For
    shape-level tests and downstream robustness, all-masked rows are treated as
    unmasked instead of being passed through as invalid attention inputs.
    """

    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError(f"padding mask must have shape [B, N]; got {tuple(mask.shape)}.")
    output = mask.to(dtype=torch.bool).clone()
    all_masked = output.all(dim=1)
    if all_masked.any():
        output[all_masked] = False
    return output
