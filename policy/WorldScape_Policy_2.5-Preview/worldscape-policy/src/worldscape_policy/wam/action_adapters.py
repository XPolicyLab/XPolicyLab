"""Embodiment-specific action projection layers shared by WAM backends."""

import torch
from torch import nn
from torch.nn import functional as F


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(
            half_dim, dtype=torch.float, device=timesteps.device
        ) * (torch.log(torch.tensor(10000.0)) / half_dim)
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class CategorySpecificLinear(nn.Module):
    def __init__(
        self, num_categories: int, input_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(
            0.02 * torch.randn(num_categories, input_dim, hidden_dim)
        )
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(
        self, x: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(
        self, action_dim: int, hidden_size: int, num_embodiments: int
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(
            num_embodiments, 2 * hidden_size, hidden_size
        )
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        cat_ids: torch.Tensor,
    ) -> torch.Tensor:
        action_embedding = self.W1(actions, cat_ids)
        time_embedding = self.pos_encoding(timesteps).to(
            dtype=action_embedding.dtype
        )
        hidden = swish(
            self.W2(torch.cat([action_embedding, time_embedding], dim=-1), cat_ids)
        )
        return self.W3(hidden, cat_ids)


__all__ = [
    "CategorySpecificLinear",
    "CategorySpecificMLP",
    "MultiEmbodimentActionEncoder",
    "SinusoidalPositionalEncoding",
    "swish",
]
