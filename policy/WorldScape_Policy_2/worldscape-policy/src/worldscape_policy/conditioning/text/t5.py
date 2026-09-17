"""WorldScape-owned Wan T5 text encoder.

The numerical module and parameter names are copied from
``groot.vla.model.dreamzero.modules.wan_video_text_encoder``. Only the
raw-string tokenizer adapter is added here so legacy checkpoints retain their
exact state-dict layout.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from worldscape_policy.conditioning.text.tokenizer import HuggingfaceTokenizer


def fp16_clamp(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )


class T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dim_attn % num_heads:
            raise ValueError("dim_attn must be divisible by num_heads")
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context
        batch, heads, head_dim = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(batch, -1, heads, head_dim)
        k = self.k(context).view(batch, -1, heads, head_dim)
        v = self.v(context).view(batch, -1, heads, head_dim)
        attn_bias = x.new_zeros(batch, heads, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            if mask.ndim not in (2, 3):
                raise ValueError("mask must have rank 2 or 3")
            mask = mask.view(batch, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        x = self.o(x.reshape(batch, -1, heads * head_dim))
        return self.dropout(x)


class T5FeedForward(nn.Module):
    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.fc1(x) * self.gate(x))
        return self.dropout(self.fc2(x))


class T5SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = (
            None
            if shared_pos
            else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e = (
            pos_bias
            if self.shared_pos
            else self.pos_embedding(x.size(1), x.size(1))  # type: ignore[misc]
        )
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        return fp16_clamp(x + self.ffn(self.norm2(x)))


class T5RelativeEmbedding(nn.Module):
    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        bidirectional: bool,
        max_dist: int = 128,
    ) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(
            lq, device=device
        ).unsqueeze(1)
        rel_pos_embeds = self.embedding(self._relative_position_bucket(rel_pos))
        return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (
            torch.log(rel_pos.float() / max_exact)
            / math.log(self.max_dist / max_exact)
            * (num_buckets - max_exact)
        ).long()
        rel_pos_large = torch.min(
            rel_pos_large,
            torch.full_like(rel_pos_large, num_buckets - 1),
        )
        return rel_buckets + torch.where(
            rel_pos < max_exact, rel_pos, rel_pos_large
        )


def init_weights(module: nn.Module) -> None:
    if isinstance(module, T5LayerNorm):
        nn.init.ones_(module.weight)
    elif isinstance(module, T5FeedForward):
        nn.init.normal_(module.gate[0].weight, std=module.dim**-0.5)
        nn.init.normal_(module.fc1.weight, std=module.dim**-0.5)
        nn.init.normal_(module.fc2.weight, std=module.dim_ffn**-0.5)
    elif isinstance(module, T5Attention):
        nn.init.normal_(module.q.weight, std=(module.dim * module.dim_attn) ** -0.5)
        nn.init.normal_(module.k.weight, std=module.dim**-0.5)
        nn.init.normal_(module.v.weight, std=module.dim**-0.5)
        nn.init.normal_(
            module.o.weight, std=(module.num_heads * module.dim_attn) ** -0.5
        )
    elif isinstance(module, T5RelativeEmbedding):
        nn.init.normal_(
            module.embedding.weight,
            std=(2 * module.num_buckets * module.num_heads) ** -0.5,
        )


class WanTextEncoder(nn.Module):
    def __init__(
        self,
        vocab: int | nn.Embedding = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        shared_pos: bool = False,
        dropout: float = 0.1,
        text_encoder_pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.text_encoder_pretrained_path = text_encoder_pretrained_path
        self.token_embedding = nn.Embedding(vocab, dim) if isinstance(vocab, int) else vocab
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
            if shared_pos
            else None
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                T5SelfAttention(
                    dim,
                    dim_attn,
                    dim_ffn,
                    num_heads,
                    num_buckets,
                    shared_pos,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = T5LayerNorm(dim)
        self.apply(init_weights)

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.dropout(self.token_embedding(ids))
        e = (
            self.pos_embedding(x.size(1), x.size(1))
            if self.shared_pos and self.pos_embedding is not None
            else None
        )
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        return self.dropout(self.norm(x))

    @staticmethod
    def state_dict_converter() -> WanTextEncoderStateDictConverter:
        return WanTextEncoderStateDictConverter()


class WanTextEncoderStateDictConverter:
    def from_diffusers(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict

    def from_civitai(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict


class T5InstructionEncoder(WanTextEncoder):
    """Wan T5 encoder with native raw-string tokenization."""

    def __init__(
        self,
        *,
        tokenizer_path: str,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path,
            seq_len=max_length,
            clean="whitespace",
        )

    def encode_text(self, instructions: list[str]) -> torch.Tensor:
        ids, mask = self._tokenizer(
            instructions,
            return_mask=True,
            add_special_tokens=True,
        )
        device = self.token_embedding.weight.device
        ids = ids.to(device=device)
        mask = mask.to(device=device)
        embeddings = self.forward(ids, mask).to(dtype=torch.bfloat16)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        for batch_index, length in enumerate(sequence_lengths):
            embeddings[batch_index, int(length.item()) :] = 0
        return embeddings


__all__ = ["T5InstructionEncoder", "WanTextEncoder"]
