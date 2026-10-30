"""Offline mock backbones for unit tests and shape-level debugging."""

from __future__ import annotations

import hashlib
from typing import Sequence

import torch
from torch import Tensor, nn

from georefiner.backbones.base import TextBackboneBase, VisualBackboneBase


class MockVisualBackbone(VisualBackboneBase):
    """Small Conv2d patch projection returning patch-like visual tokens.

    This module is only a test double for DINOv2-style token output.
    """

    def __init__(
        self,
        output_dim: int,
        patch_size: int = 16,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive; got {output_dim}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive; got {patch_size}.")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive; got {in_channels}.")

        self.output_dim = output_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=output_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B, C, H, W]; got {tuple(images.shape)}."
            )
        if images.shape[1] != self.in_channels:
            raise ValueError(
                f"images channel dimension must be {self.in_channels}; "
                f"got shape {tuple(images.shape)}."
            )

        patches = self.proj(images)
        return patches.flatten(2).transpose(1, 2).contiguous()


class MockTextBackbone(TextBackboneBase):
    """Tiny deterministic text encoder for offline tests.

    It accepts either token ids `[B, L]` or raw instruction strings. String tokens
    are mapped to ids with a stable hash. This is not a CLIP implementation.
    """

    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 4096,
        max_length: int = 16,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive; got {output_dim}.")
        if vocab_size <= 1:
            raise ValueError(f"vocab_size must be greater than 1; got {vocab_size}.")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive; got {max_length}.")

        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, output_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_length, output_dim))
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        instruction: str | Sequence[str] | None = None,
        token_ids: Tensor | None = None,
    ) -> Tensor:
        if token_ids is None:
            if instruction is None:
                raise ValueError("Provide instruction or token_ids to MockTextBackbone.")
            token_ids = self._tokenize(instruction)
        else:
            if token_ids.ndim != 2:
                raise ValueError(
                    f"token_ids must have shape [B, L]; got {tuple(token_ids.shape)}."
                )
            if token_ids.shape[1] > self.max_length:
                raise ValueError(
                    f"token_ids length must be <= max_length={self.max_length}; "
                    f"got L={token_ids.shape[1]}."
                )
            if token_ids.dtype != torch.long:
                raise TypeError(f"token_ids must have dtype torch.long; got {token_ids.dtype}.")

        token_ids = token_ids.to(self.embedding.weight.device)
        embeddings = self.embedding(token_ids)
        positions = self.position_embedding[: token_ids.shape[1]].unsqueeze(0)
        return embeddings + positions

    def _tokenize(self, instruction: str | Sequence[str]) -> Tensor:
        if isinstance(instruction, str):
            texts = [instruction]
        else:
            texts = list(instruction)
            if not all(isinstance(text, str) for text in texts):
                raise TypeError("instruction sequence must contain only strings.")
            if not texts:
                raise ValueError("instruction sequence must be non-empty.")

        rows = []
        for text in texts:
            pieces = text.lower().strip().split()
            ids = [self._stable_token_id(piece) for piece in pieces[: self.max_length]]
            ids.extend([0] * (self.max_length - len(ids)))
            rows.append(ids)
        return torch.tensor(rows, dtype=torch.long)

    def _stable_token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        value = int.from_bytes(digest, byteorder="big")
        return 1 + (value % (self.vocab_size - 1))
