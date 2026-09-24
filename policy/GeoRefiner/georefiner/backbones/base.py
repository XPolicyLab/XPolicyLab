"""Abstract backbone interfaces used by GeoRefiner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from torch import Tensor, nn


@dataclass(slots=True)
class TextBackboneOutput:
    """Token-level text features and a GeoRefiner padding mask.

    ``padding_mask`` follows PyTorch attention semantics: ``True`` positions
    are padding and must be ignored.
    """

    tokens: Tensor
    padding_mask: Tensor | None = None


class VisualBackboneBase(nn.Module, ABC):
    """Visual encoder interface.

    Implementations should map RGB images `[B, 3, H, W]` to token sequences
    `[B, N, visual_feature_dim]`.
    """

    output_dim: int

    @abstractmethod
    def forward(self, images: Tensor) -> Tensor:
        """Encode a batch of RGB images into visual tokens."""


class TextBackboneBase(nn.Module, ABC):
    """Text encoder interface.

    Implementations should map instructions or token ids to text tokens
    `[B, L, text_feature_dim]`.
    """

    output_dim: int

    @abstractmethod
    def forward(
        self,
        instruction: str | Sequence[str] | None = None,
        token_ids: Tensor | None = None,
    ) -> Tensor | TextBackboneOutput:
        """Encode text instructions or token ids into language tokens."""
