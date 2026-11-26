"""Language encoder for instruction or precomputed language tokens."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from georefiner.backbones.base import TextBackboneBase, TextBackboneOutput
from georefiner.backbones.mock import MockTextBackbone
from georefiner.config import GeoRefinerConfig
from georefiner.types import Instruction


@dataclass(slots=True)
class LanguageEncoderOutput:
    """Projected language tokens and optional padding mask."""

    language_tokens: Tensor
    padding_mask: Tensor | None = None


class LanguageEncoder(nn.Module):
    """Project text-backbone features into GeoRefiner hidden space."""

    def __init__(
        self,
        config: GeoRefinerConfig,
        text_backbone: TextBackboneBase | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.text_backbone = text_backbone or MockTextBackbone(config.text_feature_dim)
        self.proj = nn.Linear(config.text_feature_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

        if config.freeze_text_backbone:
            for parameter in self.text_backbone.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        instruction: Instruction | None = None,
        language_tokens: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> LanguageEncoderOutput:
        if language_tokens is None:
            if instruction is None:
                raise ValueError("LanguageEncoder requires instruction or language_tokens.")
            backbone_output = self.text_backbone(instruction=instruction)
            if isinstance(backbone_output, TextBackboneOutput):
                language_tokens = backbone_output.tokens
                if padding_mask is None:
                    padding_mask = backbone_output.padding_mask
            else:
                language_tokens = backbone_output

        if language_tokens.ndim != 3:
            raise ValueError(
                "language_tokens must have shape [B, L, text_feature_dim]; "
                f"got {tuple(language_tokens.shape)}."
            )
        if language_tokens.shape[-1] != self.config.text_feature_dim:
            raise ValueError(
                "language_tokens last dimension must match config.text_feature_dim="
                f"{self.config.text_feature_dim}; got {tuple(language_tokens.shape)}."
            )

        projected = self.norm(self.proj(language_tokens))
        if padding_mask is not None and padding_mask.shape != projected.shape[:2]:
            raise ValueError(
                "language padding mask must have shape [B, L]; "
                f"got {tuple(padding_mask.shape)} for tokens {tuple(projected.shape)}."
            )
        return LanguageEncoderOutput(language_tokens=projected, padding_mask=padding_mask)
