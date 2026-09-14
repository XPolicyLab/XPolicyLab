"""Dual-view geometry encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from georefiner.backbones.base import VisualBackboneBase
from georefiner.backbones.mock import MockVisualBackbone
from georefiner.config import GeoRefinerConfig
from georefiner.modules.common import make_transformer_encoder, sanitize_padding_mask


@dataclass(slots=True)
class GeometryEncoderOutput:
    """Geometry tokens and per-view projected tokens from `GeometryEncoder`."""

    geometry_tokens: Tensor
    global_tokens: Tensor
    wrist_tokens: Tensor
    padding_mask: Tensor | None = None


class GeometryEncoder(nn.Module):
    """Encode global and wrist RGB/tokens into cross-view geometry tokens."""

    def __init__(
        self,
        config: GeoRefinerConfig,
        visual_backbone: VisualBackboneBase | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.visual_backbone = visual_backbone or MockVisualBackbone(config.visual_feature_dim)
        self.visual_projection = nn.Sequential(
            nn.Linear(config.visual_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.view_embedding = nn.Parameter(torch.zeros(2, config.hidden_dim))
        nn.init.normal_(self.view_embedding, std=0.02)
        self.transformer = make_transformer_encoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            num_layers=config.geometry_layers,
            dropout=config.dropout,
        )

        if config.freeze_visual_backbone:
            for parameter in self.visual_backbone.parameters():
                parameter.requires_grad_(False)

    def forward(
        self,
        global_rgb: Tensor | None = None,
        wrist_rgb: Tensor | None = None,
        global_visual_tokens: Tensor | None = None,
        wrist_visual_tokens: Tensor | None = None,
        global_visual_padding_mask: Tensor | None = None,
        wrist_visual_padding_mask: Tensor | None = None,
    ) -> GeometryEncoderOutput:
        global_tokens = self._tokens_for_view(
            view_name="global",
            rgb=global_rgb,
            visual_tokens=global_visual_tokens,
            view_index=0,
        )
        wrist_tokens = self._tokens_for_view(
            view_name="wrist",
            rgb=wrist_rgb,
            visual_tokens=wrist_visual_tokens,
            view_index=1,
        )

        if global_tokens.shape[0] != wrist_tokens.shape[0]:
            raise ValueError(
                "global and wrist token batch sizes must match; "
                f"got {global_tokens.shape[0]} and {wrist_tokens.shape[0]}."
            )

        padding_mask = self._concat_masks(
            global_tokens,
            wrist_tokens,
            global_visual_padding_mask,
            wrist_visual_padding_mask,
        )
        tokens = torch.cat([global_tokens, wrist_tokens], dim=1)
        geometry_tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return GeometryEncoderOutput(
            geometry_tokens=geometry_tokens,
            global_tokens=global_tokens,
            wrist_tokens=wrist_tokens,
            padding_mask=padding_mask,
        )

    def _tokens_for_view(
        self,
        view_name: str,
        rgb: Tensor | None,
        visual_tokens: Tensor | None,
        view_index: int,
    ) -> Tensor:
        if visual_tokens is None:
            if rgb is None:
                raise ValueError(
                    f"Provide {view_name}_rgb or {view_name}_visual_tokens."
                )
            visual_tokens = self.visual_backbone(rgb)

        if visual_tokens.ndim != 3:
            raise ValueError(
                f"{view_name}_visual_tokens must have shape [B, N, visual_feature_dim]; "
                f"got {tuple(visual_tokens.shape)}."
            )
        if visual_tokens.shape[-1] != self.config.visual_feature_dim:
            raise ValueError(
                f"{view_name}_visual_tokens last dimension must match "
                f"config.visual_feature_dim={self.config.visual_feature_dim}; "
                f"got {tuple(visual_tokens.shape)}."
            )

        projected = self.visual_projection(visual_tokens)
        return projected + self.view_embedding[view_index].view(1, 1, -1)

    def _concat_masks(
        self,
        global_tokens: Tensor,
        wrist_tokens: Tensor,
        global_mask: Tensor | None,
        wrist_mask: Tensor | None,
    ) -> Tensor | None:
        if global_mask is None and wrist_mask is None:
            return None

        batch_size = global_tokens.shape[0]
        device = global_tokens.device
        if global_mask is None:
            global_mask = torch.zeros(
                batch_size,
                global_tokens.shape[1],
                dtype=torch.bool,
                device=device,
            )
        if wrist_mask is None:
            wrist_mask = torch.zeros(
                batch_size,
                wrist_tokens.shape[1],
                dtype=torch.bool,
                device=device,
            )
        self._validate_mask("global_visual_padding_mask", global_mask, global_tokens)
        self._validate_mask("wrist_visual_padding_mask", wrist_mask, wrist_tokens)
        return sanitize_padding_mask(
            torch.cat(
                [
                    global_mask.to(device=device, dtype=torch.bool),
                    wrist_mask.to(device=device, dtype=torch.bool),
                ],
                dim=1,
            )
        )

    @staticmethod
    def _validate_mask(name: str, mask: Tensor, tokens: Tensor) -> None:
        if mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"{name} must have shape [B, N] matching tokens; "
                f"got mask {tuple(mask.shape)} and tokens {tuple(tokens.shape)}."
            )
