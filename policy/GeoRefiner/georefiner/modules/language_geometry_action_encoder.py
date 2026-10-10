"""Composite language-geometry-action encoder."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from georefiner.backbones.base import TextBackboneBase, VisualBackboneBase
from georefiner.config import GeoRefinerConfig
from georefiner.modules.action_encoder import ActionDynamics, ActionEncoder
from georefiner.modules.geometry_encoder import GeometryEncoder
from georefiner.modules.language_encoder import LanguageEncoder
from georefiner.modules.region_qformer import RegionQFormer
from georefiner.types import GeoRefinerInput


@dataclass(slots=True)
class LanguageGeometryActionEncoderOutput:
    """Outputs from the combined language-geometry-action encoder."""

    language_tokens: Tensor
    geometry_tokens: Tensor
    region_tokens: Tensor
    trajectory_tokens: Tensor
    dynamics: ActionDynamics
    language_padding_mask: Tensor | None = None
    geometry_padding_mask: Tensor | None = None


class LanguageGeometryActionEncoder(nn.Module):
    """Combine language, geometry, region, and temporal action encoders."""

    def __init__(
        self,
        config: GeoRefinerConfig,
        visual_backbone: VisualBackboneBase | None = None,
        text_backbone: TextBackboneBase | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.language_encoder = LanguageEncoder(config, text_backbone=text_backbone)
        self.geometry_encoder = GeometryEncoder(config, visual_backbone=visual_backbone)
        self.region_qformer = RegionQFormer(config)
        self.action_encoder = ActionEncoder(config)

    def forward(
        self,
        inputs: GeoRefinerInput,
        canonical_nominal_action: Tensor,
        action_feature: Tensor,
        state_feature: Tensor,
    ) -> LanguageGeometryActionEncoderOutput:
        language_output = self.language_encoder(
            instruction=inputs.instruction,
            language_tokens=inputs.language_tokens,
            padding_mask=inputs.language_padding_mask,
        )
        geometry_output = self.geometry_encoder(
            global_rgb=inputs.global_rgb,
            wrist_rgb=inputs.wrist_rgb,
            global_visual_tokens=inputs.global_visual_tokens,
            wrist_visual_tokens=inputs.wrist_visual_tokens,
            global_visual_padding_mask=inputs.global_visual_padding_mask,
            wrist_visual_padding_mask=inputs.wrist_visual_padding_mask,
        )
        region_output = self.region_qformer(
            geometry_tokens=geometry_output.geometry_tokens,
            language_tokens=language_output.language_tokens,
            geometry_padding_mask=geometry_output.padding_mask,
            language_padding_mask=language_output.padding_mask,
        )
        action_output = self.action_encoder(
            canonical_nominal_action=canonical_nominal_action,
            action_feature=action_feature,
            state_feature=state_feature,
            action_padding_mask=inputs.action_padding_mask,
        )
        return LanguageGeometryActionEncoderOutput(
            language_tokens=language_output.language_tokens,
            geometry_tokens=geometry_output.geometry_tokens,
            region_tokens=region_output.region_tokens,
            trajectory_tokens=action_output.trajectory_tokens,
            dynamics=action_output.dynamics,
            language_padding_mask=language_output.padding_mask,
            geometry_padding_mask=geometry_output.padding_mask,
        )
