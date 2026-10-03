from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .base_layers import FP32LayerNorm, ResidualMLP

from .decoder import MaskedTaxelCrossAttention, SpatialRefinementBlock
from .layers import PlainRelationBlock, fourier_coordinates
from .surface_encoder import SurfaceTokenizerV41


MANIFEEL_TOKEN_NAMES = (
    "right_tacff/top_left",
    "right_tacff/top_right",
    "right_tacff/bottom_left",
    "right_tacff/bottom_right",
)


@dataclass(frozen=True)
class ManiFeelTactileAEV41Config:
    """One 10x14 ManiFeel TacFF surface represented by four 48-D tokens."""

    regions: int = 1
    input_channels: int = 3
    height: int = 10
    width: int = 14
    surface_hidden_dim: int = 128
    latent_dim: int = 48
    tokens_per_region: int = 4
    surface_attention_heads: int = 4
    relation_attention_heads: int = 4
    relation_layers: int = 2
    relation_residual_scale_init: float = 0.1
    decoder_layers: int = 2
    decoder_refinement_blocks: int = 1
    decoder_refinement_hidden: int = 64
    ffn_dim: int = 192
    window_height: int = 5
    window_width: int = 7
    shift_height: int = 2
    shift_width: int = 3

    def __post_init__(self) -> None:
        fixed = {
            "regions": 1,
            "input_channels": 3,
            "height": 10,
            "width": 14,
            "latent_dim": 48,
            "tokens_per_region": 4,
            "window_height": 5,
            "window_width": 7,
            "decoder_refinement_blocks": 1,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(f"ManiFeel tactile AE V4.1 fixes {name}={expected}")
        if self.surface_hidden_dim % self.surface_attention_heads:
            raise ValueError("surface hidden size must be divisible by attention heads")
        if self.latent_dim % self.relation_attention_heads:
            raise ValueError("latent size must be divisible by relation attention heads")
        if not 0 <= self.shift_height < self.window_height:
            raise ValueError("invalid shifted-window height")
        if not 0 <= self.shift_width < self.window_width:
            raise ValueError("invalid shifted-window width")

    @property
    def frame_tokens(self) -> int:
        return self.tokens_per_region


class ManiFeelQuadrantRelationEncoderV41(nn.Module):
    """Plain self-attention across the four fixed spatial quadrants."""

    def __init__(self, config: ManiFeelTactileAEV41Config):
        super().__init__()
        self.quadrant_embedding = nn.Embedding(4, config.latent_dim)
        self.layers = nn.ModuleList(
            PlainRelationBlock(
                config.latent_dim,
                config.relation_attention_heads,
                config.ffn_dim,
                config.relation_residual_scale_init,
            )
            for _ in range(config.relation_layers)
        )
        self.output_norm = FP32LayerNorm(config.latent_dim)
        nn.init.trunc_normal_(self.quadrant_embedding.weight, std=0.02)
        self.register_buffer("quadrant_ids", torch.arange(4), persistent=True)

    def forward(self, tokens: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[1] != 4:
            raise ValueError("ManiFeel surface tokens must have shape [B,4,D]")
        if valid.shape != tokens.shape[:2]:
            raise ValueError("ManiFeel token mask must have shape [B,4]")
        values = (
            tokens + self.quadrant_embedding(self.quadrant_ids)[None].to(tokens.dtype)
        ) * valid[..., None]
        pre_relation = values
        for layer in self.layers:
            values = layer(values, valid)
        values = self.output_norm(values) * valid[..., None].to(values.dtype)
        return {
            "pre_relation_tokens": pre_relation,
            "frame_tokens": values,
            "frame_token_valid": valid,
        }


class ManiFeelForceDecoderV41(nn.Module):
    """Single decoder from four quadrant tokens back to one 10x14 force field."""

    def __init__(self, config: ManiFeelTactileAEV41Config):
        super().__init__()
        self.coordinate_embedding = nn.Sequential(
            nn.Linear(16, config.latent_dim),
            nn.SiLU(),
            nn.Linear(config.latent_dim, config.latent_dim),
        )
        self.cross_attention = MaskedTaxelCrossAttention(
            config.latent_dim, config.relation_attention_heads
        )
        self.layers = nn.ModuleList(
            ResidualMLP(config.latent_dim, config.ffn_dim)
            for _ in range(config.decoder_layers)
        )
        self.output_norm = FP32LayerNorm(config.latent_dim)
        self.refinement = SpatialRefinementBlock(
            config.latent_dim, config.decoder_refinement_hidden
        )
        self.force_head = nn.Conv2d(config.latent_dim, 3, 1)
        self.contact_head = nn.Conv2d(config.latent_dim, 1, 1)
        nn.init.normal_(self.force_head.weight, std=0.01)
        with torch.no_grad():
            self.force_head.bias.copy_(torch.tensor((-4.0, 0.0, 0.0)))
        nn.init.zeros_(self.contact_head.weight)
        nn.init.zeros_(self.contact_head.bias)
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, config.height),
            torch.linspace(-1.0, 1.0, config.width),
            indexing="ij",
        )
        self.register_buffer(
            "taxel_coordinates", torch.stack((x, y), -1).reshape(-1, 2),
            persistent=True,
        )
        self.config = config

    def forward(
        self,
        frame_tokens: torch.Tensor,
        token_valid: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = frame_tokens.shape[0]
        if frame_tokens.shape != (batch, 4, self.config.latent_dim):
            raise ValueError("ManiFeel frame tokens must have shape [B,4,D]")
        if token_valid.shape != (batch, 4):
            raise ValueError("ManiFeel token mask must have shape [B,4]")
        if support_mask.shape != (batch, 1, 1, 10, 14):
            raise ValueError("ManiFeel support mask must have shape [B,1,1,10,14]")

        coordinates = self.coordinate_embedding(fourier_coordinates(self.taxel_coordinates))
        taxels = coordinates[None].expand(batch, -1, -1)
        hidden = self.cross_attention(taxels, frame_tokens, token_valid)
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.output_norm(hidden).transpose(1, 2).reshape(
            batch, self.config.latent_dim, 10, 14
        )
        support = support_mask[:, 0].to(hidden.dtype)
        hidden = self.refinement(hidden, support)
        force_logits = self.force_head(hidden)
        raw = torch.cat(
            (force_logits[:, :1].sigmoid(), force_logits[:, 1:].tanh()), dim=1
        ) * support
        contact_logits = self.contact_head(hidden)
        probability = contact_logits.sigmoid() * support
        reconstructed = raw * probability
        return {
            "raw_reconstructed_force": raw[:, None],
            "reconstructed_force": reconstructed[:, None],
            "contact_logits": contact_logits[:, None],
            "contact_probability": probability[:, None],
        }


class ManiFeelSpatialSwinTactileAEV41(nn.Module):
    """Online single-surface ManiFeel codec with four fixed spatial tokens."""

    def __init__(
        self, config: ManiFeelTactileAEV41Config = ManiFeelTactileAEV41Config()
    ):
        super().__init__()
        self.surface_tokenizer = SurfaceTokenizerV41(config)  # structural duck type
        self.relation_encoder = ManiFeelQuadrantRelationEncoderV41(config)
        self.decoder = ManiFeelForceDecoderV41(config)
        self.config = config

    @staticmethod
    def _validate(force: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        if force.ndim != 5 or force.shape[1:] != (1, 3, 10, 14):
            raise ValueError("ManiFeel frame force must have shape [B,1,3,10,14]")
        if support.ndim == 4:
            support = support.unsqueeze(2)
        if support.shape != (force.shape[0], 1, 1, 10, 14):
            raise ValueError("ManiFeel support must have shape [B,1,1,10,14]")
        return support.bool()

    def encode_frame(
        self, force: torch.Tensor, support: torch.Tensor, *, return_details: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        support = self._validate(force, support)
        local = self.surface_tokenizer(force[:, 0], support[:, 0])
        surface_tokens = local["surface_tokens"]
        surface_valid = local["surface_token_valid"]
        relation = self.relation_encoder(surface_tokens, surface_valid)
        if not return_details:
            return relation["frame_tokens"]
        return {
            "surface_feature_map": local["surface_feature_map"][:, None],
            "quadrant_pooling_weights": local["quadrant_pooling_weights"][:, None],
            "surface_tokens": surface_tokens[:, None],
            "surface_token_valid": surface_valid[:, None],
            **relation,
        }

    def forward(self, force: torch.Tensor, support: torch.Tensor) -> dict[str, torch.Tensor]:
        if force.ndim == 6:
            return self.forward_sequence(force, support)
        support = self._validate(force, support)
        details = self.encode_frame(force, support, return_details=True)
        decoded = self.decoder(
            details["frame_tokens"], details["frame_token_valid"], support
        )
        return {**decoded, **details}

    def encode_sequence(self, force: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        if force.ndim != 6 or force.shape[2:] != (1, 3, 10, 14):
            raise ValueError("ManiFeel sequence force must have shape [B,T,1,3,10,14]")
        if support.ndim == 5:
            support = support.unsqueeze(3)
        batch, time = force.shape[:2]
        tokens = self.encode_frame(force.flatten(0, 1), support.flatten(0, 1))
        return tokens.reshape(batch, time, 4, self.config.latent_dim)

    def forward_sequence(self, force: torch.Tensor, support: torch.Tensor) -> dict:
        if force.ndim != 6 or force.shape[2:] != (1, 3, 10, 14):
            raise ValueError("ManiFeel sequence force must have shape [B,T,1,3,10,14]")
        if support.ndim == 5:
            support = support.unsqueeze(3)
        batch, time = force.shape[:2]
        output = self.forward(force.flatten(0, 1), support.flatten(0, 1))
        keys = (
            "raw_reconstructed_force",
            "reconstructed_force",
            "contact_logits",
            "contact_probability",
            "surface_tokens",
            "surface_token_valid",
            "pre_relation_tokens",
            "frame_tokens",
            "frame_token_valid",
        )
        result = {
            key: output[key].reshape(batch, time, *output[key].shape[1:])
            for key in keys
        }
        with torch.no_grad():
            baseline = self.surface_tokenizer(
                force.new_zeros(batch, 3, 10, 14), support[:, 0, 0]
            )["surface_tokens"]
        result["zero_surface_tokens"] = baseline[:, None].detach()
        return result

    def schema(self) -> dict:
        return {
            "architecture": "unified_spatial_swin_tactile_ae",
            "input_frame": "[B,1,3,10,14]",
            "support_mask": "[B,1,1,10,14] external structural mask",
            "input_channels": ["compressive_normal", "shear_u", "shear_v"],
            "latent_frame": "[B,4,48]",
            "latent_sequence": "[B,T,4,48]",
            "time_compression": False,
            "token_names": list(MANIFEEL_TOKEN_NAMES),
            "local_interaction": "masked W-MSA -> masked SW-MSA",
            "pooling": "fixed_quadrant_masked_content_weighted",
            "quadrant_relation": "plain_self_attention",
            "decoder_count": 1,
            "contact_gating": "soft",
            "config": asdict(self.config),
        }
