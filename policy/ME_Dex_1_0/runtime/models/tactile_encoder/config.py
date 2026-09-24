from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotwinTactileAEV41Config:
    """RoboTwin unified tactile architecture contract."""

    regions: int = 4
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
            "regions": 4,
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
                raise ValueError(f"RoboTwin tactile AE V4.1 fixes {name}={expected}")
        if self.height % self.window_height or self.width % self.window_width:
            raise ValueError("surface size must be divisible by the Swin window size")
        if not 0 <= self.shift_height < self.window_height:
            raise ValueError("shift_height must be in [0, window_height)")
        if not 0 <= self.shift_width < self.window_width:
            raise ValueError("shift_width must be in [0, window_width)")
        if self.surface_hidden_dim % self.surface_attention_heads:
            raise ValueError("surface hidden size must be divisible by attention heads")
        if self.latent_dim % self.relation_attention_heads:
            raise ValueError("latent size must be divisible by relation attention heads")
        if self.decoder_refinement_hidden % 8:
            raise ValueError("decoder refinement hidden size must be divisible by 8")
        if self.relation_layers < 1 or self.decoder_layers < 1:
            raise ValueError("relation and decoder layers must be positive")
        if not 0.0 <= self.relation_residual_scale_init <= 1.0:
            raise ValueError("relation residual scale must be in [0,1]")

    @property
    def frame_tokens(self) -> int:
        return self.regions * self.tokens_per_region


@dataclass(frozen=True)
class TactileAEV41LossConfig:
    """Four explicit tactile objectives and no hidden auxiliary losses."""

    force_weight: float = 1.0
    contact_weight: float = 1.0
    change_weight: float = 1.0
    latent_background_weight: float = 0.5
    background_force_weight: float = 0.5
    transition_boost: float = 4.0
    smooth_l1_beta: float = 0.1

    def __post_init__(self) -> None:
        weights = (
            self.force_weight,
            self.contact_weight,
            self.change_weight,
            self.latent_background_weight,
            self.background_force_weight,
            self.transition_boost,
        )
        if any(value < 0 for value in weights):
            raise ValueError("tactile loss weights must be non-negative")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")


@dataclass(frozen=True)
class ForceNormalizationV41Config:
    """Fixed, zero-preserving physical-force transform."""

    deadband_low_n: float = 1.0e-4
    deadband_high_n: float = 1.0e-3
    knee_n: tuple[float, float, float] = (0.01, 0.001, 0.001)
    upper_n: tuple[float, float, float] = (50.0, 5.0, 6.1)

    def __post_init__(self) -> None:
        if not 0 <= self.deadband_low_n < self.deadband_high_n:
            raise ValueError("deadband bounds must satisfy 0 <= low < high")
        if len(self.knee_n) != 3 or len(self.upper_n) != 3:
            raise ValueError("normalizer requires exactly three channel parameters")
        if any(value <= 0 for value in (*self.knee_n, *self.upper_n)):
            raise ValueError("normalizer knee and upper values must be positive")
