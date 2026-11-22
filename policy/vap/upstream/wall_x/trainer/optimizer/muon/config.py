"""Typed config for the legacy internal Muon optimizer."""

from __future__ import annotations

from dataclasses import dataclass

from wall_x.config.hyperparams_config import OptimizerConfig
from wall_x.config.registry import register_optimizer_config


@register_optimizer_config("muon")
@dataclass
class MuonConfig(OptimizerConfig):
    """Legacy Wall-X Muon optimizer config."""

    optimizer_type: str = "muon"
    beta_1: float = 0.9
    beta_2: float = 0.95
    weight_decay: float = 0.1
    eps: float = 1e-8


__all__ = ["MuonConfig"]
