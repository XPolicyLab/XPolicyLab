"""Model architecture configuration for Focus-VLWA."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ExpertVariant = Literal["debug", "gemma_300m", "gemma_2b"]


@dataclass(frozen=True)
class ExpertConfig:
    """Transformer dimensions shared by the language, action, and world experts."""

    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_expert_config(variant: ExpertVariant) -> ExpertConfig:
    """Return the transformer dimensions for a supported expert variant."""
    if variant == "debug":
        return ExpertConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return ExpertConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return ExpertConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
    raise ValueError(f"Unsupported expert variant: {variant}")


@dataclass(frozen=True)
class FocusVLWAConfig:
    """Complete architecture definition for the released Focus-VLWA checkpoint."""

    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    vision_language_variant: ExpertVariant = "gemma_2b"
    action_expert_variant: ExpertVariant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 400
    history_mode: Literal["head_history"] = "head_history"
    use_discrete_state: bool = True
    use_world_model: bool = True
    world_model_horizon: int = 270
    world_model_chunks: int = 10
    world_model_head_cells: int = 9
    world_model_wrist_cells: int = 9
    world_model_dim: int = 32
    world_model_loss_weight: float = 1.0
    world_model_action_loss_weight: float = 0.0
    world_model_event_loss_weight: float = 1.0
    world_model_state_loss_weight: float = 0.3
    compile_mode: str | None = None

    def __post_init__(self) -> None:
        if self.history_mode != "head_history":
            raise ValueError(f"Unsupported history_mode: {self.history_mode}")
        expected_horizon = self.world_model_chunks * (self.world_model_head_cells + 2 * self.world_model_wrist_cells)
        if self.use_world_model and self.world_model_horizon != expected_horizon:
            raise ValueError(
                f"world_model_horizon={self.world_model_horizon} does not match "
                f"world_model_chunks * cells={expected_horizon}"
            )


def load_model_config(checkpoint: str | Path) -> FocusVLWAConfig:
    """Restore architecture settings from the checkpoint configuration."""
    path = Path(checkpoint) / "model_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Model configuration not found: {path}")
    return FocusVLWAConfig(**json.loads(path.read_text()))
