"""Typed post-training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PostTrainingConfig:
    """Minimal settings required to reproduce Focus-VLWA post-training."""

    dataset: str
    init_checkpoint: Path
    output_dir: Path
    norm_stats_dir: Path
    tokenizer_path: Path | None = None
    batch_size: int = 32
    num_workers: int = 4
    num_steps: int = 5_000
    seed: int = 42
    warmup_steps: int = 200
    decay_steps: int = 30_000
    peak_learning_rate: float = 2.5e-5
    final_learning_rate: float = 2.5e-6
    weight_decay: float = 1e-10
    gradient_clip_norm: float = 1.0
    log_interval: int = 100
    save_interval: int = 1_000
    world_model_loss_balance: bool = True
    history_mode: Literal["head_history"] | None = "head_history"
    max_token_len: int | None = None
    world_model_loss_weight: float | None = None
    world_model_event_loss_weight: float | None = None
    world_model_state_loss_weight: float | None = None
    freeze_world_model_expert: bool = False
    init_world_model_from_action: bool = False
    parameter_precision: Literal["float32"] | None = None
    resume_checkpoint: Path | None = None
    resume_batch_change_step: int | None = None
    resume_old_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.history_mode not in (None, "head_history"):
            raise ValueError(f"Unsupported history_mode: {self.history_mode}")
        for name in ("batch_size", "num_steps", "save_interval", "log_interval", "decay_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0 or self.warmup_steps < 0:
            raise ValueError("Worker count and warmup steps must be nonnegative")
        for name in ("world_model_loss_weight", "world_model_event_loss_weight", "world_model_state_loss_weight"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative")

        if self.decay_steps <= self.warmup_steps:
            raise ValueError("decay_steps must exceed warmup_steps")
        if self.resume_checkpoint is not None and self.init_world_model_from_action:
            raise ValueError("Resume cannot overwrite WM weights from the action expert")
        if (self.resume_batch_change_step is None) != (self.resume_old_batch_size is None):
            raise ValueError("Batch-change resume requires both anchor step and old global batch size")
        if self.resume_batch_change_step is not None:
            if self.resume_checkpoint is None or self.resume_batch_change_step < 0 or self.resume_old_batch_size < 1:
                raise ValueError("Invalid batch-change resume settings")


def stage_defaults(stage: str) -> dict:
    """Choose portable defaults for the joint and frozen-WM stages."""
    if stage not in ("joint", "frozen"):
        raise ValueError(f"Unsupported post-training stage: {stage}")
    return {
        "history_mode": "head_history",
        "max_token_len": 400,
        "num_steps": 5_000 if stage == "joint" else 10_000,
        "warmup_steps": 200,
        "decay_steps": 30_000 if stage == "joint" else 10_000,
        "peak_learning_rate": 2.5e-5,
        "final_learning_rate": 2.5e-6,
        "world_model_loss_weight": 1.0 if stage == "joint" else 0.0,
        "world_model_event_loss_weight": 1.0,
        "world_model_state_loss_weight": 0.3,
        "freeze_world_model_expert": stage == "frozen",
    }
