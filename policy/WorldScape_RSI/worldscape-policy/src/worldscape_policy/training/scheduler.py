"""Optimizer schedule primitives matching Hugging Face step semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NativeLRSchedulerConfig:
    schedule: str = "linear"
    warmup_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.schedule not in {"linear", "constant", "cosine"}:
            raise ValueError("schedule must be 'linear', 'constant', or 'cosine'")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")


def lr_factor(
    current_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    schedule: str = "linear",
) -> float:
    """Return the HF-style warmup schedule multiplier."""

    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    if schedule == "constant":
        return 1.0
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, float(current_step - warmup_steps) / float(decay_steps))
    if schedule == "cosine":
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return max(
        0.0,
        float(total_steps - current_step) / float(decay_steps),
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    config: NativeLRSchedulerConfig | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    value = config or NativeLRSchedulerConfig()
    warmup_steps = math.ceil(total_steps * value.warmup_ratio)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            schedule=value.schedule,
        ),
    )


__all__ = ["NativeLRSchedulerConfig", "build_lr_scheduler", "lr_factor"]
