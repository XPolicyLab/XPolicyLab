"""Training state container.

TrainingState holds all mutable state shared between the trainer and callbacks
during training. It serves as the single source of truth for training progress.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class TrainingState:
    """Mutable training state shared across trainer and callbacks.

    Attributes:
        epoch: Current epoch (0-indexed).
        max_epochs: Total number of epochs.
        global_step: Global optimization step count.
        max_steps: Maximum number of optimization steps (0 = unlimited).
        batch_idx: Current batch index within the epoch.
        num_batches: Total batches per epoch.
        loss_dict: Loss components from the most recent forward pass.
        lr: Current learning rate.
        grad_norm: Gradient norm after clipping (if applicable).
        model: Reference to the model (set by trainer).
        optimizer: Reference to the optimizer (set by trainer).
        scheduler: Reference to the LR scheduler (set by trainer).
        train_dataloader: Reference to the training dataloader.
        eval_dataloader: Reference to the eval dataloader (if any).
        work_dir: Path to the experiment work directory.
        config: The resolved OmegaConf config.
        timestamp: Training start timestamp.
        extras: Arbitrary key-value store for callbacks to communicate.
    """

    # Progress
    epoch: int = 0
    max_epochs: int = 1
    global_step: int = 0
    max_steps: int = 0
    batch_idx: int = 0
    num_batches: int = 0

    # Current batch info
    loss_dict: Dict[str, Any] = field(default_factory=dict)
    lr: float = 0.0
    grad_norm: Optional[float] = None
    batch: Optional[Any] = None

    # References (set by trainer during init)
    model: Optional[Any] = None
    optimizer: Optional[Any] = None
    scheduler: Optional[Any] = None
    train_dataloader: Optional[Any] = None
    eval_dataloader: Optional[Any] = None

    # Experiment info
    work_dir: str = ""
    config: Optional[Any] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y%m%d_%H%M%S"))

    # Arbitrary extras for callbacks to communicate
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Training progress as a fraction in [0, 1]."""
        if self.max_steps > 0:
            return min(self.global_step / self.max_steps, 1.0)
        if self.max_epochs > 0 and self.num_batches > 0:
            total = self.max_epochs * self.num_batches
            current = self.epoch * self.num_batches + self.batch_idx
            return min(current / total, 1.0)
        return 0.0

    @property
    def iter(self) -> int:
        """Alias for global_step, compatible with MMEngine runner.iter."""
        return self.global_step

    @property
    def max_iters(self) -> int:
        """Total number of iterations."""
        if self.max_steps > 0:
            return self.max_steps
        return self.max_epochs * self.num_batches

    @property
    def is_last_batch(self) -> bool:
        """Whether this is the last batch of the current epoch."""
        return self.batch_idx >= self.num_batches - 1

    @property
    def is_last_epoch(self) -> bool:
        """Whether this is the last epoch."""
        return self.epoch >= self.max_epochs - 1

    @property
    def should_stop(self) -> bool:
        """Whether training should stop (max_steps reached)."""
        if self.max_steps > 0:
            return self.global_step >= self.max_steps
        return False

    def log_scalar(self, key: str, value: float) -> None:
        """Store a scalar for logging callbacks to pick up."""
        self.extras.setdefault("_scalars", {})[key] = value

    def get_scalars(self) -> Dict[str, float]:
        """Retrieve all pending scalars."""
        return self.extras.get("_scalars", {})

    def clear_scalars(self) -> None:
        """Clear pending scalars after they have been logged."""
        self.extras.pop("_scalars", None)
