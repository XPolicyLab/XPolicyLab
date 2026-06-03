# Copyright (C) 2026 Xiaomi Corporation.
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    max_lr: float = 4e-4,
    warmup_lr_start: float = 1e-6,
    min_lr: float = 8e-5,
    last_epoch: int = -1,
) -> LambdaLR:
    """Create a cosine-annealing LR schedule with linear warmup.

    During warmup the LR rises linearly from ``warmup_lr_start`` to ``max_lr``.
    After warmup the LR follows a cosine decay from ``max_lr`` down to ``min_lr``.

    Args:
        optimizer: Wrapped optimizer whose LR will be scheduled.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        max_lr: Peak learning rate after warmup.
        warmup_lr_start: Starting LR at step 0.  If negative, defaults to ``max_lr``.
        min_lr: Minimum (floor) learning rate after cosine decay.
        last_epoch: The index of the last epoch passed to ``LambdaLR``.

    Returns:
        A ``LambdaLR`` scheduler instance.
    """

    def lr_lambda(current_step: int) -> float:
        # --- Warmup phase: linear ramp ---
        warmup_lr_start_ = warmup_lr_start
        if current_step < num_warmup_steps:
            if warmup_lr_start_ < 0:
                warmup_lr_start_ = max_lr
            lr = min(
                max_lr,
                warmup_lr_start_ + (max_lr - warmup_lr_start_) * current_step / max(num_warmup_steps, 1),
            )
            return lr
        # --- Cosine decay phase ---
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        lr = (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)) + min_lr
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
