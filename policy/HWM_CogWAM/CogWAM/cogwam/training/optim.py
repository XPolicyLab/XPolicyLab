"""Optimizer and LR schedule for the released CogWAM recipe.

Per-module learning rates are the whole point of this file: the pretrained VLM
(``qwen_vl_interface``) is fine-tuned an order of magnitude slower (1e-5) than
the randomly initialized planner queries and World--Action MoT (1e-4), which
have to travel much further. AdamW runs with betas (0.9, 0.95) -- the low
second-moment horizon usual for large-batch transformer pretraining -- and a
``cosine_with_min_lr`` schedule that floors at ``min_lr`` instead of decaying to
zero, so the last thousands of steps still move the weights.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from transformers import get_scheduler

from cogwam.training.utils import build_param_lr_groups

logger = get_logger(__name__)


def setup_optimizer_and_scheduler(
    model,
    cfg,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build AdamW optimizer + LR scheduler from cfg.trainer.

    Supports:
    - Per-module learning rates via cfg.trainer.learning_rate
    - cosine_with_min_lr (or any transformers scheduler) via cfg.trainer.lr_scheduler_type
    """
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


__all__ = ["setup_optimizer_and_scheduler"]
