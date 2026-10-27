"""Opt-in training budget and scheduler for the immutable Robot37 view plan."""

import math
import re
from pathlib import Path

from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LRScheduler


MULTIVIEW_RESUME_SCHEMA = "phoenix.local_joint96.multiview.resume.v1"
MULTIVIEW_READER = "wam.datasets.local_robot37_multiview.Robot37MultiViewDataset"


def multiview_enabled(cfg):
    return bool(cfg.get("local_robot_multiview", {}).get("enabled", False))


def validate_multiview_config(cfg):
    """Keep the new budget exception confined to its explicit reader/recipe."""
    if not multiview_enabled(cfg):
        return
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    select = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    required = {
        "robot37.enabled": True,
        "local_robot_joint96.enabled": True,
        "local_robot_softprompt.enabled": True,
        "model.joint96_three_view": True,
        "data.train._target_": MULTIVIEW_READER,
        "data.train.verify_pair_acceptance": True,
        "data.sampling.global_sample_stride": 1,
        "data.sampling.training_block_size": 4,
        "local_robot_multiview.extension_lr_policy": "hold_minimum",
        "lr_scheduler_type": "cosine",
        "gradient_accumulation_steps": 1,
        "init_checkpoint": None,
        "local_robot_softprompt.init_checkpoint": None,
    }
    for key, expected in required.items():
        if select(key) != expected:
            raise ValueError(f"Multiview recipe requires {key}={expected!r}")
    rgb_keys = ('rgb_variants_root', 'rgb_variants_sha256',
                'rgb_selection_root', 'rgb_selection_sha256')
    rgb_enabled = any(
        select(prefix + key) is not None
        for prefix in ('local_robot_multiview.', 'data.train.', 'data.val.')
        for key in rgb_keys
    )
    if rgb_enabled:
        if select('data.val._target_') != MULTIVIEW_READER:
            raise ValueError('RGB exclusions require the validation selection reader')
        for key in rgb_keys:
            value = select('local_robot_multiview.' + key)
            valid = (isinstance(value, str) and bool(value)
                     and (Path(value).is_absolute() if key.endswith('_root')
                          else bool(re.fullmatch(r'[0-9a-f]{64}', value))))
            if not valid:
                raise ValueError('RGB selection requires an explicit artifact path/hash: ' + key)
            for split in ('train', 'val'):
                if select(f'data.{split}.{key}') != value:
                    raise ValueError('RGB artifact differs between train/validation/recipe: ' + key)
        if select('data.val.multiview_plan_root') != select('local_robot_multiview.plan_root'):
            raise ValueError('Validation exclusions must refer to the same parent plan')
    elif select('data.val._target_') not in (
        'wam.datasets.local_robot37.Robot37Dataset', MULTIVIEW_READER,
    ):
        raise ValueError('Multiview requires the fixed Robot37 validation reader')
    for key in ("plan_root", "plan_sha256"):
        value = select("local_robot_multiview." + key)
        if not isinstance(value, str) or not value:
            raise ValueError("An explicit published multiview plan and SHA-256 are required")
    if not Path(select("local_robot_multiview.plan_root")).is_absolute():
        raise ValueError("Multiview plan_root must be an absolute path")
    if not re.fullmatch(r"[0-9a-f]{64}", select("local_robot_multiview.plan_sha256")):
        raise ValueError("Multiview plan_sha256 must be a lowercase SHA-256")
    if select("data.train.multiview_plan_root") != select("local_robot_multiview.plan_root"):
        raise ValueError("Training reader and recipe must use the same multiview plan")
    horizon = select("local_robot_multiview.schedule_steps")
    budget = select("max_steps")
    warmup = select("warmup_steps")
    if type(horizon) is not int or type(budget) is not int or horizon <= 0 or budget <= 0:
        raise ValueError("Multiview schedule_steps and max_steps must be positive integers")
    if type(warmup) is not int or not 0 <= warmup < horizon:
        raise ValueError("Multiview warmup must be shorter than the fixed schedule")


def validate_multiview_budget(saved_contract, cfg):
    """Allow extending the stop budget without changing any training identity."""
    saved_budget = saved_contract.get("run_budget", {}).get("max_steps")
    requested_budget = cfg.max_steps
    if type(saved_budget) is not int or type(requested_budget) is not int:
        raise ValueError("Multiview resume requires integer saved/requested budgets")
    if saved_budget <= 0 or requested_budget < saved_budget:
        raise ValueError("Multiview max_steps may stay unchanged or increase, never decrease")


def immutable_multiview_contract(contract):
    return {key: value for key, value in contract.items() if key != "run_budget"}


class CappedWarmupCosineLR(LRScheduler):
    """Linear warmup, fixed-horizon cosine, then a permanent minimum LR.

    ``last_epoch`` counts completed optimizer updates, as in the existing
    Joint96 checkpoint contract. Extending max_steps never changes this horizon.
    """

    def __init__(self, optimizer, *, schedule_steps, warmup_steps, min_lr):
        self.schedule_steps = int(schedule_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)
        if not 0 <= self.warmup_steps < self.schedule_steps:
            raise ValueError("Expected 0 <= warmup_steps < schedule_steps")
        if not 0 < self.min_lr <= min(float(group["lr"]) for group in optimizer.param_groups):
            raise ValueError("Minimum LR must be positive and no larger than any peak LR")
        super().__init__(optimizer)

    def get_lr(self):
        step = max(int(self.last_epoch), 0)
        if self.warmup_steps and step < self.warmup_steps:
            fraction = step / self.warmup_steps
            return [self.min_lr + (peak - self.min_lr) * fraction for peak in self.base_lrs]
        if step >= self.schedule_steps:
            return [self.min_lr for _ in self.base_lrs]
        progress = (step - self.warmup_steps) / (self.schedule_steps - self.warmup_steps)
        fraction = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (peak - self.min_lr) * fraction for peak in self.base_lrs]
