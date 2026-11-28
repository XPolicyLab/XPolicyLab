#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LambdaLR, LRScheduler, LinearLR, SequentialLR

from lerobot.datasets.utils import write_json
from lerobot.utils.constants import SCHEDULER_STATE
from lerobot.utils.io_utils import deserialize_json_into_object


@dataclass
class LRSchedulerConfig(draccus.ChoiceRegistry, abc.ABC):
    num_warmup_steps: int

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    name: str = "cosine"
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("vqbet")
@dataclass
class VQBeTSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_vqvae_training_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step):
            if current_step < self.num_vqvae_training_steps:
                return float(1)
            else:
                adjusted_step = current_step - self.num_vqvae_training_steps
                if adjusted_step < self.num_warmup_steps:
                    return float(adjusted_step) / float(max(1, self.num_warmup_steps))
                progress = float(adjusted_step - self.num_warmup_steps) / float(
                    max(1, num_training_steps - self.num_warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0.

    Automatically scales warmup and decay steps if num_training_steps < num_decay_steps.
    This ensures the learning rate schedule completes properly even with shorter training runs.
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        # Auto-scale scheduler parameters if training steps are shorter than configured decay steps
        actual_warmup_steps = self.num_warmup_steps
        actual_decay_steps = self.num_decay_steps

        if num_training_steps < self.num_decay_steps:
            # Calculate scaling factor to fit the schedule into the available training steps
            scale_factor = num_training_steps / self.num_decay_steps
            actual_warmup_steps = int(self.num_warmup_steps * scale_factor)
            actual_decay_steps = num_training_steps

            logging.info(
                f"Auto-scaling LR scheduler: "
                f"num_training_steps ({num_training_steps}) < num_decay_steps ({self.num_decay_steps}). "
                f"Scaling warmup: {self.num_warmup_steps} → {actual_warmup_steps}, "
                f"decay: {self.num_decay_steps} → {actual_decay_steps} "
                f"(scale factor: {scale_factor:.3f})"
            )

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (actual_warmup_steps + 1)
                frac = 1 - current_step / actual_warmup_steps
                return (1 / (actual_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                step = min(current_step, actual_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / actual_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < actual_warmup_steps:
                return linear_warmup_schedule(current_step)

            return cosine_decay_schedule(current_step)

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("fastwam_native")
@dataclass
class FastWAMNativeSchedulerConfig(LRSchedulerConfig):
    """Replicates the native FastWAM trainer scheduler behavior.

    Native FastWAM uses:
    - optional linear warmup from `1 / warmup_steps`
    - cosine annealing with `eta_min = peak_lr * min_lr_ratio`
    - warmup defaults to 5% of total optimizer steps when not specified
    """

    num_warmup_steps: int | None = None
    peak_lr: float = 1e-4
    min_lr_ratio: float = 0.01
    scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler:
        total_train_steps = max(int(num_training_steps), 1)
        if self.num_warmup_steps is None:
            warmup_steps = int(total_train_steps * float(self.warmup_ratio))
        else:
            warmup_steps = int(self.num_warmup_steps)
        warmup_steps = min(max(warmup_steps, 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        scheduler_type = str(self.scheduler_type).strip().lower()
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=remaining_steps,
                eta_min=float(self.peak_lr) * float(self.min_lr_ratio),
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported native MoT scheduler_type: {self.scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )


@LRSchedulerConfig.register_subclass("wsa_large_native")
@LRSchedulerConfig.register_subclass("WSA_Large_native")
@LRSchedulerConfig.register_subclass("tbot_sa1_wan_native")
@LRSchedulerConfig.register_subclass("TBot_SA1_Wan_native")
@dataclass
class WSALargeNativeSchedulerConfig(FastWAMNativeSchedulerConfig):
    """WSA_Large native trainer scheduler behavior."""


def save_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> None:
    state_dict = scheduler.state_dict()
    write_json(state_dict, save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler: LRScheduler, save_dir: Path) -> LRScheduler:
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler
