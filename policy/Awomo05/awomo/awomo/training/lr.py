# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import logging
from math import ceil
from typing import Any, Dict, Optional
import torch
from awomo.awomo.training.optim.scheduler import OptimizerParamScheduler


def build_lr_scheduler(
    cfg,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> Optional[OptimizerParamScheduler]:
    if cfg is None:
        return None

    user_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)

    base_lr = optimizer.param_groups[0]["lr"]
    base_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    default_cfg: Dict[str, Any] = {
        "optimizer": optimizer,
        "lr_warmup_steps": min(1000, total_steps // 10),
        "lr_decay_steps": total_steps,
        "lr_decay_style": "cosine",
        "init_lr": base_lr * 0.1,
        "max_lr": base_lr,
        "min_lr": base_lr * 0.01,
        "start_wd": base_wd,
        "end_wd": base_wd,
        "wd_incr_steps": total_steps,
        "wd_incr_style": "constant",
    }

    if "lr_warmup_steps" in user_cfg:
        warmup = user_cfg["lr_warmup_steps"]
        if isinstance(warmup, float) and 0 < warmup < 1:
            user_cfg["lr_warmup_steps"] = ceil(warmup * total_steps)

    if user_cfg.get("lr_decay_style") == "WSD":
        default_cfg["wsd_decay_steps"] = max(1, total_steps // 10)
        default_cfg["lr_wsd_decay_style"] = "cosine"

    default_cfg.update(user_cfg)
    if default_cfg["lr_warmup_steps"] > 0 and "init_lr" not in user_cfg:
        default_cfg["init_lr"] = 0.0

    if default_cfg["lr_warmup_steps"] == 0:
        default_cfg["init_lr"] = default_cfg["max_lr"]

    if default_cfg["lr_warmup_steps"] > default_cfg["lr_decay_steps"]:
        default_cfg["lr_warmup_steps"] = default_cfg["lr_decay_steps"]

    logging.info(
        f"[INFO] LR Scheduler: style={default_cfg['lr_decay_style']}, "
        f"warmup={default_cfg['lr_warmup_steps']}, total={default_cfg['lr_decay_steps']}, "
        f"max_lr={default_cfg['max_lr']}, min_lr={default_cfg['min_lr']}"
    )

    scheduler = OptimizerParamScheduler(
        optimizer=default_cfg["optimizer"],
        init_lr=default_cfg["init_lr"],
        max_lr=default_cfg["max_lr"],
        min_lr=default_cfg["min_lr"],
        lr_warmup_steps=default_cfg["lr_warmup_steps"],
        lr_decay_steps=default_cfg["lr_decay_steps"],
        lr_decay_style=default_cfg["lr_decay_style"],
        start_wd=default_cfg["start_wd"],
        end_wd=default_cfg["end_wd"],
        wd_incr_steps=default_cfg["wd_incr_steps"],
        wd_incr_style=default_cfg["wd_incr_style"],
        wsd_decay_steps=default_cfg.get("wsd_decay_steps"),
        lr_wsd_decay_style=default_cfg.get("lr_wsd_decay_style"),
    )
    scheduler.step(0)
    return scheduler
