"""Inference contract for the 20-second RoboDojo DM0.5-Mem checkpoint.

This mirrors
``dm05_mem_sft_robodojo_real_arx_x5_history_1fps20.py``: three current
camera views, 20 head-camera history slots sampled at 1 Hz, 16 soft visual
tokens per slot, 25 Hz observations, absolute 14-D actions, and a 50-step
model horizon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from opendm.constants.robot import ActionMode
from opendm.exp.dm05_exp import DM05DataConfig as _DM05DataConfig
from opendm.exp.dm05_exp import DM05Exp as _DM05Exp
from opendm.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from opendm.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig


ROBODOJO_IMAGE_KEYS = ["images_1", "images_2", "images_3"]
ROBODOJO_ACTION_DIM = 14


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    # The SFT inference config keeps FP32 weights, runs BF16 autocast, and
    # keeps the action expert in FP32 while explicitly selecting SDPA.
    bf16: bool = field(default=False)
    force_fp32_action_path: bool = field(default=True)
    llm_attn_implementation: str = field(default="sdpa")
    vision_attn_implementation: str = field(
        default_factory=lambda: os.environ.get("OPENDM_VISION_ATTN", "sdpa")
    )
    action_attn_implementation: str = field(default="sdpa")


@dataclass
class DM05DataConfig(_DM05DataConfig):
    action_mode: ActionMode = field(default=ActionMode.ABSOLUTE)
    is_history: bool = field(default=True)


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    enable_bf16_compute: bool = field(default=True)
    output_action_dim: int = field(default=ROBODOJO_ACTION_DIM)
    image_keys: list[str] = field(default_factory=lambda: list(ROBODOJO_IMAGE_KEYS))


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool | None = field(default=False)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    data_config: DM05DataConfig = field(default_factory=DM05DataConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)
