from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class FlexJointConfig:
    enabled: bool = False
    p_present_video: float = 1.0
    p_present_dino: float = 1.0
    p_present_pointmap: float = 1.0
    p_jv: float = 0.5
    p_jd: float = 0.5
    p_jp: float = 0.5
    cross_modal_predict_video: bool = False
    cross_modal_predict_dino: bool = False
    cross_modal_predict_pointmap: bool = False

    def __post_init__(self):
        for name in ('p_present_video', 'p_present_dino', 'p_present_pointmap', 'p_jv', 'p_jd', 'p_jp'):
            val = float(getattr(self, name))
            if not 0.0 <= val <= 1.0:
                raise ValueError(f'FlexJointConfig.{name} must be in [0, 1], got {val}')
            setattr(self, name, val)
