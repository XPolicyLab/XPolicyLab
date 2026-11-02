from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping

import torch
import torch.nn as nn


class BaseFusion(nn.Module, ABC):
    """融合插件协议：输入 token_map，返回更新后的 token_map。"""

    @abstractmethod
    def forward(
        self,
        *,
        token_map: Mapping[str, torch.Tensor],
        obs_dict: Mapping[str, torch.Tensor],
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


__all__ = ["BaseFusion"]
