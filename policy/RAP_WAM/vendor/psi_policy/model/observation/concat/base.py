from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Tuple

import torch
import torch.nn as nn


class BaseConcatStrategy(nn.Module, ABC):
    """拼接策略协议：输入 token_map，输出 [B,L,D] 与分段长度。"""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    def bind_input_dims(self, input_dims: Mapping[str, int]) -> None:
        del input_dims

    @abstractmethod
    def forward(
        self,
        *,
        token_map: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, list[int]]:
        raise NotImplementedError


__all__ = ["BaseConcatStrategy"]
