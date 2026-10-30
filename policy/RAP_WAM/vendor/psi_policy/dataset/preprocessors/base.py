from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
import torch


class BaseObsPreprocessor(ABC):
    """训练 dataset 层的按-key 预处理器。"""

    @abstractmethod
    def process(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> torch.Tensor:
        raise NotImplementedError

    def get_normalizer_fit_value(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> np.ndarray | None:
        del value, key, obs_shape
        return None

    def needs_normalizer_fit_value(
        self,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> bool:
        del key, obs_shape
        return False

    def get_normalizer(self, *, key: str) -> Any | None:
        del key
        return None

    def make_missing(
        self,
        *,
        key: str,
        steps: int,
        obs_shape: Sequence[int],
    ) -> torch.Tensor:
        shape = tuple(int(v) for v in obs_shape)
        raw = np.zeros((int(steps),) + shape, dtype=np.float32)
        return self.process(raw, key=key, obs_shape=shape)


__all__ = ["BaseObsPreprocessor"]
