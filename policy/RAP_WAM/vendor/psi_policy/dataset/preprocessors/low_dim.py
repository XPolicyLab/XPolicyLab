from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from psi_policy.dataset.preprocessors.base import BaseObsPreprocessor
from psi_policy.dataset.preprocessors.utils import shape_to_dim
from psi_policy.dataset.preprocessors.utils import to_float_2d_numpy


class LowDimPreprocessor(BaseObsPreprocessor):
    """将低维时序观测规范化为 [T,D] float tensor。"""

    def needs_normalizer_fit_value(
        self,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> bool:
        del key, obs_shape
        return True

    def process(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> torch.Tensor:
        expected_dim = shape_to_dim(obs_shape)
        return torch.from_numpy(
            to_float_2d_numpy(np.asarray(value), key=key, expected_dim=expected_dim)
        )

    def get_normalizer_fit_value(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> np.ndarray | None:
        expected_dim = shape_to_dim(obs_shape)
        return to_float_2d_numpy(np.asarray(value), key=key, expected_dim=expected_dim)


__all__ = ["LowDimPreprocessor"]
