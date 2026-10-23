from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from psi_policy.dataset.preprocessors.base import BaseObsPreprocessor


class PassthroughPreprocessor(BaseObsPreprocessor):
    """不做额外变换，保持原始张量语义。"""

    def process(
        self,
        value: np.ndarray,
        *,
        key: str,
        obs_shape: Sequence[int],
    ) -> torch.Tensor:
        del key, obs_shape
        return torch.from_numpy(np.asarray(value))


__all__ = ["PassthroughPreprocessor"]
