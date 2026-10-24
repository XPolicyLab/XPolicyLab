from __future__ import annotations

from typing import Any

import numpy as np
import torch

from psi_policy.model.common.normalizer import SingleFieldLinearNormalizer


class StreamingArrayStats:
    """Accumulates per-dimension stats without materializing the whole dataset."""

    def __init__(self, *, last_n_dims: int = 1):
        self.last_n_dims = int(last_n_dims)
        self._count = 0
        self._dim: int | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._sum: np.ndarray | None = None
        self._sum_sq: np.ndarray | None = None

    @property
    def count(self) -> int:
        return int(self._count)

    def update(self, value: np.ndarray | torch.Tensor | None) -> None:
        if value is None:
            return

        array = _reshape_stats_input(value, last_n_dims=self.last_n_dims)
        if array.shape[0] == 0:
            return

        if self._dim is None:
            self._dim = int(array.shape[1])
            self._min = array.min(axis=0)
            self._max = array.max(axis=0)
            self._sum = array.sum(axis=0, dtype=np.float64)
            self._sum_sq = np.square(array.astype(np.float64, copy=False)).sum(axis=0, dtype=np.float64)
            self._count = int(array.shape[0])
            return

        if int(array.shape[1]) != int(self._dim):
            raise ValueError(f"stats dim mismatch: expected {self._dim}, got {array.shape[1]}")

        assert self._min is not None
        assert self._max is not None
        assert self._sum is not None
        assert self._sum_sq is not None

        self._min = np.minimum(self._min, array.min(axis=0))
        self._max = np.maximum(self._max, array.max(axis=0))
        self._sum += array.sum(axis=0, dtype=np.float64)
        self._sum_sq += np.square(array.astype(np.float64, copy=False)).sum(axis=0, dtype=np.float64)
        self._count += int(array.shape[0])

    def build_normalizer(
        self,
        *,
        dtype=torch.float32,
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
        range_eps=1e-4,
        fit_offset=True,
    ) -> SingleFieldLinearNormalizer:
        if self._count <= 0:
            raise ValueError("cannot build normalizer without any observed values")

        assert self._min is not None
        assert self._max is not None
        assert self._sum is not None
        assert self._sum_sq is not None

        mean = self._sum / float(self._count)
        if self._count <= 1:
            std = np.full_like(mean, np.nan, dtype=np.float64)
        else:
            numerator = self._sum_sq - (self._sum * self._sum) / float(self._count)
            numerator = np.maximum(numerator, 0.0)
            std = np.sqrt(numerator / float(self._count - 1))

        return SingleFieldLinearNormalizer.create_from_input_stats(
            {
                "min": self._min,
                "max": self._max,
                "mean": mean,
                "std": std,
            },
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
        )


def pop_progress_kwargs(
    kwargs: dict[str, Any],
    *,
    default_desc: str,
) -> tuple[bool, str]:
    show_progress = bool(kwargs.pop("show_progress", True))
    raw_desc = kwargs.pop("progress_desc", default_desc)
    progress_desc = str(raw_desc).strip() or default_desc
    return show_progress, progress_desc


def _reshape_stats_input(
    value: np.ndarray | torch.Tensor,
    *,
    last_n_dims: int,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if last_n_dims < 0:
        raise ValueError(f"last_n_dims must be >= 0, got {last_n_dims}")

    dim = 1
    if last_n_dims > 0:
        dim = int(np.prod(array.shape[-last_n_dims:]))
    return array.reshape(-1, dim)
