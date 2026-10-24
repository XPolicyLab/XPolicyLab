from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ZScoreStatistics:
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.shape != self.mean.shape:
            raise ValueError("z-score mean/std must be matching one-dimensional arrays")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("z-score statistics contain non-finite values")
        if np.any(self.std <= 0):
            raise ValueError("z-score std values must be positive")


class GlobalZScoreNormalizer:
    """State/action z-score normalization using global mean/std statistics."""

    def __init__(
        self,
        statistics_path: str | Path,
        *,
        clip_range: tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        path = Path(statistics_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Z-score statistics do not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Z-score statistics are invalid JSON: {path}") from exc
        low, high = (float(value) for value in clip_range)
        if not np.isfinite((low, high)).all() or low >= high:
            raise ValueError("normalization clip_range must be finite and increasing")
        self.path = path.resolve()
        self.clip_range = (low, high)
        self.state = self._statistics(payload, "state")
        self.action = self._statistics(payload, "action")

    @staticmethod
    def _statistics(payload: Any, group: str) -> ZScoreStatistics:
        try:
            values = payload[group]["default"]
            mean = np.asarray(values["global_mean"], dtype=np.float32)
            std = np.asarray(values["global_std"], dtype=np.float32)
        except (KeyError, TypeError) as exc:
            raise KeyError(
                f"Z-score statistics require {group}.default.global_mean/global_std"
            ) from exc
        return ZScoreStatistics(mean=mean, std=std)

    def normalize_state(self, value: np.ndarray) -> np.ndarray:
        return self._normalize(value, self.state, group="state")

    def normalize_action(self, value: np.ndarray) -> np.ndarray:
        return self._normalize(value, self.action, group="action")

    def _normalize(
        self,
        value: np.ndarray,
        statistics: ZScoreStatistics,
        *,
        group: str,
    ) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape[-1] != statistics.mean.shape[0]:
            raise ValueError(
                f"{group} width {array.shape[-1]} does not match z-score "
                f"statistics width {statistics.mean.shape[0]}"
            )
        normalized = (array - statistics.mean) / statistics.std
        return np.clip(normalized, *self.clip_range).astype(np.float32, copy=False)

    def field_statistics(self, group: str) -> dict[str, list[float]]:
        statistics = self.state if group == "state" else self.action
        low, high = self.clip_range
        return {
            "mean": statistics.mean.astype(float).tolist(),
            "std": statistics.std.astype(float).tolist(),
            "clip_min": np.full(statistics.mean.shape, low).tolist(),
            "clip_max": np.full(statistics.mean.shape, high).tolist(),
        }


__all__ = ["GlobalZScoreNormalizer", "ZScoreStatistics"]
