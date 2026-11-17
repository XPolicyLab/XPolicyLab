"""Checkpoint normalization statistics and transformations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


def load_norm_stats(checkpoint_dir: str | Path, asset_id: str = "arx_x5_sim") -> dict[str, NormStats]:
    """Load normalization statistics stored beside a Focus-VLWA checkpoint."""
    path = Path(checkpoint_dir) / "assets" / asset_id / "norm_stats.json"
    if not path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {path}")
    payload = json.loads(path.read_text())["norm_stats"]
    return {
        name: NormStats(**{key: np.asarray(value) for key, value in values.items()}) for name, values in payload.items()
    }


def normalize_quantile(values: np.ndarray, stats: NormStats) -> np.ndarray:
    """Map values between the first and ninety-ninth quantiles to [-1, 1]."""
    dimension = values.shape[-1]
    q01 = stats.q01[..., :dimension]
    q99 = stats.q99[..., :dimension]
    return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def unnormalize_quantile(values: np.ndarray, stats: NormStats) -> np.ndarray:
    """Invert quantile normalization while preserving unsupervised padded dimensions."""
    dimension = stats.q01.shape[-1]
    restored = (values[..., :dimension] + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01
    if dimension == values.shape[-1]:
        return restored
    return np.concatenate([restored, values[..., dimension:]], axis=-1)
