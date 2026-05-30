"""
Normalization utilities for Mem_0 arms (first 14 dims).
Gripper dims are assumed normalized externally to [0, 1].
"""

import json
import os
from typing import Dict

import numpy as np


def load_stats(stats_path: str) -> Dict[str, np.ndarray]:
    """
    Load state/action statistics from a JSON file.

    Supports two layouts:
    1) Mean/std file with keys: state_mean, state_std, action_mean, action_std.
    2) Quantile file with keys: {"observation.state": {q01, q99}, "action": {q01, q99}}.
    At least one layout must be present.
    """
    if not stats_path:
        raise ValueError("stats_path is empty")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"stats file not found: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stats: Dict[str, np.ndarray] = {
        "state_mean": None,
        "state_std": None,
        "action_mean": None,
        "action_std": None,
        "state_min": None,
        "state_max": None,
        "action_min": None,
        "action_max": None,
        "state_q01": None,
        "state_q99": None,
        "action_q01": None,
        "action_q99": None,
    }

    # Mean/std style
    mean_std_keys = ("state_mean", "state_std", "action_mean", "action_std")
    has_mean_std = all(key in data for key in mean_std_keys)
    if has_mean_std:
        for key in mean_std_keys:
            stats[key] = np.array(data[key], dtype=np.float32)

    # Min/max style
    min_max_keys = ("state_min", "state_max", "action_min", "action_max")
    has_min_max = all(key in data for key in min_max_keys)
    if has_min_max:
        for key in min_max_keys:
            stats[key] = np.array(data[key], dtype=np.float32)

    # Quantile style
    if "observation.state" in data and "action" in data:
        obs_block = data["observation.state"]
        act_block = data["action"]
        for q_key in ("q01", "q99"):
            if q_key in obs_block:
                stats[f"state_{q_key}"] = np.array(obs_block[q_key], dtype=np.float32)
            if q_key in act_block:
                stats[f"action_{q_key}"] = np.array(act_block[q_key], dtype=np.float32)

    has_quantile = stats["state_q01"] is not None and stats["state_q99"] is not None and stats["action_q01"] is not None and stats["action_q99"] is not None
    if not has_mean_std and not has_quantile and not has_min_max:
        raise KeyError(
            "stats file must contain mean/std keys or quantile keys (q01/q99) or min/max keys"
        )

    return stats


def normalize_arms(
    vec: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    min: np.ndarray | None,
    max: np.ndarray | None,
    arm_dims: int = 14,
    *,
    quantile: bool = False,
    q01: np.ndarray | None = None,
    q99: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalize vector.
    Maps dims to [-1, 1].
    """
    vec = np.asarray(vec, dtype=np.float32)
    out = vec.copy()

    if quantile and q01 is not None and q99 is not None:
        q01_arr = np.asarray(q01, dtype=np.float32)
        q99_arr = np.asarray(q99, dtype=np.float32)
        dims = min(out.shape[-1], q01_arr.shape[-1], q99_arr.shape[-1])
        q_range = q99_arr[:dims] - q01_arr[:dims]
        mask = np.where (q_range > 1e-8, 1.0, 0.0)
        q_range = np.where(q_range > 1e-8, q_range, 1.0)
        clipped = np.clip(out[..., :dims], q01_arr[:dims], q99_arr[:dims])
        out[..., :dims] = 2.0 * (clipped - q01_arr[:dims]) / q_range - 1.0
        return out * mask

    if mean is None or std is None:
        min_arr = np.asarray(min, dtype=np.float32)
        max_arr = np.asarray(max, dtype=np.float32)
        q_range = max_arr[:arm_dims] - min_arr[:arm_dims]
        mask = np.where (q_range > 1e-8, 1.0, 0.0)
        q_range = np.where(q_range > 1e-8, q_range, 1.0)
        clipped = np.clip(out[..., :arm_dims], min_arr[:arm_dims], max_arr[:arm_dims])
        out[..., :arm_dims] = 2.0 * (clipped - min_arr[:arm_dims]) / q_range - 1.0
        out[..., :arm_dims] *= mask
        return out

    safe_std = np.where(std[:arm_dims] < 1e-6, 1.0, std[:arm_dims])
    out[..., :arm_dims] = (out[..., :arm_dims] - mean[:arm_dims]) / safe_std
    return out


def denormalize_arms(
    vec: np.ndarray,
    mean: np.ndarray | None,
    std: np.ndarray | None,
    min: np.ndarray | None,
    max: np.ndarray | None,
    arm_dims: int = 14,
    *,
    quantile: bool = False,
    q01: np.ndarray | None = None,
    q99: np.ndarray | None = None,
) -> np.ndarray:
    """
    Denormalize vector.
    """
    vec = np.asarray(vec, dtype=np.float32)
    out = vec.copy()

    if quantile and q01 is not None and q99 is not None:
        q01_arr = np.asarray(q01, dtype=np.float32)
        q99_arr = np.asarray(q99, dtype=np.float32)
        dims = min(out.shape[-1], q01_arr.shape[-1], q99_arr.shape[-1])
        q_range = q99_arr[:dims] - q01_arr[:dims]
        q_range = np.where(q_range > 1e-8, q_range, 1.0)
        out[..., :dims] = 0.5 * (out[..., :dims] + 1.0) * q_range + q01_arr[:dims]
        out[..., :dims] = np.clip(out[..., :dims], q01_arr[:dims], q99_arr[:dims])
        return out

    if mean is None or std is None:
        min_arr = np.asarray(min, dtype=np.float32)
        max_arr = np.asarray(max, dtype=np.float32)
        q_range = max_arr[:arm_dims] - min_arr[:arm_dims]
        q_range = np.where(q_range > 1e-8, q_range, 1.0)
        out[..., :arm_dims] = 0.5 * (out[..., :arm_dims] + 1.0) * q_range + min_arr[:arm_dims]
        out[..., :arm_dims] = np.clip(out[..., :arm_dims], min_arr[:arm_dims], max_arr[:arm_dims])
        return out

    out[..., :arm_dims] = out[..., :arm_dims] * std[:arm_dims] + mean[:arm_dims]
    return out
