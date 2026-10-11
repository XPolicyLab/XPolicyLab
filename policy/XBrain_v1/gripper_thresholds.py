import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GripperThreshold:
    task: str
    prompt: str
    left: float
    right: float
    left_scale: float = 1.3
    right_scale: float = 1.3
    action_horizon: int | None = None


def normalize_prompt(prompt):
    """Normalize harmless prompt formatting differences for exact task lookup."""
    return " ".join(str(prompt).casefold().split()).rstrip(".")


def load_gripper_thresholds(path, env_cfg_type):
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    entries = data.get(env_cfg_type)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No gripper thresholds configured for {env_cfg_type!r}")

    thresholds = {}
    for entry in entries:
        horizon = entry.get("action_horizon")
        if horizon is not None and (type(horizon) is not int or not 1 <= horizon <= 50):
            raise ValueError(f"Invalid action_horizon for {env_cfg_type}/{entry['task']}: {horizon!r}")
        rule = GripperThreshold(
            task=str(entry["task"]),
            prompt=str(entry["prompt"]),
            left=float(entry["left"]),
            right=float(entry["right"]),
            left_scale=float(entry.get("left_scale", 1.3)),
            right_scale=float(entry.get("right_scale", 1.3)),
            action_horizon=horizon,
        )
        if not all(math.isfinite(value) and value >= 0 for value in (rule.left, rule.right)):
            raise ValueError(f"Invalid gripper threshold for {env_cfg_type}/{rule.task}")
        if not all(math.isfinite(value) and value > 0 for value in (rule.left_scale, rule.right_scale)):
            raise ValueError(f"Invalid gripper scale for {env_cfg_type}/{rule.task}")
        key = normalize_prompt(rule.prompt)
        if not key or key in thresholds:
            raise ValueError(f"Invalid or duplicate prompt for {env_cfg_type}/{rule.task}")
        thresholds[key] = rule
    return thresholds


def apply_gripper_thresholds(actions, rule):
    """Zero values below each threshold; otherwise apply that arm's task scale."""
    filtered = actions.copy()
    for column, threshold, scale in ((6, rule.left, rule.left_scale),
                                     (13, rule.right, rule.right_scale)):
        below = actions[:, column] < threshold
        filtered[below, column] = 0.0
        filtered[~below, column] *= scale
    return filtered
