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
        rule = GripperThreshold(
            task=str(entry["task"]),
            prompt=str(entry["prompt"]),
            left=float(entry["left"]),
            right=float(entry["right"]),
        )
        if not all(math.isfinite(value) and value >= 0 for value in (rule.left, rule.right)):
            raise ValueError(f"Invalid gripper threshold for {env_cfg_type}/{rule.task}")
        key = normalize_prompt(rule.prompt)
        if not key or key in thresholds:
            raise ValueError(f"Invalid or duplicate prompt for {env_cfg_type}/{rule.task}")
        thresholds[key] = rule
    return thresholds


def apply_gripper_thresholds(actions, rule):
    """Zero values below each threshold; scale values at/above it by 1.3."""
    filtered = actions.copy()
    for column, threshold in ((6, rule.left), (13, rule.right)):
        below = actions[:, column] < threshold
        filtered[below, column] = 0.0
        filtered[~below, column] *= 1.3
    return filtered
