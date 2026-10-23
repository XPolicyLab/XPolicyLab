import json
import math
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class EpisodeJoint2Policy:
    cutoff: float
    selected_threshold: float
    selected_scale: float
    other_threshold: float
    other_scale: float


@dataclass(frozen=True)
class GripperThreshold:
    task: str
    prompt: str
    left: float
    right: float
    left_scale: float = 1.3
    right_scale: float = 1.3
    action_horizon: int | None = None
    episode_joint2: EpisodeJoint2Policy | None = None


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
        episode_policy = None
        if "episode_joint2" in entry:
            values = entry["episode_joint2"]
            if not isinstance(values, dict) or set(values) != {
                "cutoff", "selected_threshold", "selected_scale", "other_threshold", "other_scale"
            }:
                raise ValueError(f"Invalid episode_joint2 policy for {env_cfg_type}/{entry['task']}")
            episode_policy = EpisodeJoint2Policy(**{k: float(v) for k, v in values.items()})
            if not all(math.isfinite(v) for v in vars(episode_policy).values()):
                raise ValueError("episode_joint2 values must be finite")
            if min(episode_policy.selected_threshold, episode_policy.other_threshold) < 0:
                raise ValueError("episode_joint2 thresholds must be nonnegative")
            if min(episode_policy.selected_scale, episode_policy.other_scale) <= 0:
                raise ValueError("episode_joint2 scales must be positive")
        rule = GripperThreshold(
            task=str(entry["task"]),
            prompt=str(entry["prompt"]),
            left=float(entry["left"]),
            right=float(entry["right"]),
            left_scale=float(entry.get("left_scale", 1.3)),
            right_scale=float(entry.get("right_scale", 1.3)),
            action_horizon=horizon,
            episode_joint2=episode_policy,
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


def resolve_episode_gripper_rule(rule, state):
    """Choose an unambiguous qualifying arm, or keep the neutral pending rule."""
    policy = rule.episode_joint2
    if policy is None:
        return rule, None
    left_high = float(state[1]) > policy.cutoff
    right_high = float(state[8]) > policy.cutoff
    if left_high == right_high:
        # Neither qualifying is normal before the threshold is reached. Both
        # qualifying in the same received sample doesn't reveal which was first.
        return replace(rule, episode_joint2=None), None
    selected = "left" if left_high else "right"
    resolved = replace(
        rule,
        left=policy.selected_threshold if left_high else policy.other_threshold,
        right=policy.selected_threshold if right_high else policy.other_threshold,
        left_scale=policy.selected_scale if left_high else policy.other_scale,
        right_scale=policy.selected_scale if right_high else policy.other_scale,
        episode_joint2=None,
    )
    return resolved, selected


def apply_gripper_thresholds(actions, rule):
    """Zero values below each threshold; otherwise apply that arm's task scale."""
    if rule.episode_joint2 is not None:
        raise RuntimeError(f"Resolve {rule.task} to the pending or selected gripper rule first")
    filtered = actions.copy()
    for column, threshold, scale in ((6, rule.left, rule.left_scale),
                                     (13, rule.right, rule.right_scale)):
        below = actions[:, column] < threshold
        filtered[below, column] = 0.0
        filtered[~below, column] *= scale
    return filtered
