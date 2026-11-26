"""Optional RoboDojo gripper hysteresis, migrated from WorldScapePolicy.

Configuration remains stdlib-only for the shell launcher's validation path.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class GripperDiscretizationConfig:
    enabled: bool = False
    close_threshold: float = 0.60
    open_threshold: float = 0.85
    min_closed_steps: int = 0
    open_confirmation_steps: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> GripperDiscretizationConfig:
        if value is None:
            value = {}
        defaults = asdict(cls())
        if not isinstance(value, Mapping) or set(value) - set(defaults):
            raise ValueError("gripper_discretization must be a mapping containing only "
                             + ", ".join(defaults))
        options = {**defaults, **value}
        if not isinstance(options["enabled"], bool):
            raise ValueError("gripper_discretization.enabled must be a boolean")
        for key in ("close_threshold", "open_threshold"):
            number = options[key]
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or not 0 <= number <= 1):
                raise ValueError(f"gripper_discretization.{key} must be finite and within [0, 1]")
        if options["close_threshold"] >= options["open_threshold"]:
            raise ValueError("gripper_discretization.close_threshold must be less than open_threshold")
        for key, minimum in (("min_closed_steps", 0), ("open_confirmation_steps", 1)):
            if type(options[key]) is not int or options[key] < minimum:
                raise ValueError(f"gripper_discretization.{key} must be an integer >= {minimum}")
        return cls(**options)


def resolve_gripper_config(
    config: Mapping[str, Any], provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Read only the enable flag from checkpoint; tuning belongs to deployment."""
    deployment = asdict(GripperDiscretizationConfig.from_mapping(config.get("gripper_discretization")))
    checkpoint = provenance.get("gripper_discretization", {})
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint gripper_discretization must be a mapping")
    unknown = set(checkpoint) - set(deployment)
    if unknown:
        raise ValueError(f"Unknown checkpoint gripper_discretization fields: {sorted(unknown)}")
    # Older checkpoints may contain tuning fields; ignore those values so the
    # deployment configuration remains authoritative for thresholds and timing.
    if "enabled" in checkpoint:
        deployment["enabled"] = checkpoint["enabled"]
    return asdict(GripperDiscretizationConfig.from_mapping(deployment))


@dataclass(frozen=True)
class GripperLatchState:
    closed: bool
    closed_steps_remaining: int = 0
    open_request_steps: int = 0


def discretize_gripper_action_chunk(
    actions: list[dict[str, np.ndarray]],
    *,
    current_state: Mapping[str, Any],
    config: GripperDiscretizationConfig,
    previous_states: Mapping[str, GripperLatchState] | None = None,
) -> tuple[list[dict[str, np.ndarray]], dict[str, GripperLatchState]]:
    """Convert continuous X5 gripper targets into latched open/closed commands."""

    import numpy as np

    previous_states = previous_states or {}
    states: dict[str, GripperLatchState] = {}
    for side in ("left", "right"):
        if side in previous_states:
            states[side] = previous_states[side]
            continue
        key = f"{side}_ee_joint_state"
        if key not in current_state:
            raise KeyError(f"Gripper discretization requires state[{key!r}]")
        observed = np.asarray(current_state[key], dtype=np.float64).reshape(-1)
        if observed.shape != (1,) or not np.isfinite(observed).all():
            raise ValueError(f"state.{key} must contain one finite value, got {observed}")
        states[side] = GripperLatchState(
            closed=float(observed[0]) <= config.close_threshold
        )

    discretized_actions: list[dict[str, np.ndarray]] = []
    for step, action in enumerate(actions):
        discretized_action = dict(action)
        for side in ("left", "right"):
            key = f"{side}_ee_joint_state"
            raw = np.asarray(action[key], dtype=np.float64).reshape(-1)
            if raw.shape != (1,) or not np.isfinite(raw).all():
                raise ValueError(f"action[{step}].{key} must contain one finite value, got {raw}")

            state = states[side]
            closed = state.closed
            closed_steps_remaining = state.closed_steps_remaining
            open_request_steps = state.open_request_steps
            if closed:
                if closed_steps_remaining > 0:
                    closed_steps_remaining -= 1
                    open_request_steps = 0
                elif float(raw[0]) >= config.open_threshold:
                    open_request_steps += 1
                    if open_request_steps >= config.open_confirmation_steps:
                        closed = False
                        open_request_steps = 0
                else:
                    open_request_steps = 0
            elif float(raw[0]) <= config.close_threshold:
                closed = True
                closed_steps_remaining = config.min_closed_steps
                open_request_steps = 0

            states[side] = GripperLatchState(
                closed=closed,
                closed_steps_remaining=closed_steps_remaining,
                open_request_steps=open_request_steps,
            )
            discretized_action[key] = np.asarray(
                [0.0 if closed else 1.0], dtype=np.float32
            )
        discretized_actions.append(discretized_action)
    return discretized_actions, states
