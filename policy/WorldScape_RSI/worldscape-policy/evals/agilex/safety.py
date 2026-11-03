import math
from dataclasses import dataclass

import numpy as np


class SafetyError(RuntimeError):
    """Raised when a candidate robot trajectory violates a safety invariant."""


@dataclass(frozen=True)
class SafetyGuard:
    """Fail-closed validation for robot command trajectories."""

    max_trajectory_steps: int = 128
    max_position_step_m: float = 0.20
    max_joint_step_rad: float = math.pi
    gripper_min: float = -1e-6
    gripper_max: float = 0.080001

    def validate(
        self,
        left: list[list[float]],
        right: list[list[float]],
        *,
        action_mode: str,
        current_state: dict[str, np.ndarray] | None = None,
        enforce_motion_limits: bool = True,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if not left or len(left) != len(right):
            raise SafetyError("Left/right trajectories must be non-empty and equal")
        if len(left) > self.max_trajectory_steps:
            raise SafetyError("Trajectory exceeds the configured safety horizon")
        arrays = [np.asarray(side, dtype=np.float64) for side in (left, right)]
        if any(array.ndim != 2 or not np.isfinite(array).all() for array in arrays):
            raise SafetyError("Trajectory contains malformed or non-finite values")
        if action_mode == "eef":
            if any(array.shape[1] != 8 for array in arrays):
                raise SafetyError("EEF trajectory rows must be [xyz, quaternion, gripper]")
            if any(
                np.any(np.abs(np.linalg.norm(array[:, 3:7], axis=1) - 1.0) > 0.1)
                for array in arrays
            ):
                raise SafetyError("EEF trajectory contains a non-unit quaternion")
        elif action_mode == "joint":
            if arrays[0].shape[1] != arrays[1].shape[1]:
                raise SafetyError("Left/right joint command dimensions differ")
        else:
            raise SafetyError(f"Unsupported action mode: {action_mode!r}")
        if any(
            np.any(
                (array[:, -1] < self.gripper_min)
                | (array[:, -1] > self.gripper_max)
            )
            for array in arrays
        ):
            raise SafetyError("Gripper command is outside the allowed range")
        limit = (
            self.max_joint_step_rad
            if action_mode == "joint"
            else self.max_position_step_m
        )
        width = -1 if action_mode == "joint" else 3
        if any(
            len(array) > 1
            and np.linalg.norm(np.diff(array[:, :width], axis=0), axis=1).max()
            > limit
            for array in arrays
        ):
            message = "Trajectory contains an unsafe inter-step jump"
            if enforce_motion_limits:
                raise SafetyError(message)
            warnings.append(message)
        if current_state is not None:
            keys = (
                ("state.left_joint", "state.right_joint")
                if action_mode == "joint"
                else ("state.left_pos", "state.right_pos")
            )
            for side, array, key in zip(("left", "right"), arrays, keys):
                if key not in current_state:
                    raise SafetyError(f"Current robot state is missing {key}")
                current = np.asarray(current_state[key], dtype=np.float64).reshape(-1)
                command = array[0, :width]
                current = current[: command.shape[0]]
                if current.shape != command.shape:
                    raise SafetyError(f"Current robot state has invalid shape for {key}")
                distance = float(np.linalg.norm(command - current))
                if distance > limit:
                    message = (
                        "First command jumps too far from current state: "
                        f"side={side}, distance={distance:.4f}, limit={limit:.4f}, "
                        f"current={current.tolist()}, command={command.tolist()}"
                    )
                    if enforce_motion_limits:
                        raise SafetyError(message)
                    warnings.append(message)
        return tuple(warnings)


__all__ = ["SafetyError", "SafetyGuard"]
