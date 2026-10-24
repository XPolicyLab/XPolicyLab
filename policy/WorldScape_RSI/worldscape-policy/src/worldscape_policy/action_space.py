from __future__ import annotations

import numpy as np

EEF_ARM_DIM = 10
EEF_DIMS = (EEF_ARM_DIM, EEF_ARM_DIM * 2)
EEF_ACTION_HORIZON = 24
JOINT_ACTION_DIM = 14


def parse_action_mode(value: str) -> str:
    """Validate the public action-space name."""

    mode = str(value).strip().lower()
    if mode in {"eef", "joint"}:
        return mode
    raise ValueError(
        f"unsupported action_mode {value!r}; expected 'eef' or 'joint'"
    )


def convert_eef_actions_to_relative(
    actions: np.ndarray,
    robot_state: np.ndarray,
    *,
    action_horizon: int = EEF_ACTION_HORIZON,
) -> np.ndarray:
    """Convert absolute EEF actions to chunk-relative EEF targets.

    Position is a direct 3D delta from the chunk anchor. Rotation is expressed
    in the anchor frame as ``R_relative = R_anchor.T @ R_action``.
    Gripper coordinates remain absolute.
    """

    action = _eef_array(actions, name="actions")
    state = _eef_array(robot_state, name="robot_state")
    if action.shape[-1] != state.shape[-1]:
        raise ValueError("actions and robot_state must use the same EEF width")
    if action_horizon <= 0:
        raise ValueError("action_horizon must be positive")
    references = _chunk_references(action, state, action_horizon=action_horizon)
    result = action.copy()
    for start in range(0, action.shape[-1], EEF_ARM_DIM):
        result[..., start : start + 3] -= references[..., start : start + 3]
        result[..., start + 3 : start + 9] = relative_rotation6d(
            action[..., start + 3 : start + 9],
            references[..., start + 3 : start + 9],
        )
    return result


def relative_rotation6d(
    target_rotation6d: np.ndarray,
    reference_rotation6d: np.ndarray,
) -> np.ndarray:
    target = rotation6d_to_matrix(target_rotation6d)
    reference = rotation6d_to_matrix(reference_rotation6d)
    relative = np.swapaxes(reference, -1, -2) @ target
    return matrix_to_rotation6d(relative)


def compose_rotation6d(
    reference_rotation6d: np.ndarray,
    relative_rotation6d_value: np.ndarray,
) -> np.ndarray:
    reference = rotation6d_to_matrix(reference_rotation6d)
    relative = rotation6d_to_matrix(relative_rotation6d_value)
    return matrix_to_rotation6d(reference @ relative)


def rotation6d_to_matrix(value: np.ndarray) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float32)
    if rotation.shape[-1] != 6:
        raise ValueError(f"rotation6d must end in dimension 6, got {rotation.shape}")
    columns = rotation.reshape(rotation.shape[:-1] + (3, 2))
    first = _normalize(columns[..., :, 0])
    second_raw = columns[..., :, 1]
    second = _normalize(
        second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    )
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def matrix_to_rotation6d(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in shape [3, 3], got {matrix.shape}")
    return matrix[..., :, :2].reshape(matrix.shape[:-2] + (6,))


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("rotation6d contains a degenerate basis vector")
    return value / norm


def _chunk_references(
    action: np.ndarray,
    state: np.ndarray,
    *,
    action_horizon: int,
) -> np.ndarray:
    action_steps = len(action)
    state_steps = len(state)
    if action_steps == 0 or state_steps == 0:
        raise ValueError("relative EEF conversion requires non-empty action and state")

    # Temporal packing stores one anchor state for every full action horizon.
    if state_steps != action_steps:
        if action_steps % state_steps != 0:
            raise ValueError(
                "packed relative EEF actions must contain an integer number of "
                "steps per anchor state"
            )
        steps_per_anchor = action_steps // state_steps
        if steps_per_anchor != action_horizon:
            raise ValueError(
                "packed relative EEF action/state ratio must equal action_horizon"
            )
        return np.repeat(state, steps_per_anchor, axis=0)

    # Non-packed samples carry state at every step. Each action horizon uses
    # the state at its first step, matching the original DreamZero convention.
    anchors = np.arange(action_steps) // action_horizon * action_horizon
    return state[np.minimum(anchors, state_steps - 1)]


def _eef_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[-1] not in EEF_DIMS:
        raise ValueError(
            f"{name} must have monomanual/bimanual EEF shape [T, 10|20], "
            f"got {array.shape}"
        )
    return array


__all__ = [
    "EEF_ACTION_HORIZON",
    "EEF_ARM_DIM",
    "EEF_DIMS",
    "JOINT_ACTION_DIM",
    "compose_rotation6d",
    "convert_eef_actions_to_relative",
    "matrix_to_rotation6d",
    "parse_action_mode",
    "relative_rotation6d",
    "rotation6d_to_matrix",
]
