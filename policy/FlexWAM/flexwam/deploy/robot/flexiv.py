"""Flexiv single-arm pose6D deployment profile."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from deploy.robot.common import derive_router_display_order

NAME = "flexiv"
STATE_DIM = 10
ACTION_DIM = 10
ACTION_REPRESENTATION = "flexiv_pose6d"
# Model keys (training camera names / processor view order) -> client websocket keys.
CLIENT_KEYS = {
    "state": "observation/state",
    "prompt": "prompt",
    "action": "actions",
    "cameras": {
        "global": "observation/global",
        "global2": "observation/global2",
        "left_wrist": "observation/left_wrist",
    },
}
CAMERA_ORDER = tuple(CLIENT_KEYS["cameras"])
STATE_LAYOUT = "Flexiv single-arm xyz + rot6D(first two rows) + gripper"


def _rot6d_rows_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(6)
    row0 = rot6d[0:3]
    row1 = rot6d[3:6]
    row0_norm = np.linalg.norm(row0)
    if row0_norm < 1e-12:
        raise ValueError("Invalid rot6D: first row norm is too small.")
    row0 = row0 / row0_norm
    row1 = row1 - np.dot(row0, row1) * row0
    row1_norm = np.linalg.norm(row1)
    if row1_norm < 1e-12:
        raise ValueError("Invalid rot6D: second row is degenerate.")
    row1 = row1 / row1_norm
    row2 = np.cross(row0, row1)
    return np.stack([row0, row1, row2], axis=0)


def _matrix_to_rot6d_rows(matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    return np.concatenate([mat[0, :], mat[1, :]]).astype(np.float32)


def integrate_delta_actions(raw_state: np.ndarray, delta_actions: np.ndarray) -> np.ndarray:
    state = np.asarray(raw_state, dtype=np.float32).reshape(STATE_DIM)
    delta = np.asarray(delta_actions, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != STATE_DIM:
        raise ValueError(
            f"Expected Flexiv pose6D delta action chunk shape [H,{STATE_DIM}], got {delta.shape}"
        )

    base_pos = state[0:3].astype(np.float64)
    base_rot = _rot6d_rows_to_matrix(state[3:9])
    actions = np.empty_like(delta, dtype=np.float32)
    for step_idx, row in enumerate(delta):
        delta_rot = _rot6d_rows_to_matrix(row[3:9])
        abs_rot = base_rot @ delta_rot
        actions[step_idx, 0:3] = (base_pos + row[0:3].astype(np.float64)).astype(np.float32)
        actions[step_idx, 3:9] = _matrix_to_rot6d_rows(abs_rot)
        actions[step_idx, 9:10] = row[9:10]
    return actions


def action_summary(state: np.ndarray, actions: np.ndarray) -> str:
    del state
    return (
        f"xyz_first={np.round(actions[0, :3], 4).tolist()} "
        f"xyz_last={np.round(actions[-1, :3], 4).tolist()} "
        f"grip_first={float(actions[0, 9]):.4f} grip_last={float(actions[-1, 9]):.4f}"
    )


class FlexivRobot:
    name = NAME
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    action_representation = ACTION_REPRESENTATION
    camera_order = CAMERA_ORDER
    client_keys = CLIENT_KEYS

    def resolve_action_representation(self, requested: str) -> str:
        del requested
        return ACTION_REPRESENTATION

    def resolve_camera_order(
        self,
        *,
        view_names: Sequence[str] | None = None,
        image_specs: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[str, ...]:
        mapped = tuple(self.client_keys["cameras"])
        if view_names:
            names = tuple(str(name) for name in view_names)
        elif image_specs:
            names = tuple(str(spec["key"]) for spec in image_specs)
        else:
            names = mapped
        missing = [name for name in names if name not in self.client_keys["cameras"]]
        if missing:
            raise ValueError(
                f"Flexiv client_keys['cameras'] is missing model cameras {missing}. "
                f"Known: {list(self.client_keys['cameras'])}."
            )
        return names

    def router_display_order(self, camera_order: tuple[str, ...]) -> tuple[str, ...]:
        return derive_router_display_order(camera_order)

    def integrate_delta(self, raw_state: np.ndarray, delta_actions: np.ndarray) -> np.ndarray:
        return integrate_delta_actions(raw_state, delta_actions)

    def action_summary(self, state: np.ndarray, actions: np.ndarray) -> str:
        return action_summary(state, actions)

    def action_log_label(self, *, delta: bool) -> str:
        return "delta_flexiv_pose6d->absolute" if delta else "flexiv_pose6d_absolute"

    def layouts(self) -> tuple[str, str]:
        return STATE_LAYOUT, "Flexiv single-arm absolute xyz + rot6D(first two rows) + gripper"

    def policy_name(self, *, backbone: str) -> str:
        return f"{backbone}_{self.name}_policy"
