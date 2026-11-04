"""Piper dual-arm joint deployment profile."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

NAME = "piper"
STATE_DIM = 14
ACTION_DIM = 14
ACTION_REPRESENTATION = "joint"
# Model keys (training camera names / processor view order) -> client websocket keys.
CLIENT_KEYS = {
    "state": "observation/state",
    "prompt": "prompt",
    "action": "actions",
    "cameras": {
        "global": "observation/global",
        "left_wrist": "observation/left_wrist",
        "right_wrist": "observation/right_wrist",
    },
}
CAMERA_ORDER = tuple(CLIENT_KEYS["cameras"])
ROUTER_DISPLAY_ORDER = ("left_wrist", "global", "right_wrist")
STATE_LAYOUT = "left(6 joints + 1 gripper width) + right(6 joints + 1 gripper width)"
JOINT_DELTA_MASK = np.asarray(
    [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
    dtype=bool,
)


def integrate_delta_actions(raw_state: np.ndarray, delta_actions: np.ndarray) -> np.ndarray:
    state = np.asarray(raw_state, dtype=np.float32).reshape(STATE_DIM)
    delta = np.asarray(delta_actions, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != STATE_DIM:
        raise ValueError(f"Expected joint delta action chunk shape [H,{STATE_DIM}], got {delta.shape}")
    actions = delta.copy()
    actions[:, JOINT_DELTA_MASK] = actions[:, JOINT_DELTA_MASK] + state[None, JOINT_DELTA_MASK]
    return actions.astype(np.float32, copy=False)


def action_summary(state: np.ndarray, actions: np.ndarray) -> str:
    if actions.ndim != 2 or actions.shape[1] < 14:
        return f"actions_shape={actions.shape}"

    first = actions[0]
    last = actions[-1]
    left_first_delta = float(np.linalg.norm(first[:6] - state[:6]))
    right_first_delta = float(np.linalg.norm(first[7:13] - state[7:13]))
    left_chunk_span = float(np.linalg.norm(np.ptp(actions[:, :6], axis=0)))
    right_chunk_span = float(np.linalg.norm(np.ptp(actions[:, 7:13], axis=0)))
    return (
        f"left(d0={left_first_delta:.4f}, span={left_chunk_span:.4f}, "
        f"grip={state[6]:.4f}->{first[6]:.4f}->{last[6]:.4f}) "
        f"right(d0={right_first_delta:.4f}, span={right_chunk_span:.4f}, "
        f"grip={state[13]:.4f}->{first[13]:.4f}->{last[13]:.4f})"
    )


class PiperRobot:
    name = NAME
    state_dim = STATE_DIM
    action_dim = ACTION_DIM
    action_representation = ACTION_REPRESENTATION
    camera_order = CAMERA_ORDER
    client_keys = CLIENT_KEYS

    def resolve_action_representation(self, requested: str) -> str:
        value = "joint" if requested == "auto" else requested
        if value not in {"joint", "flexiv_pose6d"}:
            raise ValueError(
                f"Invalid action representation: {value}. "
                "Expected 'joint', 'flexiv_pose6d', or 'auto'."
            )
        return value

    def resolve_camera_order(
        self,
        *,
        view_names: Sequence[str] | None = None,
        image_specs: Sequence[dict[str, Any]] | None = None,
    ) -> tuple[str, ...]:
        del view_names, image_specs
        return CAMERA_ORDER

    def router_display_order(self, camera_order: tuple[str, ...]) -> tuple[str, ...]:
        del camera_order
        return ROUTER_DISPLAY_ORDER

    def integrate_delta(self, raw_state: np.ndarray, delta_actions: np.ndarray) -> np.ndarray:
        return integrate_delta_actions(raw_state, delta_actions)

    def action_summary(self, state: np.ndarray, actions: np.ndarray) -> str:
        return action_summary(state, actions)

    def action_log_label(self, *, delta: bool) -> str:
        return "delta_joint->absolute" if delta else "left6+grip+right6+grip"

    def layouts(self) -> tuple[str, str]:
        return STATE_LAYOUT, STATE_LAYOUT

    def policy_name(self, *, backbone: str) -> str:
        return f"{backbone}_{self.name}_policy"
