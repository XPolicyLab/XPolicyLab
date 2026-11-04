"""Bimanual EE pose6D deployment profile (two Flexiv-style 10D arms = 20D).

Each arm is [xyz(3), rot6D first two rows(6), gripper(1)]. Delta integration
reuses the Flexiv single-arm pose6D rule independently on the left and right
halves. Cameras follow the RoboDojo / Piper three-view layout, not Flexiv.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from deploy.robot.common import derive_router_display_order
from deploy.robot.flexiv import integrate_delta_actions as integrate_arm10_delta

NAME = "dual_ee"
ARM_DIM = 10
STATE_DIM = ARM_DIM * 2
ACTION_DIM = STATE_DIM
ACTION_REPRESENTATION = "dual_ee_rot6d"
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
STATE_LAYOUT = (
    "left(xyz + rot6D first two rows + gripper) + "
    "right(xyz + rot6D first two rows + gripper)"
)


def integrate_delta_actions(raw_state: np.ndarray, delta_actions: np.ndarray) -> np.ndarray:
    state = np.asarray(raw_state, dtype=np.float32).reshape(STATE_DIM)
    delta = np.asarray(delta_actions, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[1] != STATE_DIM:
        raise ValueError(
            f"Expected bimanual pose6D delta action chunk shape [H,{STATE_DIM}], got {delta.shape}"
        )
    left = integrate_arm10_delta(state[:ARM_DIM], delta[:, :ARM_DIM])
    right = integrate_arm10_delta(state[ARM_DIM:], delta[:, ARM_DIM:])
    return np.concatenate([left, right], axis=1).astype(np.float32, copy=False)


def action_summary(state: np.ndarray, actions: np.ndarray) -> str:
    del state
    return (
        f"left_xyz_first={np.round(actions[0, :3], 4).tolist()} "
        f"right_xyz_first={np.round(actions[0, 10:13], 4).tolist()} "
        f"grip=({float(actions[0, 9]):.4f},{float(actions[0, 19]):.4f})"
    )


class DualEeRobot:
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
                f"dual_ee client_keys['cameras'] is missing model cameras {missing}. "
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
        return "delta_dual_ee_rot6d->absolute" if delta else "dual_ee_rot6d_absolute"

    def layouts(self) -> tuple[str, str]:
        return STATE_LAYOUT, "bimanual absolute xyz + rot6D(first two rows) + gripper"

    def policy_name(self, *, backbone: str) -> str:
        return f"{backbone}_{self.name}_policy"
