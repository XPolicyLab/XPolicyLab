"""XPolicyLab StarVLA data registry overlay.

Only XPolicyLab-specific metadata lives here. The robot transforms and mixture
definitions continue to come from StarVLA's own Robotwin registry.
"""

from __future__ import annotations

from examples.Robotwin.train_files.data_registry.data_config import (
    ArxX5DataConfig as _StarVLAArxX5DataConfig,
)


class XPolicyArxX5DataConfig(_StarVLAArxX5DataConfig):
    state_key_dims = {
        "state.left_joints": 6,
        "state.right_joints": 6,
        "state.left_gripper": 1,
        "state.right_gripper": 1,
    }
    action_key_dims = {
        "action.left_joints": 6,
        "action.right_joints": 6,
        "action.left_gripper": 1,
        "action.right_gripper": 1,
    }


ROBOT_TYPE_CONFIG_MAP = {
    "arx_x5": XPolicyArxX5DataConfig(),
}
