"""Small public RoboDojo observations for CPU-only tests."""

import numpy as np


def observation(value=0):
    return {"instruction": "Test", "state": {
        "left_arm_joint_state": np.full(6, value), "right_arm_joint_state": np.zeros(6),
        "left_ee_joint_state": np.ones(1), "right_ee_joint_state": np.ones(1),
        "left_ee_pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
        "right_ee_pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
    }, "vision": {"cam_head": {"color": np.zeros((8, 8, 3), dtype=np.uint8)}}}
