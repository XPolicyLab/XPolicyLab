"""RoboTwin condition 3: the dual_arm_ee16 slow method through an EE16 adapter.

RoboTwin 0aeea2d ``take_action(action, action_type="ee")`` reads
``[l_xyz, l_quat, l_grip, r_xyz, r_quat, r_grip]`` (``envs/_base_task.py``); the
quaternions are transforms3d ``mat2quat``/``quat2mat`` order, i.e. wxyz
(``envs/robot/robot.py``), and the gripper is 1 open / 0 closed with the observed
value equal to the last command (``set_gripper`` stores it). OpenWAM labels the
same four numbers xyzw on the way in and out, so its fast path round-trips them
unchanged; everything here treats native as wxyz and reuses the RoboDojo
persona and primitives (``method.txt``, ``actions.Method``) on the 4-field dict.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .action_space import dual_to_ee16, ee16_to_dual, endpose_to_dual  # noqa: F401 - re-exported
from .actions import Method
from .contracts import plain
from .profiles import DUAL_ARM_EE16

# RoboTwin 0aeea2d envs/_base_task.py:223 sets the SAPIEN timestep to 1/250 s and
# task_config/demo_{clean,randomized}.yml record a demonstration frame every
# save_freq 15 steps: the checkpoint's data rate, and one native EE step, is 250/15 Hz.
CONTROL_HZ = 250 / 15


class RoboTwinDualArmMethod(Method):
    """``actions.Method`` on RoboTwin state; rows leave as native EE16 vectors."""

    def __init__(self, cfg):
        limits = cfg["primitives"]
        super().__init__({"max_chunk_steps": limits["chunk_rows"], "action_limits": {
            "frame": limits["frame"], "quaternion": limits["quaternion"],
            "left_xyz_bounds": limits["left_xyz_bounds"], "right_xyz_bounds": limits["right_xyz_bounds"]}})

    def seed(self, state):
        super().seed(endpose_to_dual(state))

    def acknowledged(self, action):
        super().acknowledged(ee16_to_dual(action))

    def primitives(self, params, state):
        return [dual_to_ee16(row) for row in super().primitives(params, endpose_to_dual(state))]


def png_data_url(image, size=(320, 240)) -> str:
    frame = Image.fromarray(np.asarray(image))
    frame.thumbnail(size, Image.Resampling.BOX)
    stream = io.BytesIO()
    frame.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def camera_payload(runtime) -> dict:
    """RoboTwin's three released RGB views, EE state and the policy's own commands only."""
    observation, method = runtime.observation, runtime.method
    images = [{"camera": name, "data_url": png_data_url(observation["vision"][name]["color"])}
              for name in DUAL_ARM_EE16.cameras if name in observation["vision"]]
    robot_state = {**plain(endpose_to_dual(observation["state"])),
                   **{arm + "_gripper_command": [value] for arm, value in method.gripper_commands.items()}}
    return {"observation_id": runtime.observation_id, "observation_seq": runtime.gate.step, "epoch": 0,
            "task_instruction": observation["instruction"], "robot_state": robot_state,
            "phase": method.phase, "target": method.target, "images": images,
            "previous_receipt": runtime.previous_receipt}
