"""RobotProfile of RoboDojo's dual ARX-X5 embodiment (``env_cfg/arx_x5.yml``)."""

from __future__ import annotations

import numpy as np

from knowin_agent.backends.robodojo.kinematics import CHAIN_LENGTH_M, ArxKinematics
from knowin_agent.core.geometry import rot_x, rot_y, rot_z
from knowin_agent.runtime.backend import RobotProfile
from knowin_agent.robot_model import kinematics as robot_model

CAMERAS = {"head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
WRIST_CAMERA = {"left": "left_wrist", "right": "right_wrist"}
GRASP_OFFSET = (0.145, 0.0, 0.0)
FINGERTIP_BEYOND_GRASP_M = 0.013
TABLE_Z = 0.765
CONTROL_HZ = 25.0

NOTES = """\
Frames: RoboDojo env frame (metres, z up), the frame of the native ee poses and camera extrinsics.
Arm bases sit on the table edge, turned 90 deg about z: left (-0.30, -0.45, 0.765), right (+0.30, -0.45, 0.765).
Flange = link6: +x points from the wrist to the fingertips, the fingers close along +y.
At the start both grippers point +y (away from the robot) with the fingers side by side along world x.
Grasp point = flange + 0.145 m along +x (native gripper bias); fingertips 0.013 m beyond it.
Orientation (yaw, pitch, roll) deg: R = Rz(90 + yaw) @ Ry(pitch) @ Rx(roll); pitch 90 points straight down,
pitch 0 level; yaw 0 heads +y, +90 heads -x (robot's left).
Gripper command: 0 closed .. 1 open, a normalised command, not a measured width.
One control tick = one native take_action (25 Hz); the native IK may reject a pose, then that arm holds."""


def arx_orientation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Gripper orientation convention: Rz(90 + yaw) @ Ry(pitch) @ Rx(roll), degrees."""
    return rot_z(90.0 + yaw) @ rot_y(pitch) @ rot_x(roll)


WORKSPACE_MIN = (robot_model.TABLE_X[0], robot_model.TABLE_Y[0], robot_model.TABLE_Z)
WORKSPACE_MAX = (robot_model.TABLE_X[1], robot_model.TABLE_Y[1], round(robot_model.TABLE_Z + CHAIN_LENGTH_M, 3))


def arx_x5_profile(with_kinematics: bool = True) -> RobotProfile:
    return RobotProfile(
        name="robodojo_arx_x5",
        cameras=dict(CAMERAS),
        wrist_camera=dict(WRIST_CAMERA),
        grasp_offset=GRASP_OFFSET,
        tool_axis=(1.0, 0.0, 0.0),
        closing_axis=(0.0, 1.0, 0.0),
        fingertip_beyond_grasp_m=FINGERTIP_BEYOND_GRASP_M,
        table_z=TABLE_Z,
        control_hz=CONTROL_HZ,
        workspace_min=WORKSPACE_MIN,
        workspace_max=WORKSPACE_MAX,
        kinematics=ArxKinematics() if with_kinematics else None,
        orientation_fn=arx_orientation,
        notes=NOTES,
    )
