# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import torch
from robo_orchard_core.datatypes.geometry import BatchFrameTransform
from robo_orchard_core.kinematics.chain import (
    KinematicChain,
    KinematicSerialChain,
)
from robo_orchard_core.utils.math import (
    ensure_quaternion_sequence_continuous,
)

__all__ = [
    "ROBOTWIN_DEFAULT_BASE_QUAT",
    "ROBOTWIN_DEFAULT_BASE_XYZ",
    "RoboTwinEEF",
    "RoboTwinJointsToEEF",
]

ROBOTWIN_DEFAULT_BASE_XYZ = (0.0, -0.65, 0.0)
ROBOTWIN_DEFAULT_BASE_QUAT = (0.707, 0.0, 0.0, 0.707)


def _normalize_urdf_content(
    urdf_content: str | bytes | bytearray | memoryview,
) -> str:
    if isinstance(urdf_content, str):
        urdf_str = urdf_content
    else:
        urdf_str = bytes(urdf_content).decode("utf-8")

    stripped_urdf = urdf_str.lstrip()
    if stripped_urdf.startswith("<?xml"):
        declaration_end = stripped_urdf.find("?>")
        if declaration_end != -1:
            return stripped_urdf[declaration_end + 2 :].lstrip()
    return urdf_str


def _validate_vector(
    name: str,
    values: Sequence[float],
    *,
    expected_size: int,
) -> list[float]:
    ret = list(values)
    if len(ret) != expected_size:
        raise ValueError(
            f"Expected {name} to contain {expected_size} values, got "
            f"{len(ret)}."
        )
    return ret


@dataclass
class RoboTwinEEF:
    """World-frame RoboTwin end-effector transforms.

    `left_eef` and `right_eef` follow the local naming convention where a
    transform name without an `in_xx` suffix means `in_world`.
    """

    left_eef: BatchFrameTransform
    right_eef: BatchFrameTransform

    @property
    def left_eef_in_world(self) -> BatchFrameTransform:
        return self.left_eef

    @property
    def right_eef_in_world(self) -> BatchFrameTransform:
        return self.right_eef


class RoboTwinJointsToEEF:
    """Compute RoboTwin left and right end-effector poses from arm joints.

    The helper accepts one combined dual-arm URDF payload. The returned poses
    are expressed in world frame after the configured base transform(s) are
    composed on top of each arm-local forward kinematics result.

    Args:
        urdf_content (str | bytes): Combined dual-arm URDF content.
        left_eef_name (str, optional): Left end-effector frame name inside the
            URDF. Default is ``"fl_link6"``.
        right_eef_name (str, optional): Right end-effector frame name inside
            the URDF. Default is ``"fr_link6"``.
        robot_base_xyz (Sequence[float], optional): World-frame translation of
            the robot base root frame. Default is
            ``ROBOTWIN_DEFAULT_BASE_XYZ``.
        robot_base_quat (Sequence[float], optional): World-frame quaternion of
            the robot base root frame in ``(w, x, y, z)`` order. Default is
            ``ROBOTWIN_DEFAULT_BASE_QUAT``.
    """

    def __init__(
        self,
        *,
        urdf_content: str | bytes,
        left_eef_name: str = "fl_link6",
        right_eef_name: str = "fr_link6",
        robot_base_xyz: Sequence[float] = ROBOTWIN_DEFAULT_BASE_XYZ,
        robot_base_quat: Sequence[float] = ROBOTWIN_DEFAULT_BASE_QUAT,
    ) -> None:
        robot = KinematicChain.from_content(
            data=_normalize_urdf_content(urdf_content),
            format="urdf",
        )
        left_robot = robot
        right_robot = robot

        self._left_eef_name = left_eef_name
        self._right_eef_name = right_eef_name
        self._left_arm = KinematicSerialChain(left_robot, left_eef_name)
        self._right_arm = KinematicSerialChain(right_robot, right_eef_name)

        def _build_robot_base_tf(
            *,
            device: torch.device,
            dtype: torch.dtype,
            child_frame_id: str,
        ) -> BatchFrameTransform:
            return BatchFrameTransform(
                xyz=torch.tensor(
                    _validate_vector(
                        "robot_base_xyz",
                        robot_base_xyz,
                        expected_size=3,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                quat=torch.tensor(
                    _validate_vector(
                        "robot_base_quat",
                        robot_base_quat,
                        expected_size=4,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                parent_frame_id="world",
                child_frame_id=child_frame_id,
            )

        # Single robot base: use the same base transform for both arms.
        self._left_robot_base_tf = _build_robot_base_tf(
            device=self._left_arm.device,
            dtype=self._left_arm.dtype,
            child_frame_id=left_robot.frame_names[0],
        )
        self._right_robot_base_tf = _build_robot_base_tf(
            device=self._right_arm.device,
            dtype=self._right_arm.dtype,
            child_frame_id=right_robot.frame_names[0],
        )

    def transform(
        self,
        left_arm_joints: torch.Tensor,
        right_arm_joints: torch.Tensor,
    ) -> RoboTwinEEF:
        """Compute left and right EEF poses in world frame.

        Args:
            left_arm_joints (torch.Tensor): Left-arm joint tensor with shape
                ``(N, D_left)``.
            right_arm_joints (torch.Tensor): Right-arm joint tensor with shape
                ``(N, D_right)``.

        Returns:
            RoboTwinEEF: World-frame left and right end-effector poses.
        """
        left_arm_joints = left_arm_joints.to(
            device=self._left_arm.device,
            dtype=self._left_arm.dtype,
        )
        right_arm_joints = right_arm_joints.to(
            device=self._right_arm.device,
            dtype=self._right_arm.dtype,
        )
        left_eef_in_base_tf = self._left_arm.forward_kinematics_tf(
            left_arm_joints
        )[self._left_eef_name]
        left_eef_tf = self._left_robot_base_tf @ left_eef_in_base_tf
        right_eef_in_base_tf = self._right_arm.forward_kinematics_tf(
            right_arm_joints
        )[self._right_eef_name]
        right_eef_tf = self._right_robot_base_tf @ right_eef_in_base_tf
        left_eef_tf.quat = ensure_quaternion_sequence_continuous(
            left_eef_tf.quat
        )
        right_eef_tf.quat = ensure_quaternion_sequence_continuous(
            right_eef_tf.quat
        )
        return RoboTwinEEF(left_eef=left_eef_tf, right_eef=right_eef_tf)
