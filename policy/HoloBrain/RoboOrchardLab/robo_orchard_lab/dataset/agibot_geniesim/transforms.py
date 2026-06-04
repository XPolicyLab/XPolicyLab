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

import numpy as np
import torch

from robo_orchard_lab.dataset.horizon_manipulation.transforms import (
    CalibrationToExtrinsic,
    MultiArmKinematics,
)

__all__ = [
    "GenieSim3CalibrationToExtrinsic",
    "GenieSim3Kinematics",
    "ZeroRobotState",
]


class GenieSim3CalibrationToExtrinsic(CalibrationToExtrinsic):
    """Calibration transform that supports GenieSim3 head/body chains."""

    def __init__(
        self,
        urdf,
        arm_link_keys,
        arm_joint_id,
        finger_keys=None,
        ee_to_gripper=None,
        head_link_keys=None,
        head_joint_id=None,
        body_link_keys=None,
        body_joint_id=None,
        **kwargs,
    ):
        num_arms = len(arm_link_keys)

        full_link_keys = list(arm_link_keys)
        full_joint_ids = list(arm_joint_id)
        full_finger_keys = (
            list(finger_keys)
            if finger_keys is not None
            else [[]] * num_arms
        )
        full_ee_to_gripper = (
            list(ee_to_gripper)
            if ee_to_gripper is not None
            else [None] * num_arms
        )

        if head_link_keys is not None:
            full_link_keys.append(head_link_keys)
            full_joint_ids.append(head_joint_id or [])
            full_finger_keys.append([])
            full_ee_to_gripper.append(None)

        if body_link_keys is not None:
            full_link_keys.append(body_link_keys)
            full_joint_ids.append(body_joint_id or [])
            full_finger_keys.append([])
            full_ee_to_gripper.append(None)

        super().__init__(
            urdf=urdf,
            arm_link_keys=full_link_keys,
            arm_joint_id=full_joint_ids,
            finger_keys=full_finger_keys,
            ee_to_gripper=full_ee_to_gripper,
            **kwargs,
        )


class GenieSim3Kinematics(MultiArmKinematics):
    def __init__(
        self,
        urdf,
        arm_link_keys,
        arm_joint_id,
        finger_keys=None,
        ee_to_gripper=None,
        head_link_keys=None,
        head_joint_id=None,
        body_link_keys=None,
        body_joint_id=None,
    ):
        self._orig_num_arms = len(arm_link_keys)
        self._has_head = head_link_keys is not None
        self._has_body = body_link_keys is not None

        full_link_keys = list(arm_link_keys)
        full_joint_ids = list(arm_joint_id)
        full_finger_keys = (
            list(finger_keys)
            if finger_keys is not None
            else [[]] * self._orig_num_arms
        )
        full_ee_to_gripper = (
            list(ee_to_gripper)
            if ee_to_gripper is not None
            else [None] * self._orig_num_arms
        )

        if self._has_head:
            full_link_keys.append(head_link_keys)
            full_joint_ids.append(head_joint_id or [])
            full_finger_keys.append([])
            full_ee_to_gripper.append(None)

        if self._has_body:
            full_link_keys.append(body_link_keys)
            full_joint_ids.append(body_joint_id or [])
            full_finger_keys.append([])
            full_ee_to_gripper.append(None)

        super().__init__(
            urdf=urdf,
            arm_link_keys=full_link_keys,
            arm_joint_id=full_joint_ids,
            finger_keys=full_finger_keys,
            ee_to_gripper=full_ee_to_gripper,
        )

    def get_joint_relative_pos(self):
        part_sizes = []
        for i in range(len(self.arm_link_keys)):
            n = len(self.arm_link_keys[i]) + (
                len(self.finger_keys[i]) > 0
                or self.ee_to_gripper[i] is not None
            )
            part_sizes.append(n)

        total_joints = sum(part_sizes)
        num_parts = len(part_sizes)
        matrix = torch.zeros((total_joints, total_joints), dtype=torch.float32)

        hub_idx = num_parts - 1 if self._has_body else 0

        base_dist = torch.full((num_parts, num_parts), 2.0)
        for i in range(num_parts):
            base_dist[i, i] = 0
            base_dist[i, hub_idx] = base_dist[hub_idx, i] = 1.0

        offsets = torch.cat(
            [torch.tensor([0]), torch.cumsum(torch.tensor(part_sizes), dim=0)]
        )

        for i in range(num_parts):
            idx_i = torch.arange(part_sizes[i])
            for j in range(num_parts):
                idx_j = torch.arange(part_sizes[j])
                if i == j:
                    val = torch.abs(idx_i[:, None] - idx_j)
                else:
                    val = idx_i[:, None] + base_dist[i, j] + idx_j
                matrix[
                    offsets[i] : offsets[i + 1], offsets[j] : offsets[j + 1]
                ] = val

        self._joint_relative_pos = matrix


class ZeroRobotState:
    """Sets selected robot-state joints or channels to zero."""

    def __init__(
        self,
        joint_indices=None,
        keys=("hist_robot_state",),
        state_indices=None,
    ):
        self.joint_indices = (
            [] if joint_indices is None else list(joint_indices)
        )
        self.keys = list(keys)
        self.state_indices = state_indices

    def __call__(self, data):
        if len(self.joint_indices) == 0:
            return data

        for key in self.keys:
            if key not in data:
                continue
            state = data[key]
            if not isinstance(state, (np.ndarray, torch.Tensor)):
                raise TypeError(
                    f"Unsupport zero for {key}'s type {type(state)}"
                )
            if self.state_indices is None:
                state[..., self.joint_indices, :] = 0
            else:
                state[..., self.joint_indices, self.state_indices] = 0
            data[key] = state
        return data
