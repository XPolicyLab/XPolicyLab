# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
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

import torch


def apply_scale_shift(
    robot_state: torch.Tensor,
    joint_scale_shift: torch.Tensor | None = None,
    inverse: bool = False,
    scale_only: bool = False,
):
    """Applies scale and shift normalization to joint angles in robot state.

    This function normalizes or denormalizes joint angles in the robot state
    using provided scale and shift. The normalization is applied only
    to the first channel (joint angle dimension) of the state tensor.
    The robot state tensor has shape [bs, num_step, num_joint, c], where
    c >= 1 and the first channel contains joint angle values.

    Args:
        robot_state (torch.Tensor): Robot state tensor with shape
            [batch_size, num_steps, num_joints, channels]. The joint angles are
            assumed to be in the first channel (index 0) of the last dimension.

        joint_scale_shift (torch.Tensor, optional): Scale and shift parameters
            tensor with shape [batch_size, num_joints, 2]. The last dimension
            contains [scale, shift] pairs for each joint. If None,
            no scaling/shifting is applied (identity operation).

        inverse (bool): If False, apply forward normalization: normalized =
            (original - shift) / scale. If True, apply inverse transformation
            (denormalization): original = normalized * scale + shift.

        scale_only (bool): If True, the shift will be set as 0.

    Returns:
        torch.Tensor: Normalized/denormalized robot state tensor with the same
            shape as input: [batch_size, num_steps, num_joints, c]. Only the
            first channel is modified; other channels remain unchanged.
    """

    if joint_scale_shift is None:
        return robot_state

    joint_scale_shift = joint_scale_shift.to(
        dtype=robot_state.dtype, device=robot_state.device
    )
    if robot_state.shape[0] != joint_scale_shift.shape[0]:
        num_parallel = robot_state.shape[0] // joint_scale_shift.shape[0]
        joint_scale_shift = joint_scale_shift.repeat_interleave(
            num_parallel, dim=0
        )
    scale = joint_scale_shift[:, None, :, 0:1]
    if not scale_only:
        shift = joint_scale_shift[:, None, :, 1:2]
    else:
        shift = 0
    if not inverse:
        robot_state = torch.cat(
            [(robot_state[..., :1] - shift) / scale, robot_state[..., 1:]],
            dim=-1,
        )
    else:
        robot_state = torch.cat(
            [robot_state[..., :1] * scale + shift, robot_state[..., 1:]],
            dim=-1,
        )
    return robot_state


def forward_kinematics(joint_state: torch.Tensor, inputs: dict):
    """Computes robot state from joint positions using forward kinematics.

    This function computes the robot state by applying forward kinematics to
    joint  positions. It transforms joint positions into joint poses using the
    provided kinematics and embodiedment_mat from the inputs.

    Note:
        The joint_state tensor can have shape [bs, num_steps, num_joints]
        or [bs, num_steps, num_joints, 1]. The output robot state has
        shape [batch_size, num_steps, num_joints, c], where the first channel
        contains joint positions and remaining channels contain joint 6D poses.

    Args:
        joint_state (torch.Tensor): Joint position tensor with shape
            [batch_size, num_steps, num_joints] or [batch_size, num_steps,
            num_joints, 1].

        inputs (dict): Dictionary containing required keys:
            - kinematics: Object with joint_state_to_robot_state function
            - embodiedment_mat(optional): Embodiment transformation matrix,
            from base coordinate to ego coordinate

    Returns:
        torch.Tensor: Robot state tensor with shape [batch_size, num_steps,
            num_joints, c]. Channel 0 contains joint positions, remaining
            channels contain joint 6D poses.
    """
    if joint_state.shape[-1] == 1:
        joint_state = joint_state.squeeze(-1)
    robot_state = []
    kinematics = inputs["kinematics"]
    embodiedment_mat = inputs.get("embodiedment_mat", [None] * len(kinematics))
    if len(kinematics) <= 1 or (
        all(x == kinematics[0] for x in kinematics[1:])
    ):
        num_steps = joint_state.shape[1]
        embodiedment_mat = embodiedment_mat[:, None].repeat(1, num_steps, 1, 1)
        num_parallel = joint_state.shape[0] // embodiedment_mat.shape[0]
        if num_parallel != 1:
            embodiedment_mat = embodiedment_mat.repeat_interleave(
                num_parallel, dim=0
            )
        robot_state = kinematics[0].joint_state_to_robot_state(
            joint_state, embodiedment_mat
        )
    else:
        for i in range(len(joint_state)):
            robot_state.append(
                inputs["kinematics"][i].joint_state_to_robot_state(
                    joint_state[i], embodiedment_mat[i]
                )
            )
        robot_state = torch.stack(robot_state)
    return robot_state


def recompute(robot_state: torch.Tensor, inputs: dict):
    """Recomputes robot state from normalized joint positions.

    This function recomputes the robot state by first denormalizing joint
    positions and then applying forward kinematics to compute joint poses
    from joint positions.

    Args:
        robot_state (torch.Tensor): Robot state tensor with shape
            [batch_size, num_steps, num_joints, c]. The first channel contains
            normalized joint positions.

        inputs (dict): Dictionary containing required keys:
            - joint_scale_shift: Scale and shift parameters for denormalization
            - kinematics: Object with joint_state_to_robot_state function
            - embodiedment_mat (optional): Embodiment transformation matrix,
            from base coordinate to ego coordinate

    Returns:
        torch.Tensor: Recomputed robot state tensor with shape [batch_size,
            num_steps, num_joints, c]. First channel contains joint positions,
            remaining channels contain joint 6D poses.
    """
    if "kinematics" not in inputs:
        return robot_state
    joint_state = apply_scale_shift(
        robot_state[..., :1],
        inputs.get("joint_scale_shift"),
        inverse=True,
    )
    robot_state = torch.cat(
        [
            robot_state[..., :1],
            forward_kinematics(joint_state, inputs)[..., 1:],
        ],
        dim=-1,
    )
    return robot_state


def apply_joint_mask(robot_state, joint_mask, constant_value=-1):
    """Applies a joint mask to set joint positions to a constant value.

    This function masks joint positions in the robot state by setting values
    at positions indicated by the joint mask to a constant. Only joints where
    joint_mask is True are modified.

    Args:
        robot_state (torch.Tensor): Robot state tensor with shape
            [batch_size, num_joint, num_steps, c]. Contains joint states across
            batches, joints, time steps, and channels. The first channel is
            normalized joint positions.

        joint_mask (torch.Tensor): Boolean mask tensor wiht shape
            [batch_size, num_joint].

        constant_value (int | float): Value to masked joint positions.
            Default is -1.

    Returns:
        torch.Tensor: Modified robot state with the same shape as input
            [batch_size, num_joint, num_steps, c]. Masked joints have positions
            set to constant_value.
    """
    masked_joint_state = torch.where(
        joint_mask[..., None, None],
        robot_state.new_tensor(constant_value),
        robot_state[..., :1],
    )
    robot_state = torch.cat([masked_joint_state, robot_state[..., 1:]], dim=-1)
    return robot_state
