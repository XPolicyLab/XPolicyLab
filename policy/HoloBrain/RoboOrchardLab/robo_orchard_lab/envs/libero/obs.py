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

import torch
from libero.libero.envs import OffScreenRenderEnv
from robo_orchard_core.datatypes import (
    BatchCameraData,
    BatchFrameTransform,
    BatchJointsState,
    ImageMode,
)
from robo_orchard_core.kinematics.chain import KinematicChain
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.robots.robot import Robot
from robosuite.robots.single_arm import SingleArm

from robo_orchard_lab.envs.robosuite import (
    get_camera_info,
    get_depth_image,
    get_joint_state,
    get_tf_world,
)

__all__ = ["get_joints", "get_camera_data", "get_robot_tf"]


def get_joints(libero_env: OffScreenRenderEnv) -> BatchJointsState:
    """Get the joints state of the robot in the Libero environment.

    Note:
        This method get the joint data from simulator directly instead of
        from the observation dict. This may be didfferent from the observation
        if there is action delay or noise.

    """
    # first check that the env only contain one robot
    assert len(libero_env.robots) == 1, (
        "get_joints currently only supports single-robot environments."
        f"Got {len(libero_env.robots)} robots."
    )
    robot: Robot = libero_env.robots[0]  # type: ignore
    joint_names: list[str] = [j for j in robot.robot_joints]  # type: ignore

    # add gripper joints if applicable
    if isinstance(robot, SingleArm) and robot.gripper_joints is not None:
        joint_names.extend(robot.gripper_joints)
    ret = get_joint_state(libero_env.env, joint_names)
    return ret


def get_camera_data(
    libero_obs: dict,
    libero_env: OffScreenRenderEnv,
) -> dict[str, BatchCameraData]:
    """Convert camera data from observation dict to BatchCameraData.

    This function assumes the input observation dict has the
    following structure (RoboTwin):

    .. code-block:: text

        {
            "agentview_image": (H, W, 3) np.ndarray,
            "agentview_depth": (H, W, 1) np.ndarray,
        }

    """
    env: SingleArmEnv = libero_env.env

    assert env.cur_time is not None

    timestamp = int(env.cur_time * 1e9)

    camera_names = env.camera_names
    ret = {}
    for cam_name in camera_names:
        image_key = f"{cam_name}_image"
        cam_info = get_camera_info(env, cam_name)
        ret[image_key] = BatchCameraData(
            sensor_data=torch.from_numpy(libero_obs[image_key])
            .unsqueeze(0)
            .clone(),
            pix_fmt=ImageMode.RGB,
            timestamps=[timestamp],
            **(cam_info.__dict__),
        )
        depth_key = f"{cam_name}_depth"
        if depth_key in libero_obs:
            ret[depth_key] = BatchCameraData(
                sensor_data=get_depth_image(
                    env, libero_obs[depth_key]
                ).unsqueeze(0),
                pix_fmt=ImageMode.F,
                **(cam_info.__dict__),
                timestamps=[timestamp],
            )

    return ret


def get_robot_tf(
    robot_kin_chain: KinematicChain, libero_env: OffScreenRenderEnv
) -> dict[str, BatchFrameTransform]:
    """Get the robot frame transforms for the given robot kinematic chain.

    Note:
        This method get poses from simulator directly instead of
        from the observation dict. This may be didfferent from the observation
        if there is action delay or noise.

    Args:
        robot_kin_chain (KinematicChain): The robot kinematic chain.
        libero_env (OffScreenRenderEnv): The Libero environment.

    Returns:
        list[str]: The frame transforms for the given robot kinematic chain
            in the environment.
    """
    frame_names = robot_kin_chain.frame_names
    ret = get_tf_world(libero_env.env, frame_names)
    ret.pop("world", None)  # remove world frame
    return ret
