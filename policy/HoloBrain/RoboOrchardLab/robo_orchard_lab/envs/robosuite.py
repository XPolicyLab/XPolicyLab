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
from robo_orchard_core.datatypes import (
    BatchCameraInfo,
    BatchFrameTransform,
    BatchJointsState,
)
from robo_orchard_core.utils.math import (
    Transform3D_M,
    matrix_to_quaternion,
    quaternion_standardize,
)
from robosuite.environments.robot_env import RobotEnv
from robosuite.utils.binding_utils import MjSim
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix as get_camera_pose,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

__all__ = [
    "get_camera_info",
    "get_depth_image",
    "get_joint_state",
    "get_tf_world",
]


def get_depth_image(env: RobotEnv, depth_img: np.ndarray) -> torch.Tensor:
    sim: MjSim = env.sim  # type: ignore
    return torch.from_numpy(
        get_real_depth_map(
            sim,
            depth_map=depth_img,
        )
    )


def get_camera_info(env: RobotEnv, camera_name: str) -> BatchCameraInfo:
    if camera_name not in env.camera_names:
        raise ValueError(
            f"Camera name {camera_name} not found in environment. "
            f"Available cameras: {env.camera_names}"
        )

    sim: MjSim = env.sim  # type: ignore
    cam_id = env.camera_names.index(camera_name)
    heights = env.camera_heights[cam_id]  # type: ignore
    widths = env.camera_widths[cam_id]  # type: ignore
    intrinsic_matrix_torch = torch.from_numpy(
        get_camera_intrinsic_matrix(
            sim,
            camera_name=camera_name,
            camera_height=heights,
            camera_width=widths,
        )
    ).unsqueeze(0)
    cam_pose = Transform3D_M(
        matrix=torch.from_numpy(
            get_camera_pose(
                sim,
                camera_name=camera_name,
            )
        ).unsqueeze(0),
    )

    return BatchCameraInfo(
        topic=camera_name,
        frame_id=camera_name,
        intrinsic_matrices=intrinsic_matrix_torch,
        pose=BatchFrameTransform(
            xyz=cam_pose.get_translation(),
            quat=cam_pose.get_rotation_quaternion(normalize=True),
            parent_frame_id="world",
            child_frame_id=camera_name,
        ),
        image_shape=(heights, widths),
    )


def get_joint_state(env: RobotEnv, joint_names: list[str]) -> BatchJointsState:
    """Get the joint states for the given joint names.

    Args:
        env (RobotEnv): The robosuite environment.
        joint_names (list[str]): The list of joint names.

    Returns:
        BatchJointsState: The joint states for the given joint names
            in the environment.
    """
    sim: MjSim = env.sim  # type: ignore
    pos = []
    vel = []
    for joint_name in joint_names:
        joint_pos = sim.data.get_joint_qpos(joint_name)
        joint_vel = sim.data.get_joint_qvel(joint_name)
        pos.append(joint_pos)
        vel.append(joint_vel)

    assert env.cur_time is not None
    timestamp = int(env.cur_time * 1e9)
    return BatchJointsState(
        position=torch.tensor(pos, dtype=torch.double).unsqueeze(0),
        velocity=torch.tensor(vel, dtype=torch.double).unsqueeze(0),
        names=joint_names,
        timestamps=[timestamp],
    )


def get_tf_world(
    env: RobotEnv, frame_names: list[str]
) -> dict[str, BatchFrameTransform]:
    """Get the frame transforms for the given frame names.

    Args:
        env (RobotEnv): The robosuite environment.
        frame_names (list[str]): The list of frame names.

    Returns:
        dict[str, BatchFrameTransform]: The frame transforms for the
            given frame names in the environment.
    """
    sim: MjSim = env.sim  # type: ignore
    ret = {}
    body_names = set(sim.model.body_names)
    site_names = set(sim.model.site_names)
    # Note that robosuite convert quaternion to (x,y,z,w) format
    # from mujoco, but here we directly use the quaternion from mujoco,
    # which is the same as robo_orchard.

    assert env.cur_time is not None
    timestamp = int(env.cur_time * 1e9)

    for frame_name in frame_names:
        if frame_name in body_names:
            pos = torch.from_numpy(
                sim.data.get_body_xpos(frame_name).copy()
            ).unsqueeze(0)
            quat = quaternion_standardize(
                torch.from_numpy(
                    sim.data.get_body_xquat(frame_name).copy()
                ).unsqueeze(0)
            )
        elif frame_name in site_names:
            pos = torch.from_numpy(
                sim.data.get_site_xpos(frame_name).copy()
            ).unsqueeze(0)
            quat = matrix_to_quaternion(
                torch.from_numpy(
                    sim.data.get_site_xmat(frame_name).reshape(1, 3, 3).copy()
                ),
                normalize_output=True,
            )
        else:
            raise ValueError(
                f"Frame name {frame_name} not found in environment. "
                f"Available bodies: {body_names}, "
                f"Available sites: {site_names}"
            )
        ret[frame_name] = BatchFrameTransform(
            xyz=pos,
            quat=quat,
            parent_frame_id="world",
            child_frame_id=frame_name,
            timestamps=[timestamp],
        )
    return ret
