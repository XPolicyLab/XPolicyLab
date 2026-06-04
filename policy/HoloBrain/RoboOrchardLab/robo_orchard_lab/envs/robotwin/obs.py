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
from robo_orchard_core.datatypes import (
    BatchCameraData,
    BatchFrameTransform,
    BatchJointsState,
    ImageMode,
)
from robo_orchard_core.utils.math import (
    Transform3D_M,
    check_valid_rotation_matrix,
)


def get_joints(
    obs: dict,
    joint_names: list[str] | None = None,
    timestamps: list[int] | None = None,
) -> BatchJointsState:
    """Convert joint data from observation dict to BatchJointsState.

    This function assumes the input observation dict has the
    following structure (RoboTwin):

    .. code-block:: text

        {
            "joint_action": {
                "vector": (N,) np.ndarray,
                ...
            },
            ... other keys ...
        }

    """
    arr = torch.from_numpy(obs["joint_action"]["vector"]).unsqueeze(0)
    return BatchJointsState(
        position=arr, names=joint_names, timestamps=timestamps
    )


def get_observation_cams(obs: dict) -> dict:
    """Convert all camera data in the observation dict.

    This function assumes the input observation dict has
    the following structure (RoboTwin):

    .. code-block:: text

        {
            "observation": {
                "camera_name_1": { ... camera data ... },
                "camera_name_2": { ... camera data ... },
                ...
            },
            ... other keys ...
        }

    Each camera data dict follows the same format as described
    in `get_camera_data`.

    Example output structure::

        {
            "camera_name_1": {
                "rgb": BatchCameraData,
                "depth": BatchCameraData,
            },
            "camera_name_2": {
                "rgb": BatchCameraData,
            },
            ...
        }

    Returns:
        dict: Mapping from camera name to converted BatchCameraData entries.

    """
    ret = {}
    for k, v in obs["observation"].items():
        ret[k] = get_camera_data(v, camera_name=k)
    return ret


def get_camera_data(
    cam: dict, camera_name: str, timestamps: list[int] | None = None
) -> dict[str, BatchCameraData]:
    """Convert camera data from dict to BatchCameraData.

        This function assumes the input camera dict has the following keys:

        - ``rgb``: ``(H, W, 3)`` np.ndarray, optional.
        - ``depth``: ``(H, W)`` np.ndarray, optional.
        - ``intrinsic_cv``: ``(3, 3)`` np.ndarray.
        - ``extrinsic_cv``: ``(4, 4)`` np.ndarray camera extrinsic matrix.
            Uses the OpenCV convention. It encodes the camera pose in the
            world frame after inversion.

    Args:
        cam (dict): Camera data in dict format.
        camera_name (str): Name of the camera.

    Returns:
        dict[str, BatchCameraData]: Converted camera data keyed by stream type.

    """
    # `extrinsic_cv` follows the external OpenCV camera extrinsic
    # convention. Invert it here to build the camera pose in world.
    world_to_camera_mat = torch.eye(4).reshape(1, 4, 4)
    world_to_camera_mat[0, :3, :] = torch.from_numpy(cam["extrinsic_cv"])
    camera_in_world = Transform3D_M(matrix=world_to_camera_mat).inverse()
    assert check_valid_rotation_matrix(
        camera_in_world.get_matrix()[:, :3, :3], tol=1e-5
    ), "Invalid camera_in_world rotation matrix"
    cam_tf = BatchFrameTransform(
        xyz=camera_in_world.get_translation(),
        quat=camera_in_world.get_rotation_quaternion(),
        parent_frame_id="world",
        child_frame_id=camera_name,
    )
    intrinsic = torch.from_numpy(cam["intrinsic_cv"]).unsqueeze(0)

    ret = {}

    if "rgb" in cam:
        ret["rgb"] = BatchCameraData(
            sensor_data=torch.from_numpy(cam["rgb"]).unsqueeze(0),
            pix_fmt=ImageMode.RGB,
            frame_id=camera_name,
            intrinsic_matrices=intrinsic,
            pose=cam_tf,
            timestamps=timestamps,
        )

    if "depth" in cam:
        ret["depth"] = BatchCameraData(
            sensor_data=torch.from_numpy(cam["depth"])
            .unsqueeze(0)
            .to(dtype=torch.float32),
            pix_fmt=ImageMode.F,
            frame_id=camera_name,
            intrinsic_matrices=intrinsic,
            pose=cam_tf,
            timestamps=timestamps,
        )
    return ret
