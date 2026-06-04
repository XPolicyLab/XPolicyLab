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
from sapien import Pose

from robo_orchard_lab.dataset.datatypes import BatchTransform3D


def sapien_pose_to_orchard(
    pose: Pose, timestamp: int | None = None
) -> BatchTransform3D:
    """Convert a SAPIEN Pose to a BatchTransform3D.

    Args:
        pose (Pose): The SAPIEN Pose to convert.
        timestamp (int | None): The timestamp of the pose. Should be
            int nanoseconds(1e-9s). If None, no timestamp will be set.

    Returns:
        BatchTransform3D: The converted BatchTransform3D.
    """
    t = pose.get_p()
    q = pose.get_q()

    return BatchTransform3D(
        xyz=torch.from_numpy(t).unsqueeze(0),
        quat=torch.from_numpy(q).unsqueeze(
            0
        ),  # sapien uses (w, x, y, z) format as well
        timestamps=[timestamp] if timestamp is not None else None,
    )
