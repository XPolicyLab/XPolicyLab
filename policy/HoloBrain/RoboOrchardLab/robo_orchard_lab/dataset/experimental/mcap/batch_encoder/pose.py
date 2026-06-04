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

from __future__ import annotations

from foxglove_schemas_protobuf.Pose_pb2 import Pose as FgPose
from foxglove_schemas_protobuf.PoseInFrame_pb2 import (
    PoseInFrame as FgPoseInFrame,
)
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion as FgQuaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3 as FgVector3
from google.protobuf.timestamp import from_nanoseconds

from robo_orchard_lab.dataset.datatypes import BatchPose
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base import (
    McapBatchEncoder,
    McapBatchEncoderConfig,
    StampedMessage,
)

__all__ = [
    "McapBatchFromBatchPose",
    "McapBatchFromBatchPoseConfig",
]


class McapBatchFromBatchPose(McapBatchEncoder[BatchPose]):
    def __init__(self, config: McapBatchFromBatchPoseConfig):
        super().__init__()
        self._cfg = config

    def format_batch(
        self, data: BatchPose
    ) -> dict[str, list[StampedMessage[FgPoseInFrame]]]:
        def to_pb_frame_transform_stamped(
            batch_pose: BatchPose,
        ) -> list[StampedMessage[FgPoseInFrame]]:
            if batch_pose.timestamps is None:
                raise ValueError(
                    "BatchPose must have timestamps for conversion."
                )
            if batch_pose.frame_id is None:
                raise ValueError(
                    "BatchPose must have a frame_id for conversion."
                )

            ret: list[StampedMessage[FgPoseInFrame]] = []
            batch_size = batch_pose.batch_size
            xyz = batch_pose.xyz.numpy(force=True)
            quat = batch_pose.quat.numpy(force=True)

            for i in range(batch_size):
                frame_tf = FgPoseInFrame(
                    timestamp=from_nanoseconds(batch_pose.timestamps[i]),
                    frame_id=batch_pose.frame_id,
                    pose=FgPose(
                        position=FgVector3(
                            x=xyz[i, 0],
                            y=xyz[i, 1],
                            z=xyz[i, 2],
                        ),
                        orientation=FgQuaternion(
                            w=quat[i, 0],
                            x=quat[i, 1],
                            y=quat[i, 2],
                            z=quat[i, 3],
                        ),
                    ),
                )

                # Create a stamped message with the timestamp
                stamped_msg = StampedMessage(
                    data=frame_tf,
                    log_time=batch_pose.timestamps[i],
                    pub_time=batch_pose.timestamps[i],
                )
                ret.append(stamped_msg)
            return ret

        return {self._cfg.target_topic: to_pb_frame_transform_stamped(data)}


class McapBatchFromBatchPoseConfig(
    McapBatchEncoderConfig[McapBatchFromBatchPose]
):
    class_type: type[McapBatchFromBatchPose] = McapBatchFromBatchPose

    target_topic: str
    """The target topic to publish the encoded batch messages."""
