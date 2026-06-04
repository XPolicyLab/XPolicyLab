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

from robo_orchard_schemas.sensor_msgs.JointState_pb2 import (
    MultiJointStateStamped as PbMultiJointStateStamped,
)

from robo_orchard_lab.dataset.datatypes.joint_state import BatchJointsState
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base import (
    McapBatchEncoder,
    McapBatchEncoderConfig,
    StampedMessage,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    FromBatchJointsStateConfig,
)

__all__ = [
    "McapBatchFromBatchJointState",
    "McapBatchFromBatchJointStateConfig",
]


class McapBatchFromBatchJointState(McapBatchEncoder[BatchJointsState]):
    """Convert BatchJointsState to Mcap batch messages.

    This class converts a `BatchJointsState` object into a list of
    `MultiJointStateStamped` protobuf messages, each wrapped in
    a `StampedMessage` that includes logging and publication timestamps.

    The timestamps are required in the `BatchJointsState` object.

    """

    def __init__(self, config: McapBatchFromBatchJointStateConfig):
        super().__init__()
        self._cfg = config
        self._converter = FromBatchJointsStateConfig()()

    def format_batch(
        self, data: BatchJointsState
    ) -> dict[str, list[StampedMessage[PbMultiJointStateStamped]]]:
        if data.timestamps is None:
            raise ValueError("timestamps is required")
        converted_msgs = self._converter.convert(data)
        stamped_msgs = []
        for i, msg in enumerate(converted_msgs):
            stamped_msgs.append(
                StampedMessage(
                    data=msg,
                    log_time=data.timestamps[i],
                    pub_time=data.timestamps[i],
                )
            )

        return {self._cfg.target_topic: stamped_msgs}


class McapBatchFromBatchJointStateConfig(
    McapBatchEncoderConfig[McapBatchFromBatchJointState],
):
    class_type: type[McapBatchFromBatchJointState] = (
        McapBatchFromBatchJointState
    )

    target_topic: str
    """The target topic to publish the encoded batch messages."""
