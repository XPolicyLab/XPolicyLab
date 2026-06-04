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

from foxglove_schemas_protobuf.FrameTransform_pb2 import (
    FrameTransform as FgFrameTransform,
)
from foxglove_schemas_protobuf.FrameTransforms_pb2 import (
    FrameTransforms as FgFrameTransforms,
)

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base import (
    McapBatchEncoder,
    McapBatchEncoderConfig,
    StampedMessage,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    FromBatchFrameTransformConfig,
    FromBatchFrameTransformGraphConfig,
)

__all__ = [
    "McapBatchFromBatchFrameTransform",
    "McapBatchFromBatchFrameTransformConfig",
    "McapBatchFromBatchFrameTransformGraph",
    "McapBatchFromBatchFrameTransformGraphConfig",
]


class McapBatchFromBatchFrameTransform(McapBatchEncoder[BatchFrameTransform]):
    def __init__(self, config: McapBatchFromBatchFrameTransformConfig):
        super().__init__()
        self._cfg = config
        self._converter = FromBatchFrameTransformConfig()()

    def format_batch(
        self, data: BatchFrameTransform
    ) -> dict[str, list[StampedMessage[FgFrameTransform]]]:
        frame_transform = data
        if frame_transform.timestamps is None:
            raise ValueError(
                "BatchFrameTransform must have timestamps for conversion."
            )
        ret: list[StampedMessage[FgFrameTransform]] = []
        frame_tf = self._converter.convert(frame_transform)
        for i, tf in enumerate(frame_tf):
            stamped_msg = StampedMessage(
                data=tf,
                log_time=frame_transform.timestamps[i],
                pub_time=frame_transform.timestamps[i],
            )
            ret.append(stamped_msg)

        return {self._cfg.target_topic: ret}


class McapBatchFromBatchFrameTransformConfig(
    McapBatchEncoderConfig[McapBatchFromBatchFrameTransform]
):
    """Configuration for converting BatchFrameTransform to Mcap batch messages."""  # noqa: E501

    class_type: type[McapBatchFromBatchFrameTransform] = (
        McapBatchFromBatchFrameTransform
    )

    target_topic: str
    """The target topic to publish the encoded batch messages."""


class McapBatchFromBatchFrameTransformGraph(
    McapBatchEncoder[BatchFrameTransformGraph]
):
    def __init__(self, config: McapBatchFromBatchFrameTransformGraphConfig):
        super().__init__()
        self._cfg = config
        self._converter = FromBatchFrameTransformGraphConfig()()

    def format_batch(
        self, data: BatchFrameTransformGraph
    ) -> dict[str, list[StampedMessage[FgFrameTransforms]]]:
        frame_transform_graph = data
        # Check timestamps
        # first get the first topic's timestamps
        # the timestamps should not be None
        st = data.as_state()
        timestamps = st.tf_list[0].timestamps
        if timestamps is None:
            raise ValueError(
                "BatchFrameTransformGraph must have timestamps for conversion."
            )

        ret: list[StampedMessage[FgFrameTransforms]] = []
        frame_tf_graph = self._converter.convert(frame_transform_graph)
        assert len(frame_tf_graph) == len(timestamps), (
            "The number of FrameTransforms in the converted graph must match "
            "the number of timestamps."
        )
        for i, tf in enumerate(frame_tf_graph):
            stamped_msg = StampedMessage(
                data=tf,
                log_time=timestamps[i],
                pub_time=timestamps[i],
            )
            ret.append(stamped_msg)

        return {self._cfg.target_topic: ret}


class McapBatchFromBatchFrameTransformGraphConfig(
    McapBatchEncoderConfig[McapBatchFromBatchFrameTransformGraph]
):
    """Configuration for converting BatchFrameTransformGraph to Mcap batch messages."""  # noqa: E501

    class_type: type[McapBatchFromBatchFrameTransformGraph] = (
        McapBatchFromBatchFrameTransformGraph
    )

    target_topic: str
    """The target topic to publish the encoded batch messages."""
