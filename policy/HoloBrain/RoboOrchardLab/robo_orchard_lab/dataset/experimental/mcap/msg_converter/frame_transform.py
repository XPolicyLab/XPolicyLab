# Project RoboOrchard
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
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

import torch
from foxglove_schemas_protobuf.FrameTransform_pb2 import (
    FrameTransform as FgFrameTransform,
)
from foxglove_schemas_protobuf.FrameTransforms_pb2 import (
    FrameTransforms as FgFrameTransforms,
)
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion as FgQuaternion
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3 as FgVector3
from google.protobuf.timestamp import from_nanoseconds
from robo_orchard_core.utils.config import ClassType
from robo_orchard_core.utils.torch_utils import dtype_str2torch

from robo_orchard_lab.dataset.datatypes import (
    BatchFrameTransform,
    BatchFrameTransformGraph,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
    TensorTargetConfigMixin,
)

__all__ = [
    "BatchFrameTransform",
    "ToBatchFrameTransform",
    "ToBatchFrameTransformConfig",
    "FromBatchFrameTransform",
    "FromBatchFrameTransformConfig",
    "FromBatchFrameTransformGraph",
    "FromBatchFrameTransformGraphConfig",
]


class ToBatchFrameTransform(
    MessageConverterStateless[
        FgFrameTransform | list[FgFrameTransform],
        BatchFrameTransform,
    ]
):
    """Convert a Foxglove FrameTransform message to a FrameTransform Type."""

    def __init__(
        self,
        cfg: ToBatchFrameTransformConfig,
    ):
        self._cfg = cfg
        self._dtype = dtype_str2torch(cfg.dtype)

    def convert(
        self, src: FgFrameTransform | list[FgFrameTransform]
    ) -> BatchFrameTransform:
        if not isinstance(src, list):
            tf_trans = torch.tensor(
                [src.translation.x, src.translation.y, src.translation.z],
                dtype=self._dtype,
                device=self._cfg.device,
            )
            tf_rot = torch.tensor(
                [
                    src.rotation.w,
                    src.rotation.x,
                    src.rotation.y,
                    src.rotation.z,
                ],
                dtype=self._dtype,
                device=self._cfg.device,
            )

            return BatchFrameTransform(
                child_frame_id=src.child_frame_id,
                parent_frame_id=src.parent_frame_id,
                xyz=tf_trans.to(device=self._cfg.device),
                quat=tf_rot.to(device=self._cfg.device),
                timestamps=[src.timestamp.ToNanoseconds()],
            )
        else:
            assert len(src) > 0, "List of FrameTransform cannot be empty."
            tf_trans = torch.zeros(
                (len(src), 3),
                dtype=self._dtype,
            )
            tf_rot = torch.zeros(
                (len(src), 4),
                dtype=self._dtype,
            )
            for i, tf in enumerate(src):
                tf_trans[i, :] = torch.tensor(
                    [tf.translation.x, tf.translation.y, tf.translation.z],
                    dtype=self._dtype,
                )
                tf_rot[i, :] = torch.tensor(
                    [
                        tf.rotation.w,
                        tf.rotation.x,
                        tf.rotation.y,
                        tf.rotation.z,
                    ],
                    dtype=self._dtype,
                )
            return BatchFrameTransform(
                child_frame_id=src[0].child_frame_id,
                parent_frame_id=src[0].parent_frame_id,
                xyz=tf_trans.to(device=self._cfg.device),
                quat=tf_rot.to(device=self._cfg.device),
                timestamps=[tf.timestamp.ToNanoseconds() for tf in src],
            )


class ToBatchFrameTransformConfig(
    MessageConverterConfig[ToBatchFrameTransform],
    TensorTargetConfigMixin[ToBatchFrameTransform],
):
    class_type: type[ToBatchFrameTransform] = ToBatchFrameTransform


class FromBatchFrameTransform(
    MessageConverterStateless[BatchFrameTransform, list[FgFrameTransform]]
):
    """Convert from BatchFrameTransform to list of FrameTransform.

    The output is a list of `FrameTransform` messages, each
    representing the transform at a specific timestamp.

    """

    def __init__(
        self,
        cfg: FromBatchFrameTransformConfig,
    ):
        self._cfg = cfg

    def convert(self, src: BatchFrameTransform) -> list[FgFrameTransform]:
        frame_transform = src
        if frame_transform.timestamps is None:
            raise ValueError(
                "BatchFrameTransform must have timestamps for conversion."
            )

        ret: list[FgFrameTransform] = []
        batch_size = frame_transform.batch_size
        xyz = frame_transform.xyz.numpy(force=True)
        quat = frame_transform.quat.numpy(force=True)

        for i in range(batch_size):
            frame_tf = FgFrameTransform(
                timestamp=from_nanoseconds(frame_transform.timestamps[i]),
                parent_frame_id=frame_transform.parent_frame_id,
                child_frame_id=frame_transform.child_frame_id,
                translation=FgVector3(
                    x=xyz[i, 0],
                    y=xyz[i, 1],
                    z=xyz[i, 2],
                ),
                rotation=FgQuaternion(
                    w=quat[i, 0],
                    x=quat[i, 1],
                    y=quat[i, 2],
                    z=quat[i, 3],
                ),
            )
            ret.append(frame_tf)
        return ret


class FromBatchFrameTransformConfig(
    MessageConverterConfig[FromBatchFrameTransform],
):
    class_type: ClassType[FromBatchFrameTransform] = FromBatchFrameTransform


class FromBatchFrameTransformGraph(
    MessageConverterStateless[
        BatchFrameTransformGraph, list[FgFrameTransforms]
    ]
):
    """Convert from BatchFrameTransformGraph to list of FgFrameTransforms.

    The output is a list of `FgFrameTransforms` messages, each representing the
    transforms at a specific timestamp.
    """

    def __init__(self, cfg: FromBatchFrameTransformGraphConfig) -> None:
        self._cfg = cfg
        self._tf_converter = FromBatchFrameTransform(
            FromBatchFrameTransformConfig()
        )

    def convert(
        self, src: BatchFrameTransformGraph
    ) -> list[FgFrameTransforms]:
        ret: list[FgFrameTransforms] = []
        tf_state = src.as_state()
        tf_by_topics: list[list[FgFrameTransform]] = []
        for tf in tf_state.tf_list:
            tf_by_topics.append(self._tf_converter.convert(tf))
        # convert tf_by_topics to FgFrameTransforms
        # Since `tf_by_topics` stores list of FrameTransform per topic,
        # we need to transpose it to get list of FrameTransforms per timestamp.
        num_timestamps = len(tf_by_topics[0])
        for i in range(num_timestamps):
            fg_tf_msg = FgFrameTransforms(
                transforms=[topic_tfs[i] for topic_tfs in tf_by_topics]
            )
            ret.append(fg_tf_msg)
        return ret


class FromBatchFrameTransformGraphConfig(
    MessageConverterConfig[FromBatchFrameTransformGraph],
):
    class_type: ClassType[FromBatchFrameTransformGraph] = (
        FromBatchFrameTransformGraph
    )
