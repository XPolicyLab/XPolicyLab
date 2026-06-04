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
from typing import Any

import numpy as np
from foxglove_schemas_protobuf.CameraCalibration_pb2 import (
    CameraCalibration as FgCameraCalibration,
)
from foxglove_schemas_protobuf.CompressedImage_pb2 import (
    CompressedImage as FgCompressedImage,
)
from google.protobuf.timestamp import from_nanoseconds
from robo_orchard_core.utils.config import Config

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchCameraInfo,
    BatchFrameTransform,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.base import (
    McapBatchEncoder,
    McapBatchEncoderConfig,
    StampedMessage,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder.tf import (
    McapBatchFromBatchFrameTransformConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    FromBatchImageDataConfig,
)

__all__ = [
    "McapBatchFromBatchCameraDataEncoded",
    "McapBatchFromBatchCameraDataEncodedConfig",
    "McapBatchFromBatchCameraData",
    "McapBatchFromBatchCameraDataConfig",
]


def format_batch_camera_info(
    data: BatchCameraInfo,
    timestamps: list[int],
    calib_topic: str | None,
    tf_topic: str | None,
    default_focal: float = 500.0,
) -> dict[str, list[StampedMessage[Any]]]:
    """Format BatchCameraInfo into a dictionary of StampedMessages."""
    ret: dict[str, list[StampedMessage[Any]]] = {}
    if data.frame_id is None:
        raise ValueError(
            "BatchCameraInfo must have a frame_id for conversion."
        )
    if data.image_shape is None:
        raise ValueError(
            "BatchCameraInfo must have image_shape for conversion."
        )

    if data.intrinsic_matrices is not None and calib_topic is not None:
        # convert to CameraCalibration message
        distortion_model = (
            data.distortion_model
            if data.distortion_model is not None
            else "plumb_bob"
        )
        distortion_coef = (
            data.distorsion_coefficients.numpy(force=True)
            if data.distorsion_coefficients is not None
            else np.zeros((5,))
        )
        assert data.intrinsic_matrices.shape[0] == len(timestamps), (
            "The batch size of intrinsic matrices must match the "
            "batch size of timestamps. "
        )
        # we use the intrinsic matrix and transform2d to construct the
        # K matrix for visualization.
        vis_intrinsic_mat = data.get_intrinsic_with_transform()
        k_batch = (
            vis_intrinsic_mat.reshape(-1, 9).numpy(force=True)
            if vis_intrinsic_mat is not None
            else None
        )
        if k_batch is None:
            k_batch = (
                np.array(
                    [
                        [default_focal, 0, data.image_shape[1] / 2],
                        [0, default_focal, data.image_shape[0] / 2],
                        [0, 0, 1],
                    ]
                )
                .reshape(-1, 9)
                .repeat(len(timestamps), axis=0)
            )
        p_batch = np.zeros(shape=(len(timestamps), 3, 4))
        p_batch[:, :3, :3] = k_batch[:, :9].reshape(-1, 3, 3)
        p_batch = p_batch.reshape(-1, 12)

        ret[calib_topic] = []
        for i in range(len(timestamps)):
            calibration = FgCameraCalibration(
                timestamp=from_nanoseconds(timestamps[i]),
                frame_id=data.frame_id,
                width=data.image_shape[1],
                height=data.image_shape[0],
                K=k_batch[i],
                P=p_batch[i],
                distortion_model=distortion_model,
                D=distortion_coef,
            )

            stamped_msg = StampedMessage(
                data=calibration,
                log_time=timestamps[i],
                pub_time=timestamps[i],
            )
            ret[calib_topic].append(stamped_msg)

    if data.pose is not None and tf_topic is not None:
        encoder = McapBatchFromBatchFrameTransformConfig(
            target_topic=tf_topic,
        )()
        tf = BatchFrameTransform(**data.pose.__dict__)
        tf.timestamps = timestamps
        pose_msgs = encoder.format_batch(tf)
        ret.update(pose_msgs)

    return ret


class McapBatchFromBatchCameraDataEncoded(
    McapBatchEncoder[BatchCameraDataEncoded]
):
    def __init__(self, config: McapBatchFromBatchCameraDataEncodedConfig):
        super().__init__()
        self._cfg = config

    def format_batch(
        self, data: BatchCameraDataEncoded
    ) -> dict[str, list[StampedMessage[Any]]]:
        ret: dict[str, list[StampedMessage[Any]]] = {}
        if data.timestamps is None:
            raise ValueError("BatchCameraDataEncoded must have timestamps.")
        if data.frame_id is None:
            raise ValueError("BatchCameraDataEncoded must have a frame_id.")
        ret.update(
            format_batch_camera_info(
                data=data,
                timestamps=data.timestamps,
                calib_topic=self._cfg.calib_topic,
                tf_topic=self._cfg.tf_topic,
            )
        )
        img_list: list[StampedMessage[FgCompressedImage]] = []
        for i in range(data.batch_size):
            img = FgCompressedImage(
                timestamp=from_nanoseconds(data.timestamps[i]),
                frame_id=data.frame_id,
                data=data.sensor_data[i],
                format=data.format,
            )
            stamped_msg = StampedMessage(
                data=img,
                log_time=data.timestamps[i],
                pub_time=data.timestamps[i],
            )
            img_list.append(stamped_msg)
        ret[self._cfg.image_topic] = img_list
        return ret


class McapBatchFromBatchCameraData(McapBatchEncoder[BatchCameraData]):
    _cfg: McapBatchFromBatchCameraDataConfig

    def __init__(self, config: McapBatchFromBatchCameraDataConfig):
        self._cfg = config
        self._img_data_converter = FromBatchImageDataConfig()()

    def format_batch(
        self,
        data: BatchCameraData,
    ) -> dict[str, list[StampedMessage[Any]]]:
        ret: dict[str, list[StampedMessage[Any]]] = {}
        if data.timestamps is None:
            raise ValueError("BatchCameraDataEncoded must have timestamps.")
        if data.frame_id is None:
            raise ValueError("BatchCameraDataEncoded must have a frame_id.")
        ret.update(
            format_batch_camera_info(
                data=data,
                timestamps=data.timestamps,
                calib_topic=self._cfg.calib_topic,
                tf_topic=self._cfg.tf_topic,
            )
        )
        img_list = self._img_data_converter.convert(data)
        # assign frame_id as ImageData does not have frame_id field.
        for img_msg in img_list:
            img_msg.frame_id = data.frame_id
        ret[self._cfg.image_topic] = [
            StampedMessage(
                data=img_msg,
                log_time=img_msg.timestamp.ToNanoseconds(),
                pub_time=img_msg.timestamp.ToNanoseconds(),
            )
            for img_msg in img_list
        ]
        return ret


class McapBatchFromCameraInfoMixin(Config):
    calib_topic: str | None = None
    """Topic for camera calibration messages."""
    tf_topic: str | None = None
    """Topic for camera frame transform messages."""


class McapBatchFromBatchCameraDataEncodedConfig(
    McapBatchEncoderConfig[McapBatchFromBatchCameraDataEncoded],
    McapBatchFromCameraInfoMixin,
):
    """Configuration for McapBatchFromBatchCameraInfo."""

    class_type: type[McapBatchFromBatchCameraDataEncoded] = (
        McapBatchFromBatchCameraDataEncoded
    )
    image_topic: str
    """Topic for camera image messages."""


class McapBatchFromBatchCameraDataConfig(
    McapBatchEncoderConfig[McapBatchFromBatchCameraData],
    McapBatchFromCameraInfoMixin,
):
    """Configuration for McapBatchFromBatchCameraData."""

    class_type: type[McapBatchFromBatchCameraData] = (
        McapBatchFromBatchCameraData
    )
    image_topic: str
    """Topic for camera image messages."""
