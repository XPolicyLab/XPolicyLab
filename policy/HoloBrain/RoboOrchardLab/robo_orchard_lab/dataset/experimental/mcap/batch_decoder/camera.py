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

from robo_orchard_core.utils.config import ClassConfig, T

from robo_orchard_lab.dataset.experimental.mcap.batch_decoder.base import (
    McapBatchDecoder,
    McapBatchDecoderConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter.batch_camera import (  # noqa: E501
    BatchCameraData,
    BatchCameraDataEncoded,
    FgCameraCompressedImages,
    ToBatchCameraDataConfig,
    ToBatchCameraDataEncodedConfig,
)

__all__ = [
    "McapBatch2BatchCameraData",
    "McapBatch2BatchCameraDataEncoded",
    "McapBatch2BatchCameraDataConfig",
    "McapBatch2BatchCameraDataEncodedConfig",
]


def to_FgCameraCompressedImages(  # noqa: N802
    decoded_msgs: dict[str, list], cfg: CameraDataSourceMixin
) -> FgCameraCompressedImages:
    msg_batch = FgCameraCompressedImages(images=decoded_msgs[cfg.image_topic])
    if cfg.calib_topic is not None:
        msg_batch.calib = decoded_msgs.get(cfg.calib_topic, None)
    if cfg.tf_topic is not None:
        tf_msgs = decoded_msgs.get(cfg.tf_topic, [])
        if len(tf_msgs) > 0:
            msg_batch.tf = tf_msgs
        else:
            msg_batch.tf = None
    return msg_batch


class McapBatch2BatchCameraDataEncoded(
    McapBatchDecoder[BatchCameraDataEncoded]
):
    def __init__(self, config: McapBatch2BatchCameraDataEncodedConfig):
        super().__init__()
        self._cfg = config
        self._msg_cvt = ToBatchCameraDataEncodedConfig()()
        self._required_topics = config.required_topics

    def require_topics(self) -> set[str]:
        return self._required_topics

    def format_batch(
        self, decoded_msgs: dict[str, list]
    ) -> BatchCameraDataEncoded:
        # prepare input message
        return self._msg_cvt.convert(
            to_FgCameraCompressedImages(decoded_msgs, self._cfg)
        )


class McapBatch2BatchCameraData(McapBatchDecoder[BatchCameraData]):
    def __init__(self, config: McapBatch2BatchCameraDataConfig):
        super().__init__()
        self._cfg = config
        self._msg_cvt = ToBatchCameraDataConfig()()
        self._required_topics = self._required_topics = config.required_topics

    def require_topics(self) -> set[str]:
        return self._required_topics

    def format_batch(self, decoded_msgs: dict[str, list]) -> BatchCameraData:
        # prepare input message
        return self._msg_cvt.convert(
            to_FgCameraCompressedImages(decoded_msgs, self._cfg)
        )


class CameraDataSourceMixin(ClassConfig[T]):
    """Mixin for camera data source configuration."""

    image_topic: str
    """The source topic of camera image."""

    calib_topic: str | None = None
    """The source topic of camera calibration.

    If None, no calibration will be used."""

    tf_topic: str | None = None
    """Frame transform topic of camera.

    The frame transform is usually referred as camera extrinsics, which
    is the transformation from the camera frame to the world/parent frame.

    If None, no frame transform will be used.
    """

    @property
    def required_topics(self) -> set[str]:
        """Return the required topics for this camera data source."""
        topics = {self.image_topic}
        if self.calib_topic is not None:
            topics.add(self.calib_topic)
        if self.tf_topic is not None:
            topics.add(self.tf_topic)
        return topics


class McapBatch2BatchCameraDataEncodedConfig(
    McapBatchDecoderConfig[McapBatch2BatchCameraDataEncoded],
    CameraDataSourceMixin[McapBatch2BatchCameraDataEncoded],
):
    class_type: type[McapBatch2BatchCameraDataEncoded] = (
        McapBatch2BatchCameraDataEncoded
    )


class McapBatch2BatchCameraDataConfig(
    McapBatchDecoderConfig[McapBatch2BatchCameraData],
    CameraDataSourceMixin[McapBatch2BatchCameraData],
):
    class_type: type[McapBatch2BatchCameraData] = McapBatch2BatchCameraData
