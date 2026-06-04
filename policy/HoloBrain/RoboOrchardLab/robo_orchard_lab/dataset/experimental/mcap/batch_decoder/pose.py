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

from robo_orchard_lab.dataset.experimental.mcap.batch_decoder.base import (
    McapBatchDecoder,
    McapBatchDecoderConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (  # noqa: E501
    BatchPose,
    TensorTargetConfigMixin,
    ToBatchPoseConfig,
)

__all__ = [
    "McapBatch2BatchPose",
    "McapBatch2BatchPoseConfig",
]


class McapBatch2BatchPose(McapBatchDecoder[BatchPose]):
    def __init__(self, config: McapBatch2BatchPoseConfig):
        super().__init__()
        self._cfg = config
        self._msg_cvt = ToBatchPoseConfig(
            device=config.device, dtype=config.dtype
        )()
        self._required_topics = set([config.source_topic])

    def require_topics(self) -> set[str]:
        return self._required_topics

    def format_batch(self, decoded_msgs: dict[str, list]) -> BatchPose:
        return self._msg_cvt.convert(decoded_msgs[self._cfg.source_topic])


class McapBatch2BatchPoseConfig(
    McapBatchDecoderConfig[McapBatch2BatchPose],
    TensorTargetConfigMixin[McapBatch2BatchPose],
):
    class_type: type[McapBatch2BatchPose] = McapBatch2BatchPose

    source_topic: str
    """The source topic to use from batch messages."""
