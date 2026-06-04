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

from robo_orchard_lab.dataset.experimental.mcap.batch_decoder.base import (
    McapBatchDecoder,
    McapBatchDecoderConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (  # noqa: E501
    BatchFrameTransform,
    TensorTargetConfigMixin,
    ToBatchFrameTransformConfig,
)

__all__ = [
    "McapBatch2BatchFrameTransform",
    "McapBatch2BatchFrameTransformConfig",
]


class McapBatch2BatchFrameTransform(McapBatchDecoder[BatchFrameTransform]):
    def __init__(self, config: McapBatch2BatchFrameTransformConfig):
        super().__init__()
        self._cfg = config
        self._msg_cvt = ToBatchFrameTransformConfig(
            device=config.device, dtype=config.dtype
        )()
        self._required_topics = set([config.source_topic])

    def require_topics(self) -> set[str]:
        return self._required_topics

    def format_batch(
        self, decoded_msgs: dict[str, list]
    ) -> BatchFrameTransform:
        return self._msg_cvt.convert(decoded_msgs[self._cfg.source_topic])


class McapBatch2BatchFrameTransformConfig(
    McapBatchDecoderConfig[McapBatch2BatchFrameTransform],
    TensorTargetConfigMixin[McapBatch2BatchFrameTransform],
):
    class_type: type[McapBatch2BatchFrameTransform] = (
        McapBatch2BatchFrameTransform
    )

    source_topic: str
    """The source topic to use from batch messages."""
