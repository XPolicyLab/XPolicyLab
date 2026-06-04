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


from typing import Any, BinaryIO, Optional

from foxglove import (
    Channel,
    Context,
    open_mcap,
)
from foxglove.mcap import MCAPWriteOptions

from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.msg_encoder import (
    FoxgloveEncoder,
)

__all__ = [
    "FoxgloveMcapWriter",
]


class FoxgloveMcapWriter:
    def __init__(
        self,
        path: BinaryIO,
        writer_options: MCAPWriteOptions | None = None,
    ):
        self._ctx = Context()
        self._encoder = FoxgloveEncoder(self._ctx)
        self._mcap = open_mcap(
            path,
            context=self._ctx,
            writer_options=writer_options,
        )

    def write_message(
        self,
        topic: str,
        message: Any,
        log_time: int,
        publish_time: Optional[int] = None,
    ):
        msg = self._encoder.encode_message(
            topic, message, log_time, publish_time
        )
        channel = msg.channel
        assert isinstance(channel, Channel)
        assert isinstance(msg.message, StampedMessage)
        channel.log(
            msg=msg.message.data,
            log_time=msg.message.log_time,
        )

    def __enter__(self):
        self._mcap.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._mcap.__exit__(exc_type, exc_value, traceback)

    def close(self):
        self._mcap.close()
