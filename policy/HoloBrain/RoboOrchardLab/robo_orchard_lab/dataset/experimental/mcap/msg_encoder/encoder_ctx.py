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

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from foxglove import Channel, Context
from google.protobuf.message import Message as ProtobufMessage
from typing_extensions import Self

from robo_orchard_lab.dataset.experimental.mcap.foxglove import (
    create_channel,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import (
    McapMessageTuple,
    StampedMessage,
)

__all__ = ["McapEncoderContext", "FoxgloveEncoder"]


class McapEncoderContext(metaclass=ABCMeta):
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError(
            "reset() method is not implemented in the encoder mixin."
        )

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Exit the context manager."""
        self.reset()
        if exc_type is not None and exc_value is not None:
            raise exc_value

    def encode_message(
        self,
        topic: str,
        msg: Any,
        log_time: int,
        pub_time: int | None = None,
    ) -> McapMessageTuple:
        """Encode a message to MCAP format.

        Args:
            topic (str): The topic name of the message.
            msg (Any): The message data to encode.

        Returns:
            McapMessageTuple: The encoded message in MCAP format.
        """
        raise NotImplementedError(
            "encode_message() method is not implemented in the encoder mixin."
        )


class FoxgloveEncoder(McapEncoderContext):
    """Encoder for Protobuf messages in MCAP format.

    This encoder uses the Protobuf schema to encode messages into
    MCAP format. It requires the Protobuf schema to be provided during
    initialization.
    """

    def __init__(self, ctx: Context | None):
        self.reset(ctx=ctx)

    def reset(self, ctx: Context | None = None) -> None:
        if ctx is None:
            ctx = Context()
        self._ctx = ctx
        self._channel_dict: dict[str, Channel] = {}

    def encode_message(
        self,
        topic: str,
        msg: Any,
        log_time: int,
        pub_time: int | None = None,
    ) -> McapMessageTuple:
        # create channel if not exists

        if topic not in self._channel_dict:
            channel = create_channel(topic, type(msg), context=self._ctx)
            self._channel_dict[topic] = channel
        else:
            channel = self._channel_dict[topic]

        if channel.message_encoding == "protobuf":
            if not isinstance(msg, ProtobufMessage):
                raise ValueError(
                    f"Expected ProtobufMessage for topic '{topic}', "
                    f"but got {type(msg)}"
                )
            data = StampedMessage(
                data=msg.SerializeToString(),
                log_time=log_time,
                pub_time=pub_time,
            )
        else:
            data = StampedMessage(
                data=msg, log_time=log_time, pub_time=pub_time
            )

        return McapMessageTuple(
            schema=channel.schema(), channel=channel, message=data
        )
