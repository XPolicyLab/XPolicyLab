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
from bisect import bisect_left
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    Literal,
    TypeGuard,
    TypeVar,
)

from foxglove import (
    Channel as FoxgloveChannel,
    Schema as FoxgloveSchema,
)

from mcap.records import (
    Channel as McapChannel,
    Message as McapMessage,
    Schema as McapSchema,
)

if TYPE_CHECKING:
    from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
        McapDecoderContext,
    )


__all__ = [
    "StampedMessage",
    "McapMessageTuple",
    "McapDecodedMessageTuple",
    "McapMessagesTuple",
]


T = TypeVar("T", bound=Any)


@dataclass
class StampedMessage(Generic[T]):
    """Any message with timestamp."""

    data: T
    """The actual message data."""
    log_time: int
    """The log time in nanoseconds."""
    pub_time: int | None = None
    """The publish time in nanoseconds."""


@dataclass
class McapMessageTuple:
    schema: McapSchema | None | FoxgloveSchema
    channel: McapChannel | FoxgloveChannel
    message: McapMessage | StampedMessage[Any]

    @property
    def topic(self) -> str:
        if isinstance(self.channel, McapChannel):
            return self.channel.topic
        elif isinstance(self.channel, FoxgloveChannel):
            return self.channel.topic()
        else:
            raise TypeError(
                f"Unsupported channel type: {type(self.channel)}. "
                "Expected McapChannel or FoxgloveChannel."
            )

    def decode(self, decoder_ctx: McapDecoderContext) -> Any:
        """Decode the message using the provided decoder context."""
        if isinstance(self.message, StampedMessage):
            raise NotImplementedError(
                "Decoding of StampedMessage is not implemented yet. "
            )
        return decoder_ctx.decode_message(
            message_encoding=self.channel.message_encoding,
            message=self.message,
            schema=self.schema,
        )


@dataclass
class McapDecodedMessageTuple(McapMessageTuple):
    decoded_message: Any

    def decode(self, decoder_ctx: McapDecoderContext) -> Any:
        """Return the already decoded message.

        If not decoded, decode it.
        """
        if self.decoded_message is None:
            self.decoded_message = decoder_ctx.decode_message(
                message_encoding=self.channel.message_encoding,
                message=self.message,
                schema=self.schema,
            )
        return self.decoded_message


@dataclass
class McapMessagesTuple:
    schema: McapSchema | None | FoxgloveSchema
    channel: McapChannel | FoxgloveChannel
    # messages: list[McapMessage] | list[StampedMessage[Any]]
    messages: list[McapMessage | StampedMessage[Any]]

    @property
    def topic(self) -> str:
        if isinstance(self.channel, McapChannel):
            return self.channel.topic
        elif isinstance(self.channel, FoxgloveChannel):
            return self.channel.topic()
        else:
            raise TypeError(
                f"Unsupported channel type: {type(self.channel)}. "
                "Expected McapChannel or FoxgloveChannel."
            )

    @property
    def min_log_time(self) -> int:
        """Return the minimum log time of all messages."""
        if not self.messages:
            raise ValueError("No messages in the tuple.")
        return min(
            msg.log_time for msg in self.messages if msg.log_time is not None
        )

    @property
    def max_log_time(self) -> int:
        """Return the maximum log time of all messages."""
        if not self.messages:
            raise ValueError("No messages in the tuple.")
        return max(
            msg.log_time for msg in self.messages if msg.log_time is not None
        )

    def _is_messages_stamped_message(
        self, data: list
    ) -> TypeGuard[list[StampedMessage[Any]]]:
        """Check if the messages are all StampedMessage."""
        return all(isinstance(msg, StampedMessage) for msg in data)

    def _is_messages_mcap_message(
        self, data: list
    ) -> TypeGuard[list[McapMessage]]:
        """Check if the messages are all McapMessage."""
        return all(isinstance(msg, McapMessage) for msg in data)

    def split(
        self,
        log_time: int,
        is_sorted_asc: bool = False,
    ) -> tuple[McapMessagesTuple | None, McapMessagesTuple | None]:
        """Split the messages tuple into two parts based on the log time.

        This method returns two new `McapMessagesTuple` instances:
        - The first contains messages with `log_time < log_time`.
        - The second contains messages with `log_time >= log_time`.

        Args:
            log_time (int): The log time to split the messages.
            is_sorted_asc (bool, optional): Whether the messages are already
                sorted in ascending order by log time. If True, a binary search
                will be used to find the split point. Defaults to False.

        """
        messages = self.messages
        left = []
        right = []
        if not is_sorted_asc:
            for msg in messages:
                if msg.log_time < log_time:
                    left.append(msg)
                else:
                    right.append(msg)
        else:
            # If the messages are already sorted ascending by log_time,
            # we can use binary search to find the split point.
            idx = bisect_left(messages, log_time, key=lambda msg: msg.log_time)
            left = messages[:idx]
            right = messages[idx:]

        ret_left = (
            McapMessagesTuple(
                schema=self.schema, channel=self.channel, messages=left
            )
            if len(left) > 0
            else None
        )
        ret_right = (
            McapMessagesTuple(
                schema=self.schema, channel=self.channel, messages=right
            )
            if len(right) > 0
            else None
        )

        return (ret_left, ret_right)

    def sort(
        self,
        key: Literal["log_time", "publish_time"] = "log_time",
        reverse: bool = False,
    ) -> None:
        """Sort the messages by log time."""

        if key == "log_time":
            # Sort by log_time, which is an attribute of McapMessage
            self.messages.sort(key=lambda msg: msg.log_time, reverse=reverse)
        elif key == "publish_time":
            if not self._is_messages_mcap_message(self.messages):
                raise TypeError("All messages must be McapMessage instances.")
            # Sort by pub_time, which is an attribute of McapMessage
            self.messages.sort(
                key=lambda msg: msg.publish_time, reverse=reverse
            )
        else:
            raise ValueError(
                f"Invalid sort key: {key}. Use 'log_time' or 'pub_time'."
            )

    def __iter__(self) -> Iterator[McapMessage | StampedMessage[Any]]:
        """Return an iterator over the messages in the tuple."""
        return iter(self.messages)

    def __getitem__(self, index: int) -> McapMessage | StampedMessage[Any]:
        return self.messages[index]

    def __len__(self) -> int:
        """Return the number of messages in the tuple."""
        return len(self.messages)

    def append(self, msg: McapMessage | McapMessageTuple) -> None:
        """Append a new message to the messages tuple."""
        if not self._is_messages_mcap_message(self.messages):
            raise TypeError("All messages must be McapMessage instances.")
        if isinstance(msg, McapMessage):
            self.messages.append(msg)
        elif isinstance(msg, McapMessageTuple):
            # check if the channel matches
            if self.channel.topic != msg.channel.topic:
                raise ValueError(
                    f"Channel mismatch: {self.channel.topic} != {msg.channel.topic}"  # noqa: E501
                )
            assert isinstance(msg.message, McapMessage)
            self.messages.append(msg.message)
        else:
            raise TypeError(
                "msg must be an instance of McapMessage or McapMessageTuple"
            )

    def decode(self, decoder_ctx: McapDecoderContext) -> list[Any]:
        """Decode all messages in the tuple using the provided decoder context."""  # noqa: E501
        decoded_messages = []
        for msg in self.messages:
            decoded_message = decoder_ctx.decode_message(
                message_encoding=self.channel.message_encoding,
                message=msg,
                schema=self.schema,
            )
            decoded_messages.append(decoded_message)
        return decoded_messages
