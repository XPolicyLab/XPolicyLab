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
from typing import IO, TYPE_CHECKING, Iterator, Optional, Sequence

from robo_orchard_core.utils.config import Config

from mcap.exceptions import McapError
from mcap.reader import (
    McapReader as _McapReader,
    make_reader as mcap_make_reader,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import (
    McapDecodedMessageTuple,
    McapMessageTuple,
)

if TYPE_CHECKING:
    from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
        McapDecoderContext,
    )

__all__ = [
    "McapMessageTuple",
    "McapDecodedMessageTuple",
    "MakeIterMsgArgs",
    "McapReader",
]


class MakeIterMsgArgs(Config):
    """Configuration for iterating messages in McapReader."""

    topics: Optional[Sequence[str]] = None
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    log_time_order: bool = True
    reverse: bool = False
    start_offset: Optional[int] = None
    duration: Optional[int] = None

    def __post_init__(self):
        """Post-initialization to ensure valid time range configuration."""

        if self.duration is not None and self.end_time is not None:
            raise ValueError("Cannot specify both duration and end_time.")

    def update_time_range(self, message_start_time: int):
        if self.duration is not None and self.end_time is not None:
            raise ValueError("Cannot specify both duration and end_time.")

        if self.start_offset is not None:
            if self.start_time is None:
                self.start_time = message_start_time
            self.start_time += self.start_offset
            # reset start_offset to None to maintain consistency
            self.start_offset = None

        if self.duration is not None:
            if self.start_time is None:
                self.start_time = message_start_time
            self.end_time = self.start_time + self.duration
            # reset duration to None to maintain consistency
            self.duration = None


class McapReader:
    """A wrapper around the mcap reader to provide more convenient access.

    New features compared to the original mcap reader:
    - Separate the decoding logic from the reader for better flexibility.
    - Provide batch reading of messages with configurable splitting.

    """

    def __init__(self, reader: _McapReader):
        self.reader = reader

        # Expose the reader's methods for compatibility
        self.get_header = reader.get_header
        self.get_summary = reader.get_summary
        self.iter_attachments = reader.iter_attachments
        self.iter_metadata = reader.iter_metadata

    @staticmethod
    def make_reader(
        stream: IO[bytes],
        validate_crcs: bool = False,
    ) -> McapReader:
        return McapReader(
            mcap_make_reader(stream=stream, validate_crcs=validate_crcs)
        )

    def _update_time_range(self, iter_config: MakeIterMsgArgs):
        """Update the start and end time based on the provided parameters.

        If start_offset or duration is provided, it will adjust the start_time
        and end_time based on the message start time from the summary
        statistics. Both time and offset/duration should be in
        nanoseconds(10^9 nanoseconds = 1 second).

        """

        if (
            iter_config.start_offset is not None
            or iter_config.duration is not None
        ):
            summary = self.get_summary()
            if summary is None:
                raise McapError("Summary is not available in the reader.")
            statistics = summary.statistics
            if statistics is None:
                raise McapError("Statistics are not available in the reader.")
            iter_config.update_time_range(
                message_start_time=statistics.message_start_time
            )

    def iter_messages(
        self,
        iter_config: Optional[MakeIterMsgArgs] = None,
    ) -> Iterator[McapMessageTuple]:
        if iter_config is None:
            iter_config = MakeIterMsgArgs()

        self._update_time_range(iter_config)

        topics = iter_config.topics
        for schema, channel, msg in self.reader.iter_messages(
            topics=topics,
            start_time=iter_config.start_time,
            end_time=iter_config.end_time,
            log_time_order=iter_config.log_time_order,
            reverse=iter_config.reverse,
        ):
            yield McapMessageTuple(
                schema=schema,
                channel=channel,
                message=msg,
            )

    def iter_decoded_messages(
        self,
        decoder_ctx: McapDecoderContext,
        iter_config: Optional[MakeIterMsgArgs] = None,
    ) -> Iterator[McapDecodedMessageTuple]:
        for msg in self.iter_messages(iter_config=iter_config):
            decoded_message = decoder_ctx.decode_message(
                message_encoding=msg.channel.message_encoding,
                message=msg.message,
                schema=msg.schema,
            )
            yield McapDecodedMessageTuple(
                schema=msg.schema,
                channel=msg.channel,
                message=msg.message,
                decoded_message=decoded_message,
            )
