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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Literal, Sequence

from robo_orchard_lab.dataset.experimental.mcap.messages import (
    McapMessagesTuple,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    MakeIterMsgArgs,
    McapMessageTuple,
    McapReader,
)

if TYPE_CHECKING:
    from robo_orchard_lab.dataset.experimental.mcap.msg_decoder.decoder_ctx import (  # noqa: E501
        McapDecoderContext,
    )

__all__ = ["McapDataRecordChunk", "McapDataRecordChunks"]


@dataclass
class McapMessageBatch:
    """A batch of messages grouped by topic."""

    message_dict: dict[str, McapMessagesTuple]
    is_last_batch: bool = False

    last_messages: dict[str, McapMessageTuple] | None = None
    """The last messages of each topic in the reader.

    This field is used to store static messages that are not yet
    included in the batch.
    """

    def filter_by_topic(self, keys: Iterable[str]) -> McapMessageBatch:
        """Filter the batch by the specified keys (topics).

        Args:
            keys (Iterable[str]): The topics to filter by.

        Returns:
            McapMessageBatch: A new batch containing only the specified topics.
        """
        filtered_msgs = {}
        for key in keys:
            if key in self.message_dict:
                filtered_msgs[key] = self.message_dict[key]

        filtered_last_messages = None
        if self.last_messages is not None:
            filtered_last_messages = {}
            for key in keys:
                if key in self.last_messages:
                    filtered_last_messages[key] = self.last_messages[key]

        return McapMessageBatch(
            message_dict=filtered_msgs,
            is_last_batch=self.is_last_batch,
            last_messages=filtered_last_messages,
        )

    @property
    def min_log_times(self) -> dict[str, int]:
        """Return the minimum log time for each topic in the batch."""
        return {
            topic: messages_tuple.min_log_time
            for topic, messages_tuple in self.message_dict.items()
        }

    @property
    def min_log_time(self) -> int:
        """Return the minimum log time across all topics in the batch."""
        return min(self.min_log_times.values())

    @property
    def max_log_times(self) -> dict[str, int]:
        """Return the maximum log time for each topic in the batch."""
        return {
            topic: messages_tuple.max_log_time
            for topic, messages_tuple in self.message_dict.items()
        }

    @property
    def max_log_time(self) -> int:
        """Return the maximum log time across all topics in the batch."""
        return max(self.max_log_times.values())

    @property
    def topics(self) -> list[str]:
        """Return a list of topics in the batch."""
        return list(self.message_dict.keys())

    def __getitem__(self, topic: str) -> McapMessagesTuple:
        """Get the messages tuple for the specified topic."""
        if topic not in self.message_dict:
            raise KeyError(f"Topic '{topic}' not found in the batch.")
        return self.message_dict[topic]

    def __contains__(self, topic: str) -> bool:
        """Check if the topic is in the batch."""
        return topic in self.message_dict

    def __len__(self) -> int:
        """Return the total number of messages in the batch."""
        return sum(
            len(messages_tuple)
            for messages_tuple in self.message_dict.values()
        )

    def split(
        self,
        log_time: int,
        is_sorted_asc: bool = False,
    ) -> tuple[McapMessageBatch | None, McapMessageBatch | None]:
        """Split the batch into two parts based on the log time.

        This method returns two new `McapMessageBatch` instances:
        - The first contains messages with `log_time < log_time`.
        - The second contains messages with `log_time >= log_time`.

        Args:
            log_time (int): The log time to split the messages.
            is_sorted_asc (bool, optional): Whether the messages are already
                sorted in ascending order by log time. If True, a binary search
                will be used to find the split point. Defaults to False.

        """
        left_dict = {}
        right_dict = {}

        for topic, messages_tuple in self.message_dict.items():
            left_tuple, right_tuple = messages_tuple.split(
                log_time, is_sorted_asc=is_sorted_asc
            )
            if left_tuple is not None:
                left_dict[topic] = left_tuple
            if right_tuple is not None:
                right_dict[topic] = right_tuple

        left_ret = (
            McapMessageBatch(
                left_dict,
                is_last_batch=self.is_last_batch,
                last_messages=self.last_messages,
            )
            if len(left_dict) > 0
            else None
        )
        right_ret = (
            McapMessageBatch(
                right_dict,
                is_last_batch=self.is_last_batch,
                last_messages=self.last_messages,
            )
            if len(right_dict) > 0
            else None
        )

        return (left_ret, right_ret)

    def append(self, msg: McapMessagesTuple | McapMessageTuple) -> None:
        """Append a new messages tuple to the batch."""
        c_id = msg.topic

        if isinstance(msg, McapMessagesTuple):
            if c_id not in self.message_dict:
                self.message_dict[c_id] = msg
            else:
                self.message_dict[c_id].messages.extend(msg.messages)

        else:
            if c_id not in self.message_dict:
                self.message_dict[c_id] = McapMessagesTuple(
                    schema=msg.schema,
                    channel=msg.channel,
                    messages=[msg.message],
                )
            else:
                self.message_dict[c_id].append(msg)

    def decode(self, decoder_ctx: McapDecoderContext) -> dict[str, list[Any]]:
        """Decode all messages using the provided decoder context."""
        decoded_batch = {}
        for c_id, messages_tuple in self.message_dict.items():
            decoded_messages = messages_tuple.decode(decoder_ctx)
            decoded_batch[c_id] = decoded_messages
        return decoded_batch

    def sort(
        self,
        key: Literal["log_time", "publish_time"] = "log_time",
        reverse: bool = False,
    ) -> None:
        """Sort all messages in the batch by the specified key."""
        for messages_tuple in self.message_dict.values():
            messages_tuple.sort(key=key, reverse=reverse)

        # sort message_dict key as well
        self.message_dict = dict(
            sorted(
                self.message_dict.items(),
                key=lambda item: item[0],
                reverse=reverse,
            )
        )

    def extend(
        self, other: McapMessageBatch | Sequence[McapMessagesTuple]
    ) -> None:
        if isinstance(other, McapMessageBatch):
            for c_id, messages_tuple in other.message_dict.items():
                if c_id not in self.message_dict:
                    self.message_dict[c_id] = messages_tuple
                else:
                    self.message_dict[c_id].messages.extend(
                        messages_tuple.messages
                    )
        else:
            for messages_tuple in other:
                self.append(messages_tuple)

    def merge_last_messages(
        self, sort: bool = True, missing_topic_only: bool = False
    ) -> None:
        """Merge the last messages in the batch.

        After merge, the last messages will be cleared.

        Args:
            sort (bool, optional): Whether to sort the messages after merging.
                Defaults to True.
            missing_topic_only (bool, optional): If True, only merge the last
                messages for topics that are not already in the batch or have
                no messages in the batch. If False, merge all last messages.
                Defaults to False.
        """
        if self.last_messages is None:
            return

        for topic, msg in self.last_messages.items():
            if topic not in self.message_dict:
                self.message_dict[topic] = McapMessagesTuple(
                    schema=msg.schema,
                    channel=msg.channel,
                    messages=[msg.message],
                )
                continue
            if not missing_topic_only or (
                missing_topic_only and len(self.message_dict[topic]) == 0
            ):
                is_msg_already_in_batch = False
                for m in self.message_dict[topic].messages:
                    if (
                        m.log_time == msg.message.log_time
                        and m.data == msg.message.data
                    ):
                        is_msg_already_in_batch = True
                        break
                if not is_msg_already_in_batch:
                    self.message_dict[topic].messages.append(msg.message)
                    if sort:
                        self.message_dict[topic].sort(
                            key="log_time", reverse=False
                        )
        self.last_messages = None


@dataclass
class McapDataRecordChunk:
    """Data record chunk for reading messages from an MCAP file.

    The data record chunk is defined by messages in  specific topics and
    a time range. One data record can contain multiple messages.

    For training purposes, a data record is usually treated as one
    training sample, which contains all messages in the specified
    topics within the specified time range.
    """

    topics: list[str]
    """List of topics to read messages from."""
    log_time_start: int
    """Start log_time of messages in nanoseconds."""
    log_time_end: int
    """End log_time(excluded) of messages in nanoseconds."""

    @staticmethod
    def from_message_batch(batch: McapMessageBatch) -> McapDataRecordChunk:
        """Create a data record chunk from a message batch."""
        topics = batch.topics
        if not topics:
            raise ValueError("No topics found in the message batch.")
        return McapDataRecordChunk(
            topics=topics,
            log_time_start=batch.min_log_time,
            log_time_end=batch.max_log_time + 1,
        )

    def read(self, mcap_reader: McapReader) -> McapMessageBatch:
        """Read the data record from the mcap reader."""
        ret = McapMessageBatch({}, is_last_batch=True)
        for msg in mcap_reader.iter_messages(
            MakeIterMsgArgs(
                topics=self.topics,
                start_time=self.log_time_start,
                end_time=self.log_time_end,
                log_time_order=True,
                reverse=False,
            )
        ):
            ret.append(msg)
        return ret


@dataclass
class McapDataRecordChunks:
    """Data record chunks for reading from an MCAP file.

    In some cases, for example, we need to read multiple messages from
    different time ranges and groups them into a single data record,
    we can use this class to represent multiple data record chunks.
    """

    chunks: list[McapDataRecordChunk]
    """List of data record chunks to read messages from."""

    def read(self, mcap_reader: McapReader) -> McapMessageBatch:
        """Read the data chunk record from the mcap reader."""
        ret = McapMessageBatch({}, is_last_batch=True)
        for record in self.chunks:
            batch = record.read(mcap_reader)
            ret.extend(batch)
        ret.sort()
        return ret

    @staticmethod
    def from_message_batch(
        batch: McapMessageBatch,
        seperate_by_topics: bool = True,
    ) -> McapDataRecordChunks:
        """Create a data record chunk from a message batch.

        Args:
            batch (McapMessageBatch): The message batch to create the data
                record chunk from.
            seperate_by_topics (bool, optional): Whether to create separate
                chunks for topics with different time range. If False, all
                messages in the batch will be grouped into a single chunk.
                Defaults to True.
        """
        if not seperate_by_topics:
            return McapDataRecordChunks(
                [McapDataRecordChunk.from_message_batch(batch)]
            )
        topics = batch.topics
        min_log_times = batch.min_log_times
        max_log_times = batch.max_log_times

        min_max_time_topics = {
            p: []
            for p in set(
                [
                    (min_log_times[topic], max_log_times[topic])
                    for topic in topics
                ]
            )
        }
        for topic in topics:
            min_max_time_topics[
                (min_log_times[topic], max_log_times[topic])
            ].append(topic)

        return McapDataRecordChunks(
            chunks=[
                McapDataRecordChunk(
                    topics=v,
                    log_time_start=k[0],
                    log_time_end=k[1] + 1,
                )
                for k, v in min_max_time_topics.items()
            ]
        )
