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
import warnings
from abc import ABCMeta, abstractmethod
from typing import Iterator, Optional, Sequence

from robo_orchard_core.utils.config import Config

from robo_orchard_lab.dataset.experimental.mcap.data_record import (
    McapMessageBatch,
)
from robo_orchard_lab.dataset.experimental.mcap.reader import (
    MakeIterMsgArgs,
    McapMessageTuple,
    McapReader,
)

__all__ = [
    "BatchSplitMixin",
    "SplitBatchByTopicArgs",
    "SplitBatchByTopics",
    "iter_messages_batch",
]


class BatchSplitMixin(metaclass=ABCMeta):
    """Mixin for batch splitting logic in McapReader.

    The message iterator can be configured to split messages into
    batches based on certain criteria. This mixin defines the interface
    for determining whether a message should trigger a new batch.

    """

    @abstractmethod
    def reset(self):
        """Reset the internal state of the batch splitter."""
        raise NotImplementedError

    @abstractmethod
    def need_split(self, msg: McapMessageTuple) -> bool:
        """Determine if the message needs to be split into a new batch.

        This method accepts time-ordered messages and determines
        whether the current message should trigger a new batch based
        on the internal logic of the batch splitter.

        Note:
            This method expects message to be in log_time_order and
            not reversed.

        """
        raise NotImplementedError


class SplitBatchByTopicArgs(Config):
    monitor_topic: str
    min_messages_per_topic: int | None = None
    max_messages_per_topic: int | None = None
    lookahead_duration: int = 0
    """Lookahead duration in nanoseconds."""

    def __post_init__(self):
        if (
            self.min_messages_per_topic is None
            and self.max_messages_per_topic is None
        ):
            raise ValueError(
                "Either min_messages_per_topic or max_messages_per_topic "
                "must be specified."
            )

        if self.max_messages_per_topic is None:
            self.max_messages_per_topic = self.min_messages_per_topic

        if self.min_messages_per_topic is None:
            self.min_messages_per_topic = self.max_messages_per_topic

        if self.max_messages_per_topic < self.min_messages_per_topic:  # type: ignore
            raise ValueError(
                "max_messages_per_topic must be greater than or equal to "
                "min_messages_per_topic."
            )


class SplitBatchByTopic(BatchSplitMixin):
    """Batch split based on topic.

    This class will split the messages into batches under the following
    conditions:
    - If all monitored topics have at least `min_messages_per_topic`
      messages, it will set a lookahead timestamp threshold for
      the current batch.
    - If any monitored topic is going to exceed `max_messages_per_topic`
      messages, it will trigger a new batch immediately without considering
      the lookahead timestamp threshold.
    - Try to collect as many messages as possible from the message iterator
      as long as the above conditions are not triggered.

    """

    def __init__(self, args: SplitBatchByTopicArgs):
        self._args = args
        assert args.min_messages_per_topic is not None
        assert args.max_messages_per_topic is not None
        self._topic = args.monitor_topic
        self._min_messages_per_topic = args.min_messages_per_topic
        self._max_messages_per_topic = args.max_messages_per_topic
        self._lookahead_duration = args.lookahead_duration
        self.reset()

    @property
    def topic(self) -> str:
        return self._args.monitor_topic

    def reset(self):
        self._topic_msg_count = 0
        self._next_batch_timestamp = None

    def _update_state(self, msg: McapMessageTuple):
        """Update the internal state based on the current message.

        If all monitored topics have enough messages, it will
        caculate the next batch timestamp based on the current message.
        """
        topic = msg.channel.topic
        if topic == self._topic:
            self._topic_msg_count += 1

            if (
                self._topic_msg_count >= self._min_messages_per_topic
                and self._next_batch_timestamp is None
            ):
                # If all monitored topics have enough messages,
                # reset the state for the next batch.
                self._next_batch_timestamp = (
                    msg.message.log_time + self._lookahead_duration
                )

    def need_split(self, msg: McapMessageTuple) -> bool:
        """Determine if the message needs to be split into a new batch.

        This method checks if the current message's topic is in the
        configured topics and whether it meets the criteria for
        splitting based on the minimum messages per topic and lookahead
        duration.

        Args:
            msg (McapMessageTuple): The message to check.

        Returns:
            bool: True if a new batch should be started, False otherwise.
        """

        # first to determin if current message trigger a new batch
        # by timestamp threshold.
        # _next_batch_timestamp only set if all monitored topics
        # have enough messages.
        log_time = msg.message.log_time
        if (
            self._next_batch_timestamp is not None
            and log_time > self._next_batch_timestamp
        ):
            self.reset()
            self._update_state(msg)
            return True

        # we need to split if any monitored topic has more than
        # _max_messages_per_topic messages.
        topic = msg.channel.topic
        if (
            topic == self._topic
            and self._topic_msg_count + 1 > self._max_messages_per_topic
        ):
            self.reset()
            self._update_state(msg)
            return True

        # Try to get messages as much as possible
        # if message count is not enough.
        self._update_state(msg)
        return False


class SplitBatchByTopics(BatchSplitMixin):
    """Batch split based on multiple topics.

    This class will split the messages into batches based on the
    provided topics and their respective message counts.
    It contains multiple `SplitBatchByTopic` instances, and each
    instance is responsible for a specific topic's splitting logic.
    Once any of the topics requires a new batch, it will reset
    all splits and return True.

    """

    def __init__(
        self,
        args_list: Sequence[SplitBatchByTopicArgs] | SplitBatchByTopicArgs,
    ):
        if isinstance(args_list, SplitBatchByTopicArgs):
            args_list = [args_list]

        self._splits: dict[str, SplitBatchByTopic] = {}
        for args in args_list:
            if args.monitor_topic in self._splits:
                raise ValueError(
                    f"Duplicate topic found: {args.monitor_topic}"
                )
            self._splits[args.monitor_topic] = SplitBatchByTopic(args=args)

    def reset(self):
        """Reset the internal state of all splits."""
        for _, split in self._splits.items():
            split.reset()

    def need_split(self, msg: McapMessageTuple) -> bool:
        """Determine if the message needs to be split into a new batch.

        This method checks all configured splits and returns True if
        any of them requires a new batch based on the current message.

        Args:
            msg (McapMessageTuple): The message to check.

        Returns:
            bool: True if a new batch should be started, False otherwise.
        """

        cur_topic = msg.topic
        if cur_topic not in self._splits:
            return False

        # first check if any split needs to be triggered
        ret = self._splits[cur_topic].need_split(msg)

        # if any split needs to be triggered, reset all splits
        # and use the message to update their states.
        if ret:
            for topic, split in self._splits.items():
                # reset the state for other topics
                if topic == cur_topic:
                    continue
                else:
                    split.reset()

        return ret


def iter_messages_batch(
    reader: McapReader,
    batch_split: BatchSplitMixin,
    iter_config: Optional[MakeIterMsgArgs] = None,
    do_not_split_same_log_time: bool = True,
    keep_last_topic_msgs: bool = True,
) -> Iterator[McapMessageBatch]:
    """Iterate over messages in batches.

    Args:
        reader (McapReader): The MCAP reader to iterate messages from.
        batch_split (BatchSplitMixin): The batch splitting logic to apply.
        iter_config (Optional[MakeIterMsgArgs]): Configuration for message
            iteration.
        do_not_split_same_log_time (bool, optional): If True, do not split
            batches within the same log time. This feature is useful to make
            sure that the batches are split based on log time. In some cases
            that split must be done within the same log time, this will
            trigger a warning and yield the current batch as is. Defaults to
            True.
        keep_last_topic_msgs (bool, optional): If True, keep the last messages
            for each topic in the returned batch. This is useful for
            maintaining the last message for each topic in the mcap file, such
            as camera calibration message, which does not change during the
            recording. Defaults to True.

    """
    if iter_config is None:
        iter_config = MakeIterMsgArgs()

    cur_batch = McapMessageBatch({}, is_last_batch=True)

    if iter_config.log_time_order is not True:
        raise ValueError("Batch splitting requires log_time_order to be True.")
    if iter_config.reverse is not False:
        raise ValueError("Batch splitting does not support reverse iteration.")

    last_msgs: dict[str, McapMessageTuple] = {}
    """The last messages for each topic for current timestamp."""
    prev_last_msgs: dict[str, McapMessageTuple] = {}
    """The last messages for each topic for previous timestamp."""
    prev_ts = None

    def fix_last_messages(to_return: McapMessageBatch) -> None:
        """Fix the last messages in the batch when splitted by current timestamp."""  # noqa: E501
        # In some case, the last_messages may point to the
        # future messages. This can happen for the left
        # part of split.
        # If this happens, we need to reset the
        # last_messages to the previous state.
        max_log_time = to_return.max_log_time
        if to_return.last_messages is not None:
            for _, msg in to_return.last_messages.items():
                if msg.message.log_time > max_log_time:
                    # if the last message is in the future, we need to
                    # set it to None
                    to_return.last_messages = prev_last_msgs.copy()
                    return

    for msg in reader.iter_messages(iter_config=iter_config):
        if prev_ts != msg.message.log_time:
            prev_ts = msg.message.log_time
            prev_last_msgs = last_msgs.copy()
        last_msgs[msg.topic] = msg
        if batch_split.need_split(msg):
            # set the last_batch flat to false and
            # return the previous batch
            cur_batch.is_last_batch = False
            if do_not_split_same_log_time:
                msg_log_time = msg.message.log_time
                # it is save to assume that the messages are in
                # log_time_order, because we have checked
                # iter_config.log_time_order and iter_config.reverse.
                to_return, cur_batch = cur_batch.split(
                    msg_log_time, is_sorted_asc=True
                )
                # if have to split the batch within the same log time,
                # we need to yield the current batch and reset it
                if to_return is None:
                    warnings.warn(
                        "Batch splitting is triggered within the same "
                        "log time to maintain a valid batch when try not "
                        "to split within log time. This may lead to "
                        "unexpected behavior. ",
                        UserWarning,
                    )
                    to_return = cur_batch
                    cur_batch = McapMessageBatch({}, is_last_batch=True)

                assert to_return is not None
                fix_last_messages(to_return)
                yield to_return
                if cur_batch is None:
                    cur_batch = McapMessageBatch({}, is_last_batch=True)
            else:
                yield cur_batch
                cur_batch = McapMessageBatch({}, is_last_batch=True)

        cur_batch.append(msg)
        if keep_last_topic_msgs:
            cur_batch.last_messages = last_msgs.copy()

    assert cur_batch is not None
    yield cur_batch
