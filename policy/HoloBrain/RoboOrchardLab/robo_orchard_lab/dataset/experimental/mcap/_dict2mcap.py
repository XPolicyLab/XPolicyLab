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
from __future__ import annotations
import copy
import heapq
import os
import warnings
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Type,
    TypeVar,
    overload,
)

import fsspec
from robo_orchard_core.utils.logging import LoggerManager
from robo_orchard_core.utils.registry import Registry
from typing_extensions import Concatenate, ParamSpec, TypeAlias

from robo_orchard_lab.dataset.datatypes import (
    BatchCameraData,
    BatchCameraDataEncoded,
    BatchFrameTransform,
    BatchFrameTransformGraph,
    BatchJointsState,
    BatchPose,
)
from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchFromBatchCameraDataConfig,
    McapBatchFromBatchCameraDataEncodedConfig,
    McapBatchFromBatchFrameTransformConfig,
    McapBatchFromBatchFrameTransformGraphConfig,
    McapBatchFromBatchJointStateConfig,
    McapBatchFromBatchPoseConfig,
)
from robo_orchard_lab.dataset.experimental.mcap.foxglove_writer import (
    FoxgloveMcapWriter as McapWriter,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    FromInstructionConfig,
    FromRobotConfig,
    FromTaskConfig,
)
from robo_orchard_lab.dataset.robot.db_orm import Instruction, Robot, Task

logger = LoggerManager().get_child(__name__)


__all__ = [
    "ToMcapMessageFactory",
    "DefaultToMcapMessage",
    "StampedMessage",
    "Dict2Mcap",
]

P = ParamSpec("P")
T = TypeVar("T")

MessageDict: TypeAlias = dict[str, List[StampedMessage[Any]]]


ConvertCallable = Callable[[Any, int | None], MessageDict]
"""The converter function type. It should have the signature of
(data: Any, log_time: int|None ) -> MessageDict
where data is the input data to convert, log_time is the log time to use for
the converted messages, and MessageDict is the output type of
the converter function.
"""

ConverterCreatorType = Callable[Concatenate[str, Any, P], ConvertCallable]
"""The converter creator type. It should have the signature of
(topic: str, data: Any) -> ConvertCallable
where topic is the topic name for the converter, data is the input data to
convert, and ConvertCallable is the output converter function type.
"""


@dataclass
class TopicStampedMessage:
    topic: str
    idx: int
    message: StampedMessage[Any]

    def __lt__(self, other: TopicStampedMessage) -> bool:
        # log_time is always the first criteria for sorting.
        if (
            self.message.log_time is not None
            and other.message.log_time is not None
        ):
            return self.message.log_time < other.message.log_time

        # if log_time is not available, sort by idx to ensure the order.
        return self.idx < other.idx


class ToMcapMessageFactory:
    """A registry-based factory to build MCAP converters by data type."""

    def __init__(
        self,
    ) -> None:
        self._registry: Registry = Registry(name="AutoToMcapMessageRegistry")

    @overload
    def register(
        self,
        data_type: Type[Any],
        converter_factory: ConverterCreatorType,
    ) -> None: ...

    @overload
    def register(
        self, data_type: Type[Any], converter_factory: None = None
    ) -> Callable[[ConverterCreatorType], ConverterCreatorType]: ...

    def register(
        self,
        data_type: Type[Any],
        converter_factory: ConverterCreatorType | None = None,
    ) -> None | Callable[[ConverterCreatorType], ConverterCreatorType]:
        """Register a converter factory for a specific data type.

        A converter factory is a creator function that takes in the topic
        name and the data, and returns a converter function that has the
        signature of (data: Any, log_time: int) -> MessageDict.

        One sample of data can contains multiple messages for different topics,
        so the converter should return a dictionary of topic name to
        list of stamped messages.


        The converter factory can be registered in two ways:

        1. By directly passing the converter factory function:
            ```
            def my_converter_factory(
                topic: str, data: MyDataType
            ) -> ConvertCallable: ...


            to_mcap_message_factory.register(MyDataType, my_converter_factory)
            ```

        2. By using the register function as a decorator:
            ```
            @to_mcap_message_factory.register(MyDataType)
            def my_converter_factory(
                topic: str, data: MyDataType
            ) -> ConvertCallable: ...
            ```


        Args:
            data_type (Type[Any]): The data type for which the converter
                factory is registered.
            converter_factory (ConverterCreatorType | None): The converter
                factory to register. If None, this function will be used as a
                decorator, and the converter_factory will be the decorated
                function itself.

        """

        # if converter_factory is None, this function will be used
        # as a decorator, and the converter_factory will be the
        # decorated function itself.
        if converter_factory is None:

            def decorator(func: ConverterCreatorType):
                self._registry.register(func, name=str(data_type))

                return func

            return decorator
        else:
            self._registry.register(converter_factory, name=str(data_type))

    def create_converter(
        self, topic: str, data: Any, **kwargs
    ) -> ConvertCallable | None:
        """Create a converter function for the given data and topic.

        Args:
            topic (str): The topic name for the converter.
            data (Any): The data for which the converter is created.
            **kwargs: Additional keyword arguments to pass to the converter
                factory.

        Returns:
            ConvertCallable | None: The converter function created by the
                converter factory. If no converter factory is found for the
                data type, None is returned.

        """
        converter_factory: ConverterCreatorType | None = self._registry.get(
            str(type(data)), raise_not_exist=False
        )
        if converter_factory is None:
            return None
        return converter_factory(topic, data, **kwargs)

    def clone(self) -> ToMcapMessageFactory:
        return copy.deepcopy(self)


DefaultToMcapMessage = ToMcapMessageFactory()


def make_ordered_message_iter(
    data: dict[str, Iterable[StampedMessage[Any]]],
) -> Iterator[TopicStampedMessage]:
    """Make an iterator of TopicStampedMessage sorted by log_time.

    If log_time is not available, the order of messages in the input
    dictionary will be used.

    Args:
        data (dict[str, Iterable[StampedMessage[Any]]]): The input dictionary
            mapping from topic name to iterable of stamped messages.

    Returns:
        Iterator[TopicStampedMessage]: An iterator of TopicStampedMessage
            sorted by log_time.
    """
    # use the first message of each topic to determine whether log_time is
    # available.

    # create a iterator for each topic.
    iterators: dict[str, Iterator[StampedMessage[Any]]] = {
        topic: iter(msgs) for topic, msgs in data.items()
    }
    heap: List[TopicStampedMessage] = []
    has_log_time = {}
    data_idx = 0
    for topic, msg_iter in iterators.items():
        try:
            first_msg = next(msg_iter)
            if not isinstance(first_msg, StampedMessage):
                raise ValueError(
                    f"Expected StampedMessage for topic '{topic}', "
                    f"but got {type(first_msg)}"
                )
            has_log_time[topic] = first_msg.log_time is not None
            # push the first message to the heap.
            heapq.heappush(
                heap,
                TopicStampedMessage(
                    topic=topic, idx=data_idx, message=first_msg
                ),
            )
            data_idx += 1
        except StopIteration:
            # if there is no message for this topic, we can skip it.
            iterators.pop(topic)
            continue

    def get_next_message() -> TopicStampedMessage | None:
        nonlocal data_idx
        if len(heap) == 0:
            return None
        # pop the message with the smallest log_time.
        next_msg = heapq.heappop(heap)
        topic = next_msg.topic
        # push the next message of the same topic to the heap.
        try:
            next_topic_msg = next(iterators[topic])
            if not isinstance(next_topic_msg, StampedMessage):
                raise ValueError(
                    f"Expected StampedMessage for topic '{topic}', "
                    f"but got {type(next_topic_msg)}"
                )
            if has_log_time[topic] != (next_topic_msg.log_time is not None):
                raise ValueError(
                    f"Messages of topic '{topic}' have inconsistent log_time. "
                )

            heapq.heappush(
                heap,
                TopicStampedMessage(
                    topic=topic, idx=data_idx, message=next_topic_msg
                ),
            )
            data_idx += 1
        except StopIteration:
            # if there is no more message for this topic, we can remove it
            # from the iterators.
            iterators.pop(topic)
        return next_msg

    while True:
        next_msg = get_next_message()
        if next_msg is None:
            break
        yield next_msg


class Dict2Mcap:
    def __init__(self, converter_factory: ToMcapMessageFactory | None = None):
        self._converter_factory = converter_factory
        self._default_factory = DefaultToMcapMessage.clone()

    def save_to_mcap(
        self, data: Mapping[str, Iterable[StampedMessage[Any]]], mcap_path: str
    ) -> None:
        """Save the input data to MCAP format.

        Args:
            data (Mapping[str, Iterable[StampedMessage[Any]]]): The input
                dictionary mapping from topic name to iterable of
                stamped messages.
            mcap_path (str): The output path for the MCAP file.
        """
        with fsspec.open(mcap_path, "wb") as f, McapWriter(f) as mcap_writer:  # type: ignore
            for msg in make_ordered_message_iter(data):
                self._write(msg, mcap_writer)

    def _write(
        self, msg: TopicStampedMessage, mcap_writer: McapWriter
    ) -> None:
        # first check if there is a converter for this topic.
        converter = None
        if self._converter_factory is not None:
            converter = self._converter_factory.create_converter(
                topic=msg.topic, data=msg.message.data
            )
        if converter is None:
            # if there is no converter for this topic, we can use the default
            # factory to create a converter.
            converter = self._default_factory.create_converter(
                topic=msg.topic, data=msg.message.data
            )
        if converter is None:
            mcap_writer.write_message(
                topic=msg.topic,
                message=msg.message.data,
                log_time=msg.message.log_time,
            )
            return

        # if there is a converter, we can use it to convert the data to
        # a dictionary of topic name to list of stamped messages, and write
        # each message to the mcap writer.
        converted_data = converter(msg.message.data, msg.message.log_time)
        msg_log_time = msg.message.log_time

        for topic, msgs in converted_data.items():
            if len(msgs) == 0:
                # if there is no message for this topic, we can skip it.
                continue
            log_time_offset = msg_log_time - msgs[0].log_time
            if log_time_offset != 0:
                warnings.warn(
                    f"Log time of the converted messages for topic '{topic}' is "  # noqa: E501
                    f"offset by {log_time_offset} from the original message. "
                    "The log time of the converted messages will be corrected to "  # noqa: E501
                    "match the original message log time."
                )

            for _msg in msgs:
                mcap_writer.write_message(
                    topic=topic,
                    message=_msg.data,
                    log_time=_msg.log_time,
                )


def _wrap_converter_with_log_time(
    converter: Callable[[Any], MessageDict],
) -> Callable[[Any, int | None], MessageDict]:
    def wrapper(data: Any, log_time: int | None) -> MessageDict:
        ret = converter(data)

        if log_time is None:
            return ret
        # correct the log_time of all messages to be start from the
        # given log_time, and keep the relative time
        # difference between messages.
        for _, msgs in ret.items():
            if len(msgs) == 0:
                continue
            log_time_offset = log_time - msgs[0].log_time
            for msg in msgs:
                msg.log_time += log_time_offset

        return ret

    return wrapper


@DefaultToMcapMessage.register(BatchJointsState)
def _default_joint_state_converter(
    topic: str, data: BatchJointsState, **kwargs
):
    return _wrap_converter_with_log_time(
        McapBatchFromBatchJointStateConfig(
            target_topic=topic,
        )().format_batch
    )


@DefaultToMcapMessage.register(BatchPose)
def _default_pose_converter(topic: str, data: BatchPose, **kwargs):
    return _wrap_converter_with_log_time(
        McapBatchFromBatchPoseConfig(
            target_topic=topic,
        )().format_batch
    )


@DefaultToMcapMessage.register(BatchFrameTransform)
def _default_frame_transform_converter(
    topic: str, data: BatchFrameTransform, **kwargs
):
    return _wrap_converter_with_log_time(
        McapBatchFromBatchFrameTransformConfig(
            target_topic=topic,
        )().format_batch
    )


@DefaultToMcapMessage.register(BatchFrameTransformGraph)
def _default_frame_transform_graph_converter(
    topic: str, data: BatchFrameTransformGraph, **kwargs
):
    return _wrap_converter_with_log_time(
        McapBatchFromBatchFrameTransformGraphConfig(
            target_topic=topic,
        )().format_batch
    )


@DefaultToMcapMessage.register(BatchCameraDataEncoded)
def _default_camera_data_encoded_converter(
    topic: str, data: BatchCameraDataEncoded, **kwargs
):
    return _wrap_converter_with_log_time(
        McapBatchFromBatchCameraDataEncodedConfig(
            image_topic=os.path.join(topic, "compressed_image"),
            calib_topic=os.path.join(topic, "calib"),
            tf_topic=os.path.join(topic, "tf"),
        )().format_batch
    )


@DefaultToMcapMessage.register(BatchCameraData)
def _default_camera_data_converter(
    topic: str, data: BatchCameraData, **kwargs
):
    return _wrap_converter_with_log_time(
        McapBatchFromBatchCameraDataConfig(
            image_topic=os.path.join(topic, "raw_image"),
            calib_topic=os.path.join(topic, "calib"),
            tf_topic=os.path.join(topic, "tf"),
        )().format_batch
    )


@DefaultToMcapMessage.register(Robot)
def _default_robot_converter(topic: str, data: Robot, **kwargs):
    converter = FromRobotConfig()()

    def callback(data: Robot, log_time: int | None) -> MessageDict:
        if log_time is None:
            raise ValueError("log_time must be provided for Robot data type.")
        return {
            topic: [
                StampedMessage(data=converter.convert(data), log_time=log_time)
            ]
        }

    return callback


@DefaultToMcapMessage.register(Task)
def _default_task_converter(topic: str, data: Task, **kwargs):
    converter = FromTaskConfig()()

    def callback(data: Task, log_time: int | None) -> MessageDict:
        if log_time is None:
            raise ValueError("log_time must be provided for Task data type.")
        return {
            topic: [
                StampedMessage(data=converter.convert(data), log_time=log_time)
            ]
        }

    return callback


@DefaultToMcapMessage.register(Instruction)
def _default_instruction_converter(topic: str, data: Instruction, **kwargs):
    converter = FromInstructionConfig()()

    def callback(data: Instruction, log_time: int | None) -> MessageDict:
        if log_time is None:
            raise ValueError(
                "log_time must be provided for Instruction data type."
            )
        return {
            topic: [
                StampedMessage(data=converter.convert(data), log_time=log_time)
            ]
        }

    return callback
