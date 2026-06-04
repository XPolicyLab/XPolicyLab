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

from dataclasses import dataclass
from typing import Mapping

import fsspec
from robo_orchard_schemas.action_msgs.instruction_pb2 import (
    InstructionStamped as PbInstructionStamped,
)
from robo_orchard_schemas.action_msgs.task_pb2 import (
    TaskStamped as PbTaskStamped,
)
from robo_orchard_schemas.robot_msgs.robot_desc_pb2 import RobotDesc

from robo_orchard_lab.dataset.experimental.mcap.batch_encoder import (
    McapBatchEncoderConfig,
    McapBatchEncoders,
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
from robo_orchard_lab.dataset.robot.columns import (
    PreservedIndexColumns,
    PreservedIndexColumnsKeys,
)
from robo_orchard_lab.dataset.robot.dataset import RODataset
from robo_orchard_lab.dataset.robot.db_orm import (
    Episode,
    Instruction,
    Robot,
    Task,
)

__all__ = [
    "Dataset2Mcap",
    "EpisodeInfoTopics",
]


@dataclass
class EpisodeInfoTopics:
    """A dataclass to hold the topics for episode information."""

    robot_topic: str = "/robot_description/urdf"
    task_topic: str = "/action/task"
    instruction_topic: str = "/action/instruction"


class Dataset2Mcap:
    """A class to save the RoboOrchard dataset to an MCAP file.

    TODO: To be refactored to use dict2mcap for flexibility and modularity!
    """

    def __init__(self, dataset: RODataset):
        self.dataset = dataset

    def _robot2proto(
        self, robot: Robot, log_time: int, pub_time: int | None = None
    ) -> StampedMessage[RobotDesc]:
        """Convert a Robot object to a URDF protobuf message.

        Args:
            robot (Robot): The Robot object to convert.
            log_time (int): The log time in nanoseconds.
            pub_time (int | None): The publication time in nanoseconds, if
                available.

        Returns:
            StampedMessage[RobotDesc]: The converted RobotDesc
                protobuf message.
        """
        to_mcap_msg = FromRobotConfig()()
        return StampedMessage(
            data=to_mcap_msg.convert(robot),
            log_time=log_time,
            pub_time=pub_time,
        )

    def _task2proto(
        self,
        task: Task,
        timestamp: int,
        log_time: int,
        pub_time: int | None = None,
    ) -> StampedMessage[PbTaskStamped]:
        """Convert a Task object to a TaskStamped protobuf message.

        Args:
            task (Task): The Task object to convert.
            timestamp (int): The timestamp in nanoseconds.
            log_time (int): The log time in nanoseconds.
            pub_time (int | None): The publication time in nanoseconds, if
                available.

        Returns:
            StampedMessage[PbTaskStamped]: The converted TaskStamped
                protobuf message.
        """
        to_mcap_msg = FromTaskConfig()()
        data = to_mcap_msg.convert(task)
        data.timestamp.FromNanoseconds(timestamp)
        return StampedMessage(
            data=data,
            log_time=log_time,
            pub_time=pub_time,
        )

    def _instruction2proto(
        self,
        instruction: Instruction,
        timestamp: int,
        log_time: int,
        pub_time: int | None = None,
    ) -> StampedMessage[PbInstructionStamped]:
        """Convert an Instruction object to a InstructionStamped message.

        Args:
            instruction (Instruction): The Instruction object to convert.
            timestamp (int): The timestamp in nanoseconds.
            log_time (int): The log time in nanoseconds.
            pub_time (int | None): The publication time in nanoseconds, if
                available.

        Returns:
            StampedMessage[PbInstructionStamped]: The converted
                InstructionStamped protobuf message.
        """
        to_mcap_msg = FromInstructionConfig()()
        data = to_mcap_msg.convert(instruction)
        data.timestamp.FromNanoseconds(timestamp)

        return StampedMessage(
            data=data,
            log_time=log_time,
            pub_time=pub_time,
        )

    def save_episode(
        self,
        target_path: str,
        episode_index: int,
        encoder_cfg: Mapping[str, McapBatchEncoderConfig],
        episode_info_topics: EpisodeInfoTopics | None = None,
    ):
        """Save the episode data to an MCAP file.

        Args:
            target_path (str): The path to save the MCAP file.
            episode_index (int): The index of the episode to save.
            encoder_cfg (Mapping[str, McapBatchEncoderConfig]): The
                configuration for the MCAP batch encoder.
            episode_info_topics (EpisodeInfoTopics | None, optional):
                The topics for episode information such as robot,
                task, and instruction. If None, default topics will be used.
                Defaults to None.
        """
        if episode_info_topics is None:
            episode_info_topics = EpisodeInfoTopics()
        episode_info = self.dataset.get_meta(Episode, episode_index)
        if episode_info is None:
            raise ValueError(f"Episode with index {episode_index} not found.")
        to_mcap_msg_batch = McapBatchEncoders(encoder_cfg)
        robot_info = (
            self.dataset.get_meta(Robot, episode_info.robot_index)
            if episode_info.robot_index is not None
            else None
        )
        task_info = (
            self.dataset.get_meta(Task, episode_info.task_index)
            if episode_info.task_index is not None
            else None
        )
        begin = episode_info.dataset_begin_index
        end = begin + episode_info.frame_num

        first_frame = self.dataset.frame_dataset[begin]
        start_ts: int | None = first_frame["timestamp_min"]
        if start_ts is None:
            raise ValueError(
                f"Episode {episode_index} does not have a valid "
                "timestamp for the first frame. "
                f"To use this feature, please ensure the dataset "
                "has been properly indexed with timestamps."
            )

        with fsspec.open(target_path, "wb") as f, McapWriter(f) as mcap_writer:  # type: ignore
            if robot_info is not None:
                try:
                    robot_msg = self._robot2proto(
                        robot_info,
                        log_time=start_ts,
                        pub_time=start_ts,
                    )
                    mcap_writer.write_message(
                        topic=episode_info_topics.robot_topic,
                        message=robot_msg.data,
                        log_time=robot_msg.log_time,
                        publish_time=robot_msg.pub_time,
                    )
                except NotImplementedError as e:
                    print(
                        f"Skipping robot description for episode "
                        f"{episode_index}: {e}"
                    )

            if task_info is not None:
                task_msg = self._task2proto(
                    task_info,
                    timestamp=start_ts,
                    log_time=start_ts,
                    pub_time=start_ts,
                )
                mcap_writer.write_message(
                    topic=episode_info_topics.task_topic,
                    message=task_msg.data,
                    log_time=task_msg.log_time,
                    publish_time=task_msg.pub_time,
                )

            for idx in range(begin, end):
                frame = self.dataset.frame_dataset[idx]
                preserved_index_columns = PreservedIndexColumns(
                    **{k: frame.pop(k) for k in PreservedIndexColumnsKeys}
                )
                ts_min = preserved_index_columns.timestamp_min
                if ts_min is None:
                    raise ValueError(
                        f"Frame {idx} in episode {episode_index} does not have "  # noqa: E501
                        "a valid timestamp. Please ensure the dataset has been "  # noqa: E501
                        "properly indexed with timestamps."
                    )
                instruction_info = (
                    self.dataset.get_meta(
                        Instruction, preserved_index_columns.instruction_index
                    )
                    if preserved_index_columns.instruction_index is not None
                    else None
                )
                if instruction_info is not None:
                    instruction_msg = self._instruction2proto(
                        instruction_info,
                        timestamp=ts_min,
                        log_time=ts_min,
                        pub_time=ts_min,
                    )
                    mcap_writer.write_message(
                        topic=episode_info_topics.instruction_topic,
                        message=instruction_msg.data,
                        log_time=instruction_msg.log_time,
                        publish_time=instruction_msg.pub_time,
                    )

                # encode the frame data
                msg_batch = to_mcap_msg_batch.format_batch(
                    frame, raise_if_encoder_not_found=False
                )
                for topic, msgs in msg_batch.items():
                    if len(msgs) == 0:
                        continue
                    for msg in msgs:
                        mcap_writer.write_message(
                            topic=topic,
                            message=msg.data,
                            log_time=msg.log_time,
                            publish_time=msg.pub_time,
                        )
