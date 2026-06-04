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
import json

from robo_orchard_core.utils.config import ClassType
from robo_orchard_schemas.action_msgs.instruction_pb2 import InstructionStamped
from robo_orchard_schemas.action_msgs.task_pb2 import TaskStamped
from robo_orchard_schemas.robot_msgs.robot_desc_pb2 import (
    RobotDesc,
    RobotDescFormat,
)

from robo_orchard_lab.dataset.experimental.mcap.msg_converter.base import (
    MessageConverterConfig,
    MessageConverterStateless,
)
from robo_orchard_lab.dataset.robot.db_orm import (
    Instruction,
    Robot,
    RobotDescriptionFormat,
    Task,
)

__all__ = [
    "FromRobot",
    "FromRobotConfig",
    "FromTask",
    "FromTaskConfig",
    "FromInstruction",
    "FromInstructionConfig",
]


class FromRobot(MessageConverterStateless[Robot, RobotDesc]):
    cfg: FromRobotConfig

    def __init__(self, cfg: FromRobotConfig):
        self.cfg = cfg

    def convert(self, data: Robot) -> RobotDesc:
        content = "" if data.content is None else data.content
        if data.content_format == RobotDescriptionFormat.URDF:
            content_fmt = RobotDescFormat.URDF
        elif data.content_format == RobotDescriptionFormat.MJCF:
            content_fmt = RobotDescFormat.MJCF
        else:
            raise NotImplementedError(
                f"Robot {data.name} has unsupported description format: "
                f"{data.content_format}. Supported formats are: "
                f"{RobotDescriptionFormat.URDF} and "
                f"{RobotDescriptionFormat.MJCF}."
            )

        return RobotDesc(
            format=content_fmt,
            content=content,
        )


class FromTask(MessageConverterStateless[Task, TaskStamped]):
    cfg: FromTaskConfig

    def __init__(self, cfg: FromTaskConfig):
        self.cfg = cfg

    def convert(self, data: Task) -> TaskStamped:
        return TaskStamped(
            names=[data.name],
            descriptions=[data.description] if data.description else [],
        )


class FromInstruction(
    MessageConverterStateless[Instruction, InstructionStamped]
):
    cfg: FromInstructionConfig

    def __init__(self, cfg: FromInstructionConfig):
        self.cfg = cfg

    def convert(self, data: Instruction) -> InstructionStamped:
        instruction = data
        return InstructionStamped(
            names=[instruction.name] if instruction.name else [],
            descriptions=[
                json.dumps(instruction.json_content, ensure_ascii=False)
            ],
        )


class FromRobotConfig(MessageConverterConfig[FromRobot]):
    class_type: ClassType[FromRobot] = FromRobot


class FromTaskConfig(MessageConverterConfig[FromTask]):
    class_type: ClassType[FromTask] = FromTask


class FromInstructionConfig(MessageConverterConfig[FromInstruction]):
    class_type: ClassType[FromInstruction] = FromInstruction
