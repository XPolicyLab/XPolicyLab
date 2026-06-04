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

from typing import Sequence, Type, TypeGuard, TypeVar

from google.protobuf.message import Message as PbMessage

T = TypeVar("T", bound=PbMessage)


def is_protobuf_msg_type(
    data: PbMessage,
    msg_type: Type[T],
) -> TypeGuard[T]:
    """Check if the data is of a specific protobuf message type.

    MCap protobuf message use schema from mcap file. This result in the class
    type of the protobuf message being different from the one defined in the
    python package. For processing the protobuf message in more convenient
    way, we can use this function as type guard to treat the data as the
    specific protobuf message type.

    Warning:
        This function only checks the full name of the protobuf message
        descriptor. So it is not a strict type check. Use with caution.
    """
    if not isinstance(data, PbMessage):
        return False
    if data.DESCRIPTOR.full_name == msg_type.DESCRIPTOR.full_name:
        return True
    return False


def is_list_of_protobuf_msg_type(
    data: Sequence[PbMessage],
    msg_type: Type[T],
    only_check_first: bool = True,
) -> TypeGuard[list[T]]:
    """Check if the data is of a list of specific protobuf message type.

    MCap protobuf message use schema from mcap file. This result in the class
    type of the protobuf message being different from the one defined in the
    python package. For processing the protobuf message in more convenient
    way, we can use this function as type guard to treat the data as the
    specific protobuf message type.

    Warning:
        This function only checks the full name of the protobuf message
        descriptor. So it is not a strict type check. Use with caution.
    """

    if not isinstance(data, list):
        return False
    if len(data) == 0:
        return False
    if is_protobuf_msg_type(data[0], msg_type):
        if only_check_first:
            return True
        else:
            return all(is_protobuf_msg_type(item, msg_type) for item in data)
    return False
