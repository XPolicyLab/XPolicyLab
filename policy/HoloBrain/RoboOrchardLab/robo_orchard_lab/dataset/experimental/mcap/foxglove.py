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


from typing import Any, Type

from foxglove import Channel, Context, Schema
from google.protobuf.message import Message as ProtobufMessage
from mcap_protobuf.schema import build_file_descriptor_set

__all__ = [
    "create_schema",
    "create_channel",
    "create_channels_from_examples",
]


def create_schema(msg_type: Type[Any] | Any) -> Schema | None:
    # if is subclass of ProtobufMessage, we build a schema from the
    # protobuf definition
    if issubclass(msg_type, ProtobufMessage):
        file_descriptor_set = build_file_descriptor_set(msg_type)
        return Schema(
            name=msg_type.DESCRIPTOR.full_name,
            encoding="protobuf",
            data=file_descriptor_set.SerializeToString(),
        )
    else:
        return None


def create_channel(
    topic: str,
    msg_type: Type[Any],
    context: Context | None = None,
    schema: Schema | None = None,
    metadata: dict[str, str] | None = None,
) -> Channel:
    if schema is None:
        schema = create_schema(msg_type)

    message_encoding = None
    if issubclass(msg_type, ProtobufMessage):
        message_encoding = "protobuf"
    elif issubclass(msg_type, (dict, list)):
        message_encoding = "json"
    else:
        raise ValueError(
            f"Unsupported message type {msg_type} for channel {topic}. "
            "Only protobuf messages and dicts are supported."
        )

    return Channel(
        topic=topic,
        schema=schema,
        context=context,
        message_encoding=message_encoding,
        metadata=metadata,
    )


def create_channels_from_examples(
    msgs: dict[str, Any],
    context: Context | None = None,
) -> dict[str, Channel]:
    channels = {}
    for topic, msg in msgs.items():
        channels[topic] = create_channel(topic, type(msg), context=context)
    return channels
