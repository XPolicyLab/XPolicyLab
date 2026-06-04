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

import warnings
from typing import Any, Callable, Iterable, Mapping

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.records import Schema
from robo_orchard_lab.dataset.experimental.mcap.msg_converter import (
    MessageConverter,
    MessageConverterConfig,
    MessageConverterFactoryConfig,
)

__all__ = ["DecoderFactoryWithConverter"]


class DecoderFactoryWithConverter(McapDecoderFactory):
    """Decoder factory that can convert messages to different formats.

    This class wraps an existing decoder factory and allows for
    message conversion using the provided converters. It is used to
    create decoders that can handle different message formats and
    convert them to a common format for further processing.


    Args:
        decoder_factories (Iterable[McapDecoderFactory] | None, optional):
            A list of decoder factories to use for decoding messages.
            The first factory that can decode the message will be used.
            If None, the default factories for protobuf and ROS2 messages
            will be used if available. Defaults to None.

        converters (Mapping[str, MessageConverterConfig[MessageConverter]] | None):
            A mapping of schema names to message converter configurations.
            If None, no conversion will be performed.

    """  # noqa: E501

    def __init__(
        self,
        decoder_factories: Iterable[McapDecoderFactory] | None = None,
        converters: (
            Mapping[str, MessageConverterConfig[MessageConverter]] | None
        ) = None,
    ):
        if decoder_factories is None:
            decoder_factories = []
            try:
                from mcap_protobuf.decoder import (
                    DecoderFactory as McapProtoDecoderFactory,
                )

                decoder_factories.append(McapProtoDecoderFactory())
            except ImportError:
                warning_msg = (
                    "mcap-protobuf-support is not installed. "
                    "McapDecoderFactory for protobuf messages "
                    "will not be available."
                )
                warnings.warn(warning_msg, ImportWarning)

            try:
                from mcap_ros2.decoder import (
                    DecoderFactory as McapRos2DecoderFactory,
                )

                decoder_factories.append(McapRos2DecoderFactory())
            except ImportError:
                warning_msg = (
                    "mcap-ros2-support is not installed. "
                    "McapRos2DecoderFactory for ROS2 messages "
                    "will not be available."
                )
                warnings.warn(warning_msg, ImportWarning)

        self.decoder_factories = decoder_factories
        # self.converters = converters or {}
        self.converter_factory = MessageConverterFactoryConfig(
            converters=(converters or {}),
        )()

    def decoder_for(
        self, message_encoding: str, schema: Schema | None
    ) -> Callable[[bytes], Any] | None:
        old_decoder = None
        for decoder_factory in self.decoder_factories:
            if old_decoder := decoder_factory.decoder_for(
                message_encoding, schema
            ):
                break

        if old_decoder is None:
            return None

        assert schema is not None
        converter_impl = self.converter_factory.convert_for(schema.name)
        if converter_impl is None:
            return old_decoder

        def new_decoder(data: bytes) -> Any:
            msg = old_decoder(data)
            new_msgs = list(converter_impl(msg))
            if len(new_msgs) != 1:
                raise ValueError(
                    f"Converter {converter_impl} for schema {schema.name} "
                    f"must return exactly one message, but got {len(new_msgs)}"
                )
            return new_msgs[0]

        return new_decoder
