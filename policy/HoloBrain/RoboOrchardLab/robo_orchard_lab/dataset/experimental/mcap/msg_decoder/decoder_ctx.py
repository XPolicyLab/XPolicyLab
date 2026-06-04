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

from typing import Any, Callable, Iterable, Optional

from foxglove import Schema as FoxgloveSchema

from mcap.decoder import DecoderFactory as McapDecoderFactory
from mcap.exceptions import DecoderNotFoundError
from mcap.records import (
    Message as McapMessage,
    Schema,
)
from robo_orchard_lab.dataset.experimental.mcap.messages import StampedMessage
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder.factory import (
    DecoderFactoryWithConverter,
)

__all__ = ["McapDecoderContext"]


class McapDecoderContext:
    """Mcap decoder with context for managing decoders.

    This class separates the decoding logic from `mcap.reader.McapReader`,
    allowing for a more flexible  approach to decoding messages. Users can
    explicitly reset the context to clear all cached decoders.

    """

    def __init__(
        self,
        decoder_factories: Iterable[McapDecoderFactory] = (),
    ):
        cnt = 0
        for factory in decoder_factories:
            if not isinstance(factory, McapDecoderFactory):
                raise TypeError(
                    f"Expected McapDecoderFactory, got {type(factory)}"
                )
            cnt += 1

        if cnt == 0:
            decoder_factories = (DecoderFactoryWithConverter(),)

        self._decoder_factories = decoder_factories
        self._decoders: dict[int, Callable[[bytes], Any]] = {}

    def reset(self) -> None:
        """Reset the decoder context, clearing all cached decoders."""
        self._decoders.clear()

    def __enter__(self) -> "McapDecoderContext":
        """Enter the context manager, returning self."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Exit the context manager, resetting the context."""
        self.reset()
        if exc_type is not None and exc_value is not None:
            raise exc_value

    def decode_message(
        self,
        message_encoding: str,
        message: McapMessage | StampedMessage[Any],
        schema: Schema | None | FoxgloveSchema,
    ) -> Any:
        """Decode a message using the appropriate decoder.

        This method retrieves the decoder for the given message by channel_id
        and uses it to decode the message data. If no decoder is found for the
        specified channel_id, it iterates through the registered decoder
        factories to find a suitable decoder based on the message encoding
        and schema.

        Note:
            The decoder is cached in the context for each message channel. This
            allows the decoder to be stateful for cases like video decoding,
            where the decoder may maintain internal state across multiple
            messages on the same channel.

        Args:
            message_encoding (str): The encoding type of the message.
            message (McapMessage|StampedMessage[Any]): The message to decode.
            schema (Schema | None | FoxgloveSchema): The schema associated
                with the message, or None if no schema is available.

        Returns:
            Any: The decoded message data.

        """
        if isinstance(schema, FoxgloveSchema):
            raise NotImplementedError(
                "Decoding for FoxgloveSchema is not implemented yet."
            )
        if isinstance(message, StampedMessage):
            raise NotImplementedError(
                "Decoding of StampedMessage is not implemented yet. "
            )

        decoder = self._decoders.get(message.channel_id)
        if decoder is not None:
            return decoder(message.data)
        else:
            for factory in self._decoder_factories:
                decoder = factory.decoder_for(message_encoding, schema)
                if decoder is not None:
                    self._decoders[message.channel_id] = decoder
                    return decoder(message.data)

            raise DecoderNotFoundError(
                f"No decoder found for message encoding '{message_encoding}' "
                f"and schema '{schema.name if schema else 'None'}'."
            )
