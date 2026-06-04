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

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Any, Generic, Mapping

from robo_orchard_core.datatypes.adaptor import (
    ClassInitFromConfigMixin,
)
from robo_orchard_core.utils.config import ClassConfig
from typing_extensions import TypeVar

from robo_orchard_lab.dataset.experimental.mcap.data_record import (
    McapMessageBatch,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_decoder import (
    McapDecoderContext,
)

__all__ = [
    "McapBatchDecoder",
    "McapBatchDecoderConfig",
    "McapBatchDecoders",
]

DST_T = TypeVar("DST_T", bound=Any)  # The type after decoder


class McapBatchDecoder(
    ClassInitFromConfigMixin, Generic[DST_T], metaclass=ABCMeta
):
    """The base class for message batch decoder.

    Different from a message decoder that operates on a single message,
    a message batch decoder is used to decode a batch of messages to a
    specific target format. It usually takes multiple messages from different
    channels, groups them and converts them to a single output format.


    User should implement the `require_topics` and `format_batch`
    methods to specify which topics are required by this decoder and how
    to format the batch of messages as output.

    Template parameters:
        DST_T: The type after decoder, which is the target format of the
            decoded messages.

    """

    @abstractmethod
    def require_topics(self) -> set[str]:
        """Return the set of topics required by this decoder.

        This method is used to determine which topics are required by this
        decoder. It is used to filter out messages that are not relevant to
        this decoder.

        Returns:
            set[str]: The set of topics required by this decoder.

        """
        raise NotImplementedError()

    @abstractmethod
    def format_batch(self, decoded_msgs: dict[str, list]) -> DST_T:
        """Format the batch of decoded messages to target format.

        Args:
            src (McapMessageBatch): The source batch of messages to format.

        Returns:
            McapMessageBatch: The formatted batch of messages.
        """
        raise NotImplementedError()

    def __call__(
        self, src: McapMessageBatch, msg_decoder_ctx: McapDecoderContext
    ) -> DST_T:
        """Decode the batch of messages from one format to another.

        Args:
            src (McapMessageBatch): The source batch of messages to decode.
            msg_decoder_ctx (McapDecoderContext): The decoder context for
                decoding each message.
        """
        cached_decoded_msgs = {}
        for required_topic in self.require_topics():
            if (
                required_topic not in cached_decoded_msgs
                and required_topic in src.message_dict
            ):
                cached_decoded_msgs[required_topic] = src.message_dict[
                    required_topic
                ].decode(msg_decoder_ctx)

        return self.format_batch(cached_decoded_msgs)


McapBatchDecoderType_co = TypeVar(
    "McapBatchDecoderType_co",
    bound=McapBatchDecoder,
    covariant=True,
)


class McapBatchDecoderConfig(ClassConfig[McapBatchDecoderType_co]):
    """Configuration class for message batch decoder."""

    pass


class McapBatchDecoders(McapBatchDecoder[dict[str, Any]]):
    """A collection of message batch decoders."""

    decoders: Mapping[str, McapBatchDecoder]

    def __init__(
        self,
        decoders: Mapping[str, McapBatchDecoderConfig | McapBatchDecoder],
    ):
        super().__init__()
        self.decoders = {}
        for k, v in decoders.items():
            if isinstance(v, McapBatchDecoder):
                self.decoders[k] = v
            elif isinstance(v, McapBatchDecoderConfig):
                self.decoders[k] = v()
            else:
                raise TypeError(
                    f"Expected MsgBatchDecoder or MsgBatchDecoderConfig, "
                    f"got {type(v)}"
                )

    def require_topics(self) -> set[str]:
        """Return the set of topics required by all decoders in this batch.

        This method aggregates the required topics from all decoders in the
        batch.

        Returns:
            set[str]: The set of topics required by all decoders.
        """
        required_topics = set()
        for decoder in self.decoders.values():
            required_topics.update(decoder.require_topics())
        return required_topics

    def format_batch(self, decoded_msgs: dict[str, list]) -> dict[str, Any]:
        """Format the batch of decoded messages to target format.

        Args:
            src (McapMessageBatch): The source batch of messages to format.

        Returns:
            McapMessageBatch: The formatted batch of messages.
        """
        ret = {}
        for name, decoder in self.decoders.items():
            ret[name] = decoder.format_batch(decoded_msgs)
        return ret
