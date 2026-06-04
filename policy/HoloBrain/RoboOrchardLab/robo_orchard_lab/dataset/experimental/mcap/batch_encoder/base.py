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

from robo_orchard_lab.dataset.experimental.mcap.messages import (
    McapMessagesTuple,
    StampedMessage,
)
from robo_orchard_lab.dataset.experimental.mcap.msg_encoder import (
    McapEncoderContext,
)

T = TypeVar("T", bound=Any)  # The type before decoder


class McapBatchEncoder(
    ClassInitFromConfigMixin, Generic[T], metaclass=ABCMeta
):
    """The base class for encoding datatypes to a batch of messages.

    Tempalate Args:
        T: The type of the source data before encoding.

    """

    @abstractmethod
    def format_batch(self, data: T) -> dict[str, list[StampedMessage[Any]]]:
        """Format the batch of decoded messages to target format.

        TODO: Add log_time as parameter to this function to allow customized
        log_time for each message.

        Args:
            data (T): The source batch of messages to format.

        Returns:
            dict[str, list[StampedMessage[Any]]]: The formatted batch of
                messages. It is a dictionary mapping from topic name to
                list of messages.
        """
        raise NotImplementedError()

    def __call__(
        self, src: T, msg_encoder_ctx: McapEncoderContext
    ) -> dict[str, McapMessagesTuple]:
        """Encode the source data to a batch of encoded mcap messages.

        TODO: Add log_time as parameter to this function to allow customized
        log_time for each message.

        Args:
            src (T): The source data to encode.
            msg_encoder_ctx (McapEncoderContext): The encoder context to use
                for encoding the messages.

        Returns:
            dict[str, McapMessagesTuple]: The encoded messages. It is a
                dictionary mapping from topic name to a tuple of schema,
                channel, and list of encoded messages.

        """
        formated_batch = self.format_batch(src)
        ret: dict[str, McapMessagesTuple] = {}
        for topic, msgs in formated_batch.items():
            if len(msgs) == 0:
                continue
            encoded_msg = msg_encoder_ctx.encode_message(
                topic=topic,
                msg=msgs[0].data,
                log_time=msgs[0].log_time,
                pub_time=msgs[0].pub_time,
            )
            ret[topic] = McapMessagesTuple(
                schema=encoded_msg.schema,
                channel=encoded_msg.channel,
                messages=[encoded_msg.message],  # type: ignore
            )
            for msg in msgs[1:]:
                encoded_msg = msg_encoder_ctx.encode_message(
                    topic=topic,
                    msg=msg.data,
                    log_time=msg.log_time,
                    pub_time=msg.pub_time,
                )
                ret[topic].messages.append(encoded_msg.message)  # type: ignore
        return ret


McapBatchEncoderType_co = TypeVar(
    "McapBatchEncoderType_co",
    bound=McapBatchEncoder,
    covariant=True,
)


class McapBatchEncoderConfig(ClassConfig[McapBatchEncoderType_co]):
    """Configuration class for message batch decoder."""

    pass


class McapBatchEncoders(McapBatchEncoder[Mapping[str, Any]]):
    """A collection of message batch decoders."""

    encoders: Mapping[str, McapBatchEncoder[Any]]

    def __init__(
        self,
        decoders: Mapping[str, McapBatchEncoderConfig | McapBatchEncoder],
    ):
        super().__init__()
        self.encoders = {}
        for k, v in decoders.items():
            if isinstance(v, McapBatchEncoder):
                self.encoders[k] = v
            elif isinstance(v, McapBatchEncoderConfig):
                self.encoders[k] = v()
            else:
                raise TypeError(
                    f"Expected McapBatchEncoder or McapBatchEncoderConfig, "
                    f"got {type(v)}"
                )

    def format_batch(
        self, data: Mapping[str, Any], raise_if_encoder_not_found: bool = True
    ) -> dict[str, list[StampedMessage[Any]]]:
        """Format the batch of decoded messages to target format.

        The output of all encoders are merged by topic name. If multiple
        encoders produce messages for the same topic, the messages are
        concatenated in the output list.

        Args:
            data (Mapping[str, Any]): The source batch of messages to
                format. It is a dictionary mapping from encoder name to
                the input data for that encoder.
            raise_if_encoder_not_found (bool): Whether to raise an error
                if an encoder is not found for a given name in the input
                data. If set to False, the encoder will be skipped and no
                error will be raised.

        Returns:
            dict[str, list[StampedMessage[Any]]]: The formatted batch of
                messages. It is a dictionary mapping from topic name to
                list of messages.
        """
        ret: dict[str, list[StampedMessage[Any]]] = {}
        for encoder_name, data_item in data.items():
            encoder = self.encoders.get(encoder_name, None)
            if encoder is None:
                if raise_if_encoder_not_found:
                    raise ValueError(
                        f"No encoder found for name {encoder_name}. "
                        f"Available encoders: {list(self.encoders.keys())}"
                    )
                else:
                    continue
            encoded_batch = encoder.format_batch(data_item)
            for topic, msgs in encoded_batch.items():
                if topic not in ret:
                    ret[topic] = []
                ret[topic].extend(msgs)
        return ret
