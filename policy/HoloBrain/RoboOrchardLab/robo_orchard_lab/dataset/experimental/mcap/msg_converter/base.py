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
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    TypeVar,
)

from robo_orchard_core.datatypes.adaptor import (
    ClassInitFromConfigMixin,
    TypeAdaptorImpl,
)
from robo_orchard_core.utils.config import ClassConfig

__all__ = [
    "MessageConverter",
    "MessageConverterStateless",
    "MessageConverterStateful",
    "MessageConverterConfig",
    "MessageConverterFactory",
    "MessageConverterFactoryConfig",
    "TensorTargetConfigMixin",
]

SRC_T = TypeVar("SRC_T", bound=Any)  # the type before decoder
DST_T = TypeVar("DST_T", bound=Any)  # The type after decoder


class MessageConverter(TypeAdaptorImpl[SRC_T, DST_T]):
    """The base class for message converter.

    A message converter is used to convert a message from one format to
    another. For example, a message converter that converts a protobuf
    message to a json message.

    """

    @abstractmethod
    def __call__(self, src: Optional[SRC_T]) -> Iterator[DST_T]:
        """Convert the message from one format to another."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def stateless(self) -> bool:
        """Return True if the message converter is stateless.

        A message converter is stateless if it does not depend on any
        external state. For example, a message converter that converts a
        protobuf message to a json message is stateless.

        """
        raise NotImplementedError()


class MessageConverterStateless(MessageConverter[SRC_T, DST_T]):
    """The base class for stateless message converter.

    A message converter is stateless if it does not depend on any
    external state. For example, a message converter that converts a
    protobuf message to a json message is stateless.

    """

    @abstractmethod
    def convert(self, src: SRC_T) -> DST_T:
        """Convert the message from one format to another."""
        raise NotImplementedError()

    def __call__(self, src: SRC_T) -> Iterator[DST_T]:
        yield self.convert(src)

    @property
    def stateless(self) -> bool:
        return True


class MessageConverterStateful(MessageConverter[SRC_T, DST_T]):
    """The base class for stateful message converter.

    A message converter is stateful if it depends on some external state when
    converting the message. For example, a video encoder/decoder is stateful,
    because encoding/decoding a video frame may depend on the previous/future
    frame.
    """

    def __call__(self, src: Optional[SRC_T]) -> Iterator[DST_T]:
        yield from self.convert(src)

    @abstractmethod
    def convert(self, src: Optional[SRC_T]) -> Iterator[DST_T]:
        """Convert the message from one format to another.

        The return value of this method is an iterator of the converted
        messages. It could contain zero or more messages depending on
        the state. For example, when encoding a B frame in a video, the
        encoder may need to use the future frame to encode the current frame.
        In this case, the encoder will return an empty iterator. When
        the future frame is available, more than one message can be returned
        from the iterator.

        Args:
            src: The source message to be converted. If None, the converter
                will flush the state and return the iterator of the converted
                messages in the state.

        Yields:
            The converted message.
        """
        raise NotImplementedError()

    def flush(self) -> List[DST_T]:
        """Flush the stateful message converter.

        This is used to flush the state of the message converter so that the
        stateful message converter can be used again.
        """
        raise NotImplementedError()

    def make_iterator(
        self,
        src: Iterator[SRC_T],
        append_none: bool = True,
        flush: bool = True,
    ) -> Iterator[DST_T]:
        """Make an iterator from the message converter.

        This is used to make an iterator from the message converter so that
        the message converter can be used in a for loop.

        Args:
            src: The source message to be converted. It should be an iterator
                of the source message.
            append_none: If True, the message converter will append None to
                the iterator when the source message iterator ends. This is
                used to indicate that the message converter is done with
                the source message.
            flush: If True, the message converter will flush the state after
                the source message iterator ends. This is used to refresh the
                state of the message converter so that it can be used again.

        Yields:
            The converted message.

        """
        for msg in src:
            yield from self.convert(msg)
        if append_none:
            yield from self.convert(None)
        if flush:
            yield from self.flush()

    @property
    def stateless(self) -> bool:
        return False


MessageConverterType_co = TypeVar(
    "MessageConverterType_co",
    bound=MessageConverter,
    covariant=True,
)


class MessageConverterConfig(ClassConfig[MessageConverterType_co]):
    """The config class for message converter.

    This class is used to create a message converter from the config.
    The config should be a dictionary that contains the class name and
    the arguments for the class.
    """

    pass


class TensorTargetConfigMixin(ClassConfig[SRC_T]):
    device: str = "cpu"
    """Device to use for target tensor, e.g., "cpu" or "cuda:0"."""

    dtype: Literal["float32", "float64"] = "float32"
    """Data type for the target tensor, e.g., "float32" or "float64"."""


class MessageConverterFactory(ClassInitFromConfigMixin):
    """The factory class for message converter."""

    def __init__(self, cfg: MessageConverterFactoryConfig):
        self._cfg = cfg

    def convert_for(
        self, schema_name: str
    ) -> Callable[[Any], Iterator[Any]] | None:
        """Get the converter implementation for the schema name.

        Args:
            schema_name: The schema name to get the decoder for.

        Returns:
            The converter impl for the schema name.
        """
        converter_cfg = self._cfg.converters.get(schema_name, None)
        if converter_cfg is None:
            return None

        converter_impl = converter_cfg()

        def convert_impl(data: Any) -> Iterator[Any]:
            yield from converter_impl.__call__(data)

        return convert_impl


class MessageConverterFactoryConfig(ClassConfig[MessageConverterFactory]):
    class_type: type[MessageConverterFactory] = MessageConverterFactory

    converters: Mapping[str, MessageConverterConfig[MessageConverter]]
