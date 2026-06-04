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
from abc import ABCMeta, abstractmethod
from typing import Any, Sequence, TypeVar, overload

from pydantic import Field
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    ConfigInstanceOf,
    load_from,
)
from typing_extensions import Self

__all__ = [
    "ClassType",
    "ConfigInstanceOf",
    "Codec",
    "CodecConfig",
    "ChainCodec",
    "ChainCodecConfig",
    "CodecT_co",
    "CodecConfigT_co",
]

CodecT_co = TypeVar("CodecT_co", bound="Codec", covariant=True)


class Codec(metaclass=ABCMeta):
    """CodecMixin is an API for encoding and decoding data.

    The codec concept is suitable for `Tokenizer`, `VAE`, and other models
    that transform data into a different representation and back.

    """

    cfg: CodecConfig

    InitFromConfig: bool = True

    @abstractmethod
    def encode(self, data: Any) -> Any:
        """Encode data."""
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def decode(self, data: Any) -> Any:
        """Decode data."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __repr__(self) -> str:
        ret = f"{self.__class__.__name__}("
        ret += f"cfg={self.cfg.to_dict(exclude_defaults=True)}"
        ret += ")"
        return ret

    @classmethod
    def from_config(cls, config_path: str) -> Self:
        """Load a Codec from a configuration file."""
        cfg = load_from(config_path, ensure_type=CodecConfig)
        if not issubclass(cfg.class_type, cls):
            raise TypeError(
                f"Config class_type {cfg.class_type} is not a subclass of "
                f"{cls.__name__}."
            )
        return cfg()


class CodecConfig(ClassConfig[CodecT_co]):
    class_type: ClassType[CodecT_co]

    @classmethod
    def load_from(cls, config_path: str) -> Self:
        """Load a Codec from a configuration file."""
        return load_from(config_path, ensure_type=cls)


CodecConfigT_co = TypeVar(
    "CodecConfigT_co", bound=CodecConfig[Codec], covariant=True
)


class ChainCodec(Codec):
    """ChainCodec is a codec that applies multiple codecs in sequence."""

    cfg: ChainCodecConfig

    InitFromConfig: bool = True

    def __init__(self, cfg: ChainCodecConfig):
        self.cfg = cfg
        self._codecs = [codec_cfg() for codec_cfg in self.cfg.codecs]

    def encode(self, data: Any) -> Any:
        for codec in self._codecs:
            data = codec.encode(data)
        return data

    def decode(self, data: Any) -> Any:
        for codec in reversed(self._codecs):
            data = codec.decode(data)
        return data


class ChainCodecConfig(CodecConfig[ChainCodec]):
    class_type: ClassType[ChainCodec] = ChainCodec

    codecs: Sequence[ConfigInstanceOf[CodecConfig[Codec]]] = Field(
        min_length=1
    )

    @overload
    def __getitem__(self, item: int) -> CodecConfig[Codec]:
        pass

    @overload
    def __getitem__(self, item: slice) -> ChainCodecConfig:
        pass

    def __getitem__(
        self, item: int | slice
    ) -> CodecConfig[Codec] | ChainCodecConfig:
        """Get a codec config or a sub-chain codec config."""
        if isinstance(item, int):
            return self.codecs[item]
        if isinstance(item, slice):
            new_cfg = self.model_copy()
            new_cfg.codecs = self.codecs[item]
            return new_cfg
