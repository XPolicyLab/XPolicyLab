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
from typing import TypeVar

import torch
from robo_orchard_core.utils.config import (
    ClassType,
    Config,
    TorchTensor,
)
from typing_extensions import Self

from robo_orchard_lab.models.codec.base import Codec, CodecConfig

__all__ = [
    "NormStatistics",
    "NormalizeCodec",
    "NormalizeCodecConfig",
    "MinMaxNormalizeCodec",
    "MeanStdNormalizeCodec",
    "QuantileNormalizeCodec",
    "NormalizeCodecT_co",
]


class NormStatistics(Config):
    """Statistics for normalization."""

    mean: TorchTensor | None = None
    std: TorchTensor | None = None
    min: TorchTensor | None = None
    max: TorchTensor | None = None
    q01: TorchTensor | None = None
    q99: TorchTensor | None = None

    @property
    def device(self) -> torch.device | None:
        """Get the device of the statistics tensors."""
        first_not_none = self._find_first_not_none()
        if first_not_none is not None:
            return first_not_none.device
        return None

    @property
    def dtype(self) -> torch.dtype | None:
        """Get the dtype of the statistics tensors."""
        first_not_none = self._find_first_not_none()
        if first_not_none is not None:
            return first_not_none.dtype
        return None

    def _find_first_not_none(self) -> torch.Tensor | None:
        """Find the first non-None tensor among the statistics."""
        not_none_tensor = None
        attrs = ["mean", "std", "min", "max", "q01", "q99"]
        for attr in attrs:
            if getattr(self, attr) is not None:
                not_none_tensor = getattr(self, attr)
                break

        return not_none_tensor

    def __post_init__(self):
        # find the first non-None value:
        not_none_tensor = None
        attrs = ["mean", "std", "min", "max", "q01", "q99"]
        for attr in attrs:
            if getattr(self, attr) is not None:
                not_none_tensor = getattr(self, attr)
                break

        if not_none_tensor is None:
            raise ValueError(
                "At least one of mean, std, min, max, q01, q99 must be set."
            )
        # check if all tensors have the same shape
        for attr in attrs:
            tensor = getattr(self, attr)
            if tensor is not None and tensor.shape != not_none_tensor.shape:
                raise ValueError(
                    f"All tensors must have the same shape. "
                    f"{attr} has shape {tensor.shape}, "
                    f"but mean has shape {not_none_tensor.shape}."
                )

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> NormStatistics:
        """Move all tensors to the specified device."""
        ret_dict = {}
        for attr in [
            "mean",
            "std",
            "min",
            "max",
            "q01",
            "q99",
        ]:
            tensor: torch.Tensor | None = getattr(self, attr)
            if tensor is not None:
                # setattr(self, attr, tensor.to(device=device, dtype=dtype))
                tensor = tensor.to(device=device, dtype=dtype)
            ret_dict[attr] = tensor
        return NormStatistics(**ret_dict)


class NormalizeCodec(Codec):
    cfg: NormalizeCodecConfig

    def __init__(self, cfg: NormalizeCodecConfig):
        self.cfg = cfg

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        """Move all tensors in cfg to the specified device."""
        new_cfg = self.cfg.to(device=device, dtype=dtype)
        return type(self)(new_cfg)


NormalizeCodecT_co = TypeVar(
    "NormalizeCodecT_co", bound=NormalizeCodec, covariant=True
)


class MinMaxNormalizeCodec(NormalizeCodec):
    """Normalize the data to the range [-1, 1] using min-max normalization."""

    def __init__(
        self, cfg: NormalizeCodecConfig[MinMaxNormalizeCodec]
    ) -> None:
        super().__init__(cfg)
        if self.cfg.stats.min is None or self.cfg.stats.max is None:
            raise ValueError(
                "Min and max statistics must be provided for normalization."
            )
        self._min = self.cfg.stats.min
        self._max = self.cfg.stats.max
        self._need_clamp = (
            self.cfg.clip_min is not None or self.cfg.clip_max is not None
        )

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data using min-max normalization."""
        normalized_data = (data - self._min) / (
            self._max - self._min + self.cfg.eps
        ) * 2 - 1

        return (
            torch.clamp(
                normalized_data, min=self.cfg.clip_min, max=self.cfg.clip_max
            )
            if self._need_clamp
            else normalized_data
        )

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize the data using min-max normalization."""

        return (data + 1) / 2.0 * (
            self._max - self._min + self.cfg.eps
        ) + self._min


class MeanStdNormalizeCodec(NormalizeCodec):
    """Normalize the data using mean and standard deviation."""

    def __init__(
        self, cfg: NormalizeCodecConfig[MeanStdNormalizeCodec]
    ) -> None:
        super().__init__(cfg)
        if self.cfg.stats.mean is None or self.cfg.stats.std is None:
            raise ValueError(
                "Mean and std statistics must be provided for normalization."
            )
        self._mean = self.cfg.stats.mean
        self._std = self.cfg.stats.std
        self._need_clamp = (
            self.cfg.clip_min is not None or self.cfg.clip_max is not None
        )

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data using mean and standard deviation."""
        normalized_data = (data - self._mean) / (self._std + self.cfg.eps)
        return (
            torch.clamp(
                normalized_data, min=self.cfg.clip_min, max=self.cfg.clip_max
            )
            if self._need_clamp
            else normalized_data
        )

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize the data using mean and standard deviation."""
        return data * (self._std + self.cfg.eps) + self._mean


class QuantileNormalizeCodec(NormalizeCodec):
    """Normalize the data using quantile normalization."""

    def __init__(
        self, cfg: NormalizeCodecConfig[QuantileNormalizeCodec]
    ) -> None:
        super().__init__(cfg)
        if self.cfg.stats.q01 is None or self.cfg.stats.q99 is None:
            raise ValueError(
                "q01 and q99 statistics must be provided for normalization."
            )
        self._q01 = self.cfg.stats.q01
        self._q99 = self.cfg.stats.q99
        self._need_clamp = (
            self.cfg.clip_min is not None or self.cfg.clip_max is not None
        )

    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize the data using quantile normalization."""
        normalized_data = (data - self._q01) / (
            self._q99 - self._q01 + self.cfg.eps
        ) * 2 - 1
        return (
            torch.clamp(
                normalized_data, min=self.cfg.clip_min, max=self.cfg.clip_max
            )
            if self._need_clamp
            else normalized_data
        )

    def decode(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize the data using quantile normalization."""
        return (data + 1) / 2.0 * (
            self._q99 - self._q01 + self.cfg.eps
        ) + self._q01


class NormalizeCodecConfig(CodecConfig[NormalizeCodecT_co]):
    class_type: ClassType[NormalizeCodecT_co]

    stats: NormStatistics
    """Statistics for normalization."""

    eps: float = 1e-8

    clip_min: float | None = None
    """Minimum value to clip the normalized data."""
    clip_max: float | None = None
    """Maximum value to clip the normalized data."""

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> NormalizeCodecConfig[NormalizeCodecT_co]:
        """Move all tensors in stats to the specified device."""
        data_dict = self.__dict__.copy()
        data_dict["stats"] = self.stats.to(device=device, dtype=dtype)

        return NormalizeCodecConfig[NormalizeCodecT_co](**data_dict)
