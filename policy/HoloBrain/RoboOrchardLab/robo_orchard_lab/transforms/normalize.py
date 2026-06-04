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
from typing import Literal, TypeVar

import torch

from robo_orchard_lab.models.codec.normalize import (
    MeanStdNormalizeCodec,
    MinMaxNormalizeCodec,
    NormalizeCodec,
    NormalizeCodecConfig,
    NormStatistics,
    QuantileNormalizeCodec,
)
from robo_orchard_lab.transforms.base import (
    ClassType,
    DictTransform,
    DictTransformConfig,
)

__all__ = [
    "Normalize",
    "UnNormalize",
    "NormalizeConfig",
    "NormStatistics",
]


class Normalize(DictTransform[dict[str, torch.Tensor]]):
    """A transform to normalize the input data."""

    cfg: NormalizeConfig[Normalize]

    def __init__(self, cfg: NormalizeConfig[Normalize]):
        self._setup(cfg)

    def _setup(self, cfg: NormalizeConfig[Normalize]) -> None:
        self.cfg = cfg
        self._codec = cfg.generate_codec_cfg()()

    def _get_ignore_save_attributes(self) -> list[str]:
        return super()._get_ignore_save_attributes() + [
            "_codec",
        ]

    def _set_state(self, state) -> None:
        super()._set_state(state)
        self._setup(self.cfg)

    def transform(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        ret = {}
        for key, data in kwargs.items():
            if not isinstance(data, torch.Tensor):
                raise TypeError(
                    f"Expected TorchTensor for key '{key}', "
                    f"but got {type(data).__name__}."
                )
            ret[key] = self._codec.encode(data)

        return ret


class UnNormalize(DictTransform[dict[str, torch.Tensor]]):
    cfg: NormalizeConfig[UnNormalize]

    def __init__(self, cfg: NormalizeConfig[UnNormalize]):
        self._setup(cfg)

    def _setup(self, cfg: NormalizeConfig[UnNormalize]) -> None:
        self.cfg = cfg
        self._codec = cfg.generate_codec_cfg()()

    def _get_ignore_save_attributes(self) -> list[str]:
        return super()._get_ignore_save_attributes() + [
            "_codec",
        ]

    def _set_state(self, state) -> None:
        super()._set_state(state)
        self._setup(self.cfg)

    def transform(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        ret = {}
        for key, data in kwargs.items():
            if not isinstance(data, torch.Tensor):
                raise TypeError(
                    f"Expected TorchTensor for key '{key}', "
                    f"but got {type(data).__name__}."
                )
            ret[key] = self._codec.decode(data)

        return ret


NormT = TypeVar("NormT", bound=Normalize | UnNormalize)


class NormalizeConfig(DictTransformConfig[NormT]):
    class_type: ClassType[NormT]
    """Class type of normalization transform.

    Can be either Normalize or UnNormalize.
    """

    stats: NormStatistics
    """Statistics for normalization."""

    eps: float = 1e-8

    norm_type: Literal["mean_std", "min_max", "quantile"] = "min_max"
    """Type of normalization to apply."""

    clip_min: float | None = None
    """Minimum value to clip the normalized data."""
    clip_max: float | None = None
    """Maximum value to clip the normalized data."""

    def generate_codec_cfg(self) -> NormalizeCodecConfig[NormalizeCodec]:
        if self.norm_type == "mean_std":
            return NormalizeCodecConfig(
                class_type=MeanStdNormalizeCodec,
                stats=self.stats,
                eps=self.eps,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
        elif self.norm_type == "min_max":
            return NormalizeCodecConfig(
                class_type=MinMaxNormalizeCodec,
                stats=self.stats,
                eps=self.eps,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
        elif self.norm_type == "quantile":
            return NormalizeCodecConfig(
                class_type=QuantileNormalizeCodec,
                stats=self.stats,
                eps=self.eps,
                clip_min=self.clip_min,
                clip_max=self.clip_max,
            )
        else:
            raise ValueError(f"Unknown normalization type: {self.norm_type}")
