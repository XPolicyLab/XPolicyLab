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

"""Transforms package root with explicit compatibility-export parity."""

from . import base as _base_transforms
from .base import (
    DictRowTransform,
    DictRowTransformConfig,
    DictTransform,
    DictTransformConfig,
    DictTransformPipeline,
    DictTransformPipelineConfig,
)
from .noise import (
    AddNoise,
    AddNoiseConfig,
    GaussianNoiseConfig,
    UniformNoiseConfig,
)
from .normalize import (
    Normalize,
    NormalizeConfig,
    NormStatistics,
    UnNormalize,
)
from .padding import PaddingList, PaddingListConfig
from .take import TakeKeys, TakeKeysConfig

_PUBLIC_TRANSFORM_EXPORTS = (
    "DictTransform",
    "DictTransformConfig",
    "DictTransformPipeline",
    "DictTransformPipelineConfig",
    "DictRowTransform",
    "DictRowTransformConfig",
    "GaussianNoiseConfig",
    "UniformNoiseConfig",
    "AddNoise",
    "AddNoiseConfig",
    "Normalize",
    "UnNormalize",
    "NormalizeConfig",
    "NormStatistics",
    "PaddingList",
    "PaddingListConfig",
    "TakeKeys",
    "TakeKeysConfig",
)

_COMPAT_TRANSFORM_EXPORTS = tuple(
    name
    for name in getattr(_base_transforms, "__all__", ())
    if name not in _PUBLIC_TRANSFORM_EXPORTS
)

__all__ = _PUBLIC_TRANSFORM_EXPORTS + _COMPAT_TRANSFORM_EXPORTS  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> object:
    """Keep legacy package-root imports working during the migration."""
    if name in _COMPAT_TRANSFORM_EXPORTS:
        return getattr(_base_transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
