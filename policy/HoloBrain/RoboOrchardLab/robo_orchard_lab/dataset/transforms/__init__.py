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

"""Compatibility wrapper for the canonical transform package root.

Prefer importing new code from ``robo_orchard_lab.transforms``. This module
remains as a compatibility surface while legacy callers are phased out.
"""

import robo_orchard_lab.transforms as _canonical_transforms
from robo_orchard_lab.transforms import (
    _COMPAT_TRANSFORM_EXPORTS,
    _PUBLIC_TRANSFORM_EXPORTS,
    AddNoise,
    AddNoiseConfig,
    DictRowTransform,
    DictRowTransformConfig,
    DictTransform,
    DictTransformConfig,
    DictTransformPipeline,
    DictTransformPipelineConfig,
    GaussianNoiseConfig,
    Normalize,
    NormalizeConfig,
    NormStatistics,
    PaddingList,
    PaddingListConfig,
    TakeKeys,
    TakeKeysConfig,
    UniformNoiseConfig,
    UnNormalize,
)
from robo_orchard_lab.utils.deprecation import warn_deprecated_package

__all__ = _PUBLIC_TRANSFORM_EXPORTS + _COMPAT_TRANSFORM_EXPORTS  # pyright: ignore[reportUnsupportedDunderAll]


def __getattr__(name: str) -> object:
    """Delegate compatibility-only imports to the canonical package root."""
    return getattr(_canonical_transforms, name)


warn_deprecated_package(
    __name__,
    "`robo_orchard_lab.dataset.transforms` is deprecated. "
    "Use `robo_orchard_lab.transforms` instead.",
)
