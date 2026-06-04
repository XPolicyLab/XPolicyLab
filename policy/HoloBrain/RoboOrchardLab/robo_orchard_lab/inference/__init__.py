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

"""Deprecated compatibility access to inference pipeline components.

Historically this package hosted the data processing and model inference parts
of the runtime stack. The canonical implementations now live under
``robo_orchard_lab.pipeline.inference`` and
``robo_orchard_lab.processing.io_processor``, while this package preserves the
legacy import surface during the migration window.
"""

from __future__ import annotations
import importlib
from typing import TYPE_CHECKING, Any

from robo_orchard_lab.utils.deprecation import warn_deprecated_package

from .basic import InferencePipeline, InferencePipelineCfg
from .mixin import (
    ClassType_co,
    InferencePipelineMixin,
    InferencePipelineMixinCfg,
    InputType,
    OutputType,
)

__all__ = [
    "ClassType_co",
    "InputType",
    "OutputType",
    "InferencePipeline",
    "InferencePipelineCfg",
    "InferencePipelineMixin",
    "InferencePipelineMixinCfg",
    "processor",
]

if TYPE_CHECKING:
    from . import processor
warn_deprecated_package(
    __name__,
    "`robo_orchard_lab.inference` is deprecated. "
    "Use `robo_orchard_lab.pipeline.inference` and "
    "`robo_orchard_lab.processing.io_processor` instead.",
)


def __getattr__(name: str) -> Any:
    """Lazily resolve deprecated attributes from the legacy package.

    Args:
        name (str): Attribute requested from the deprecated package.

    Returns:
        Any: Lazily imported legacy attribute.

    Raises:
        AttributeError: If ``name`` is not a supported legacy attribute.
    """
    if name == "processor":
        return importlib.import_module(f"{__name__}.processor")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
