# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
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
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .env import (
        LiberoEnv,
        LiberoEnvCfg,
        LiberoEnvStepReturn,
        LiberoEvalEnv,
        LiberoEvalEnvCfg,
        LiberoSuiteName,
        get_libero_task,
    )

_ENV_EXPORTS = (
    "get_libero_task",
    "LiberoEnvStepReturn",
    "LiberoEnv",
    "LiberoEnvCfg",
    "LiberoEvalEnv",
    "LiberoEvalEnvCfg",
    "LiberoSuiteName",
)

__all__ = (
    "get_libero_task",
    "LiberoEnvStepReturn",
    "LiberoEnv",
    "LiberoEnvCfg",
    "LiberoEvalEnv",
    "LiberoEvalEnvCfg",
    "LiberoSuiteName",
)


def __getattr__(name: str) -> Any:
    if name in _ENV_EXPORTS:
        module = importlib.import_module(f"{__name__}.env")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
