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

from importlib import import_module as _import_module
from typing import Any as _Any

from .base import *
from .benchmark import (
    BenchmarkEpisode,
    BenchmarkEpisodeRecord,
    BenchmarkEvaluator,
    BenchmarkResult,
)
from .metric_contracts import (
    EvaluatorMetrics,
    MetricUpdateTiming,
)
from .remote import *

_LAZY_PUBLIC_EXPORTS = {
    "RoboTwinBenchmarkEvaluator": (
        "robo_orchard_lab.policy.evaluator.benchmark.robotwin",
        "RoboTwinBenchmarkEvaluator",
    ),
    "RoboTwinBenchmarkEvaluatorCfg": (
        "robo_orchard_lab.policy.evaluator.benchmark.robotwin",
        "RoboTwinBenchmarkEvaluatorCfg",
    ),
}


def __getattr__(name: str) -> _Any:
    if name not in _LAZY_PUBLIC_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_PUBLIC_EXPORTS[name]
    value = getattr(_import_module(module_name), attr_name)
    globals()[name] = value
    return value
