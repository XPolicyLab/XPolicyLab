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

"""Deprecated compatibility module for training hook orchestration.

Use ``robo_orchard_lab.pipeline.training.hook_based_trainer`` instead.
"""

from robo_orchard_lab.pipeline.training.hook_based_trainer import (
    GradientClippingHookConfig,
    HookBasedTrainer,
    PipelineHookOrConfigType,
    ResumeCheckpointConfig,
    ValidationHookConfig,
)
from robo_orchard_lab.utils.deprecation import warn_deprecated_package

__all__ = [
    "HookBasedTrainer",
    "ResumeCheckpointConfig",
    "GradientClippingHookConfig",
    "ValidationHookConfig",
    "PipelineHookOrConfigType",
]

warn_deprecated_package(
    __name__,
    "`robo_orchard_lab.pipeline.hook_based_trainer` is deprecated. "
    "Use `robo_orchard_lab.pipeline.training.hook_based_trainer` instead.",
)
