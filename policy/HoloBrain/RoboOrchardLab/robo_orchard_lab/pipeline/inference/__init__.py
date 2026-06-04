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

"""Inference pipeline orchestration APIs.

This package hosts the canonical end-to-end inference pipeline abstractions
for RoboOrchardLab. It focuses on the data processing and model inference
stages of runtime execution and is intended for scenarios where a model needs
to be invoked outside the training loop.

The historical import surface under ``robo_orchard_lab.inference`` is kept as
deprecated compatibility facades, while new code should import the concrete
pipeline and mixin types from this package.
"""

from robo_orchard_lab.processing import io_processor as processor

from .basic import *
from .mixin import *
