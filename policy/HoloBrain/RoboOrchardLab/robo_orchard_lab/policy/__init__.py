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

"""Policy components and utilities.

A policy is a function or a model that maps observations from the
environment to actions. It can be deterministic or stochastic, and it is
typically used in reinforcement learning to decide what action to take
based on the current state of the environment.

Policy is a specialized form of inference pipeline, which includes
additional components for interacting with the environment, such as
action sampling and state management.

"""

from .base import *
from .evaluator.base import *
from .evaluator.remote import *
