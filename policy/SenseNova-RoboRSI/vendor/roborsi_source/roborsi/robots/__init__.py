# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Robot-specific RoboRSI extensions."""

import importlib
from typing import Any

_EXPORTS = {
    "enumerate_robots": "base",
    "get_robot_spec": "base",
    "get_toolkit": "base",
    "PromptBundle": "prompt_bundle",
    "RobotSpec": "robot_spec",
    "RunConfig": "robot_spec",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module = importlib.import_module(f"roborsi.robots.{_EXPORTS[name]}")
        return getattr(module, name)
    raise AttributeError(name)
