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

"""State capture, persistence, and recovery public API."""

from typing import Any

from robo_orchard_lab.utils.state.conversion import (
    obj2state,
    state2obj,
    validate_recovery_state,
)
from robo_orchard_lab.utils.state.core import (
    ConstructableStateApplyMode,
    State,
    StateList,
    StateSequence,
    load,
)
from robo_orchard_lab.utils.state.mixin import (
    CustomizedSaveLoadMixin,
    StateMaterializeMixin,
    StatePersistenceMixin,
    StateRuntimeMixin,
    StateRuntimeProtocol,
    StateSaveLoadMixin,
)

__all__ = [
    "State",
    "ConstructableStateApplyMode",
    "StateRuntimeProtocol",
    "StateRuntimeMixin",
    "StatePersistenceMixin",
    "StateMaterializeMixin",
    "StateSaveLoadMixin",
    "CustomizedSaveLoadMixin",
    "load",
    "obj2state",
    "state2obj",
    "validate_recovery_state",
]

_COMPAT_ATTRS = frozenset(
    {
        "StateConfig",
        "WrappedHuggingFaceObj",
    }
)


def __getattr__(name: str) -> Any:
    """Resolve legacy metadata class paths without expanding public exports."""
    if name in _COMPAT_ATTRS:
        if name == "StateConfig":
            from robo_orchard_lab.utils.state.core import StateConfig

            value = StateConfig
        else:
            from robo_orchard_lab.utils.state.mixin import (
                WrappedHuggingFaceObj,
            )

            value = WrappedHuggingFaceObj
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
