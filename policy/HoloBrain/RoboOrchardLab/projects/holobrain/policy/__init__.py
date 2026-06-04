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

# Lazy compatibility re-exports.  Importing the package no longer eagerly
# loads every policy submodule; individual symbols are resolved on first
# access so that unrelated runtime dependencies stay unloaded.

_COMPAT_IMPORTS: dict[str, tuple[str, str]] = {
    "HoloBrainRoboTwinPolicy": (
        ".robotwin_policy",
        "HoloBrainRoboTwinPolicy",
    ),
    "HoloBrainRoboTwinPolicyCfg": (
        ".robotwin_policy",
        "HoloBrainRoboTwinPolicyCfg",
    ),
}


def __getattr__(name: str):
    if name in _COMPAT_IMPORTS:
        module_path, attr = _COMPAT_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_COMPAT_IMPORTS.keys())
