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

"""Base contract for State persistence profiles."""

from typing import Any, Literal, Protocol

from robo_orchard_lab.utils.state.core import ConstructableStateApplyMode


class StateSaveProfile(Protocol):
    """Persistence profile used by State save and load entrypoints."""

    @property
    def name(self) -> str:
        """Explicit ``State.save_profile`` name."""
        ...

    @property
    def root_save_priority(self) -> int:
        """Higher priority profiles win root saves without explicit profile."""
        ...

    @property
    def load_priority(self) -> int:
        """Higher priority profiles are checked first during load."""
        ...

    @property
    def preserve_identity_during_apply(self) -> bool:
        """Whether nested payload apply decode should preserve identity."""
        ...

    def save(
        self,
        state: Any,
        *,
        path: str,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> None:
        """Save ``state`` into ``path``."""
        ...

    def load(
        self,
        path: str,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> Any:
        """Load a State API payload from ``path``."""
        ...

    def has_manifest(self, path: str) -> bool:
        """Return whether ``path`` has this profile's root manifest."""
        ...

    def has_artifact(self, path: str) -> bool:
        """Return whether ``path`` can be loaded by this profile."""
        ...

    def load_path(self, path: str) -> Any:
        """Load a generic State API path with profile-owned defaults."""
        ...

    def decode_payload_for_apply(
        self,
        obj: Any,
        *,
        active_paths: dict[int, str],
        path: str,
        constructable_state_apply_mode: ConstructableStateApplyMode,
    ) -> Any:
        """Decode a payload for live ``load_state(...)`` application."""
        ...
