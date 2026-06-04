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

"""Registry for State persistence profiles.

This module owns profile discovery for ``State.save(...)`` and filesystem
loading. Callers usually use ``resolve_save_profile(...)`` for explicit or
inherited save-profile selection and ``load_state_artifact(...)`` for generic
artifact loading.
"""

from __future__ import annotations
import importlib

from robo_orchard_lab.utils.state.save_profile.base import StateSaveProfile

StateSaveProfileName = str | None

_BUILTIN_PROFILE_MODULES = (
    "robo_orchard_lab.utils.state.save_profile.graph",
    "robo_orchard_lab.utils.state.save_profile.tree",
)
_BUILTINS_LOADED = False
_SAVE_PROFILES_BY_NAME: dict[str, StateSaveProfile] = {}
_ROOT_SAVE_PROFILES: list[StateSaveProfile] = []
_LOAD_PROFILES: list[StateSaveProfile] = []


def _load_builtin_profiles() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for module_name in _BUILTIN_PROFILE_MODULES:
        importlib.import_module(module_name)
    _BUILTINS_LOADED = True


def _default_profile() -> StateSaveProfile:
    if not _ROOT_SAVE_PROFILES:
        raise RuntimeError("Default State save profile is not registered.")
    return _ROOT_SAVE_PROFILES[0]


def _root_save_priority(profile: StateSaveProfile) -> int:
    return getattr(profile, "root_save_priority", 0)


def register_save_profile(
    profile: StateSaveProfile,
    *,
    replace: bool = False,
) -> StateSaveProfile:
    """Register a State save/load profile.

    Profile modules should call this once for their singleton profile object:
    ``profile.name`` is the explicit ``State.save_profile`` value,
    ``profile.root_save_priority`` decides root saves without an explicit
    profile, and ``profile.load_priority`` decides load-time artifact
    detection.

    Args:
        profile (StateSaveProfile): Profile singleton to register.
        replace (bool, optional): Whether an existing profile with the same
            name may be replaced in-place. Default is ``False``.

    Returns:
        StateSaveProfile: The registered profile object.
    """
    if profile.name is None:
        raise ValueError("State save profile names must be explicit strings.")
    existing = _SAVE_PROFILES_BY_NAME.get(profile.name)
    if existing is profile:
        return profile
    if existing is not None and not replace:
        raise ValueError(
            f"State save profile {profile.name!r} is already registered."
        )

    if existing is not None:
        _LOAD_PROFILES.remove(existing)
        _ROOT_SAVE_PROFILES.remove(existing)
    _SAVE_PROFILES_BY_NAME[profile.name] = profile
    _ROOT_SAVE_PROFILES.append(profile)
    _ROOT_SAVE_PROFILES.sort(
        key=_root_save_priority,
        reverse=True,
    )
    _LOAD_PROFILES.append(profile)
    _LOAD_PROFILES.sort(
        key=lambda item: item.load_priority,
        reverse=True,
    )
    return profile


def resolve_save_profile(
    name: StateSaveProfileName,
    *,
    inherited_profile: StateSaveProfile | None = None,
) -> StateSaveProfile:
    """Return the State save profile selected by ``name``.

    ``None`` means "no explicit selection": a nested save inherits the
    effective parent profile, while a root save uses the highest-priority
    root save profile.

    Args:
        name (StateSaveProfileName): Explicit ``State.save_profile`` value, or
            ``None`` when the caller did not choose a profile.
        inherited_profile (StateSaveProfile | None, optional): Effective
            parent profile for nested saves. Default is ``None``.

    Returns:
        StateSaveProfile: Profile that should own the save operation.
    """
    _load_builtin_profiles()
    if name is None:
        if inherited_profile is not None:
            return inherited_profile
        return _default_profile()
    try:
        return _SAVE_PROFILES_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported State save profile: {name!r}.") from exc


def resolve_load_profile(path: str) -> StateSaveProfile:
    """Return the registered profile that should load ``path``.

    Args:
        path (str): Candidate State API directory or artifact path.

    Returns:
        StateSaveProfile: Profile whose manifest detection matches ``path``.
    """
    _load_builtin_profiles()
    for profile in _LOAD_PROFILES:
        if profile.has_manifest(path):
            return profile
    return _default_profile()


def load_state_artifact(path: str) -> object:
    """Load any State API artifact path, including profile-owned subentries.

    Args:
        path (str): State API directory or nested artifact path.

    Returns:
        object: Payload object returned by the owning profile loader.
    """
    _load_builtin_profiles()
    for profile in _LOAD_PROFILES:
        if profile.has_artifact(path):
            return profile.load_path(path)
    return _default_profile().load_path(path)


__all__ = [
    "StateSaveProfileName",
    "load_state_artifact",
    "register_save_profile",
    "resolve_load_profile",
    "resolve_save_profile",
]
