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

"""State capture, persistence, and recovery utilities.

The State API is the repository-owned recovery boundary for objects that need
to capture mutable runtime state, apply it to a live object, or persist it to a
structured directory. It keeps construction metadata, runtime state, and
optional tensor parameters explicit instead of relying on one opaque pickle.

It is not a general-purpose replacement for pickle or cloudpickle instance
serialization. Use it for structured recovery snapshots with explicit capture
and restore contracts, not for transparent serialization of arbitrary Python
object graphs.
"""

from __future__ import annotations
import os
import pickle
from enum import Enum
from typing import TYPE_CHECKING, Literal

import cloudpickle
from pydantic import BaseModel, ConfigDict, Field
from robo_orchard_core.utils.config import (
    ClassConfig,
    ClassType,
    Config,
    ConfigInstanceOf,
)

from robo_orchard_lab.utils.path import (
    DirectoryNotEmptyError,
    is_empty_directory,
)

if TYPE_CHECKING:
    from robo_orchard_lab.utils.state.save_profile.base import (
        StateSaveProfile,
    )

__all__ = [
    "State",
    "StateList",
    "StateSequence",
    "load",
]


META_FILE_NAME = "meta.json"
_STATE_SEQUENCE_KINDS = frozenset({"list", "tuple"})


_protocol2module = {
    "pickle": pickle,
    "cloudpickle": cloudpickle,
}


# Core State containers


class ConstructableStateApplyMode(str, Enum):
    """How apply decode should handle nested State payloads with class_type."""

    MATERIALIZE = "materialize"
    """Decode constructable nested State payloads into fresh live objects."""

    PRESERVE_STATE = "preserve_state"
    """Keep constructable nested State payloads as State for owner apply."""


class State(BaseModel):
    """Canonical payload for runtime recovery and structured persistence.

    Use ``State`` when a caller needs to move an object's runtime state across
    a live-object recovery boundary or save it to disk without hiding
    everything inside one opaque pickle. ``state`` carries mutable runtime
    data, while ``class_type``, ``config``, and ``parameters`` carry the
    reconstruction metadata used by persistence and ``state2obj(...)``.

    ``State`` is not intended to serialize arbitrary live instances the way
    pickle or cloudpickle can. It represents an explicit recovery snapshot for
    objects that define how their state should be captured and restored.

    The v1 payload model is tree-shaped. Circular references and object
    identity preservation are rejected instead of being silently preserved.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    class_type: ClassType[object] | None = None
    """Class to materialize with ``state2obj(...)``.

    The class should satisfy the State API fresh-object materialization
    contract, typically via ``StateSaveLoadMixin`` or the two-phase
    ``StateMaterializeProtocol``.
    """

    state: dict[str, object]
    """Tree-shaped runtime state payload."""

    config: ConfigInstanceOf[Config] | None = None
    """Optional construction config for the object that produced this state."""

    parameters: dict[str, object] | None = None
    """Optional static tensor parameters, such as model weights."""

    hierarchical_save: bool | None = None
    """Whether this payload should be persisted as an independent folder."""

    save_profile: str | None = None
    """Optional explicit hierarchical persistence profile.

    ``None`` means there is no explicit profile selection. Root saves use the
    highest-priority root save profile from the registry; nested saves inherit
    the effective parent profile. Loaded artifacts set this field to the
    profile that interpreted the on-disk layout, so legacy tree artifacts
    without a saved field are treated as ``"tree"``.
    """

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> None:
        """Persist this State payload into an empty directory.

        This method writes reconstruction metadata, optional tensor parameters,
        and the runtime ``state`` payload. Nested State API containers may be
        saved as child directories when ``hierarchical_save`` is enabled; other
        values are pickled with the selected protocol.

        The target directory receives ``meta.json``, optional tensor parameter
        files, and either a single ``state.pkl`` payload or child directories
        for hierarchical State API containers.

        Args:
            path (str): Empty directory path to create or use for the payload.
            protocol (Literal["pickle", "cloudpickle"], optional): Pickle
                backend for non-tensor Python values. Default is
                ``"cloudpickle"``.
        """

        _save_state_api_root(self, path=path, protocol=protocol)

    @classmethod
    def load(
        cls, path: str, protocol: Literal["pickle", "cloudpickle"]
    ) -> State:
        """Load a State payload from a directory written by ``State.save``.

        This recreates the transport payload and metadata only. It does not
        instantiate ``class_type``; use ``state2obj(...)`` for fresh-object
        materialization.

        Args:
            path (str): Directory containing ``meta.json`` and persisted State
                payload files.
            protocol (Literal["pickle", "cloudpickle"]): Pickle backend used
                for non-tensor Python values.

        Returns:
            State: Loaded State payload.
        """
        from robo_orchard_lab.utils.state.save_profile import (
            resolve_load_profile,
        )

        loaded = resolve_load_profile(path).load(path, protocol=protocol)
        if not isinstance(loaded, State):
            raise TypeError(
                "State.load(...) expected a State root. "
                f"Got {type(loaded).__name__}."
            )
        return loaded


class StateSequence(BaseModel):
    """State API transport container that preserves list/tuple fidelity.

    ``obj2state(...)`` uses this type inside ``State.state`` when it encounters
    Python ``list`` or ``tuple`` values. It is not a domain recovery payload by
    itself; domain-facing recovery should still pass a top-level ``State``.
    The encoded shape is explicit:

    - ``kind`` records whether the original value was a ``list`` or ``tuple``
    - ``items`` stores the recursively encoded sequence elements

    Example:
        ``[1, 2]`` becomes ``StateSequence(kind="list", items=[1, 2])``.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
    )

    kind: Literal["list", "tuple"]
    """The original Python sequence type."""

    items: list[object]
    """The encoded sequence items."""

    hierarchical_save: bool | None = None
    """Whether nested items should inherit hierarchical save behavior."""

    save_profile: str | None = None
    """Optional explicit profile; ``None`` inherits or uses root default."""

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> None:
        """Persist this sequence transport container into an empty directory.

        Args:
            path (str): Empty directory path to create or use for the sequence.
            protocol (Literal["pickle", "cloudpickle"], optional): Pickle
                backend for non-State items. Default is ``"cloudpickle"``.
        """
        _save_state_api_root(self, path=path, protocol=protocol)

    @staticmethod
    def load(
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
        kind: Literal["list", "tuple"] | None = None,
    ) -> StateSequence:
        """Load a persisted StateSequence without materializing its items.

        Args:
            path (str): Directory containing the persisted sequence payload.
            protocol (Literal["pickle", "cloudpickle"], optional): Pickle
                backend for non-State items. Default is ``"cloudpickle"``.
            kind (Literal["list", "tuple"] | None, optional): Optional
                legacy override for the represented Python sequence type.
                Default is ``None``, which uses the persisted metadata.

        Returns:
            StateSequence: Loaded sequence transport container.
        """
        if kind is not None and kind not in _STATE_SEQUENCE_KINDS:
            raise ValueError(f"Unsupported State sequence kind: {kind!r}.")

        from robo_orchard_lab.utils.state.save_profile import (
            resolve_load_profile,
        )

        loaded = resolve_load_profile(path).load(
            path,
            protocol=protocol,
        )
        if not isinstance(loaded, StateSequence):
            raise TypeError(
                "StateSequence.load(...) expected a StateSequence root. "
                f"Got {type(loaded).__name__}."
            )
        if kind is not None and loaded.kind != kind:
            return loaded.model_copy(update={"kind": kind})
        return loaded


class StateList(list):
    """Legacy State API list container kept for persisted payloads.

    New repository-owned runtime capture should use a normal ``list`` and let
    ``obj2state(...)`` encode it as ``StateSequence``. This class remains so
    older persisted directories can still be loaded and saved compatibly.
    Treat it as a persistence compatibility transport type, not a preferred
    runtime payload shape for new code.
    """

    hierarchical_save: bool | None = None
    """Whether to save the legacy list entries to separate paths."""

    save_profile: str | None = None
    """Optional explicit profile; ``None`` inherits or uses root default."""

    def __init__(
        self,
        *args,
        hierarchical_save: bool | None = None,
        save_profile: str | None = None,
    ) -> None:
        """Initialize the legacy list transport container.

        Args:
            *args: Initial list contents forwarded to ``list``.
            hierarchical_save (bool | None, optional): Whether legacy entries
                should be saved to child paths. Default is ``None``.
            save_profile (str | None, optional): Optional explicit persistence
                profile. ``None`` inherits or uses the root default.
        """
        super().__init__(*args)
        self.hierarchical_save = hierarchical_save
        self.save_profile = save_profile

    def copy(self) -> StateList:
        return StateList(
            [data for data in self],
            hierarchical_save=self.hierarchical_save,
            save_profile=self.save_profile,
        )

    def model_copy(self) -> StateList:
        return self.copy()

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> None:
        """Persist this legacy list container into an empty directory.

        The default tree profile writes ``meta.json`` and stores list items
        under the historical ``all`` payload name. Other save profiles may
        use a different directory layout.

        Args:
            path (str): Empty directory path to create or use for the list.
            protocol (Literal["pickle", "cloudpickle"], optional): Pickle
                backend for non-State items. Default is ``"cloudpickle"``.
        """

        _save_state_api_root(self, path=path, protocol=protocol)

    @staticmethod
    def load(
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> StateList:
        """Load a legacy StateList from a directory written by StateList.save.

        Args:
            path (str): Directory containing the legacy list payload.
            protocol (Literal["pickle", "cloudpickle"], optional): Pickle
                backend for non-State items. Default is ``"cloudpickle"``.

        Returns:
            StateList: Loaded legacy list container.
        """
        from robo_orchard_lab.utils.state.save_profile import (
            resolve_load_profile,
        )

        loaded = resolve_load_profile(path).load(
            path=path,
            protocol=protocol,
        )
        if not isinstance(loaded, StateList):
            raise TypeError(
                "StateList.load(...) expected a StateList root. "
                f"Got {type(loaded).__name__}."
            )
        return loaded


# Persistence metadata


class StateConfig(ClassConfig[object]):
    """Persistence metadata used to redispatch State API directory loading.

    ``State.save(...)`` and custom adapters write this metadata to
    ``meta.json``. Generic loading reads it to decide which classmethod should
    load the directory and which reconstruction metadata belongs to the
    recovered State payload.
    """

    class_type: ClassType[object]
    """Class whose ``load(...)`` method owns this directory format."""

    load_kwargs: dict[str, object] = Field(
        default_factory=dict,
    )
    """Keyword arguments passed to ``class_type.load(...)``."""

    state_class_type: ClassType[object] | None = None
    """Original object class represented by a persisted State payload."""

    state_class_config: ConfigInstanceOf[Config] | None = None
    """Original object config represented by a persisted State payload."""


# Public entrypoints


def load(path: str) -> object:
    """Load an object or payload from a State API directory.

    This is the generic filesystem entrypoint. Directories containing
    ``State``-family payloads are materialized with ``state2obj(...)``;
    directories containing ``CustomizedSaveLoadMixin`` adapters are returned
    through their custom loader.

    Args:
        path (str): State API directory containing ``meta.json``.

    Returns:
        object: Materialized object, decoded sequence/list payload, or object
        returned by a custom adapter loader.
    """
    from robo_orchard_lab.utils.state.conversion import state2obj

    state = _load_state_from_path(path)
    if isinstance(state, (State, StateList, StateSequence)):
        return state2obj(state)
    else:
        return state


# Private persistence helpers


def _load_state_from_path(
    path: str,
) -> object:
    """Load a State API artifact without forcing fresh materialization."""
    from robo_orchard_lab.utils.state.save_profile import load_state_artifact

    return load_state_artifact(path)


def _save_state_api_root(
    state: State | StateSequence | StateList,
    *,
    path: str,
    protocol: Literal["pickle", "cloudpickle"],
    inherited_save_profile: StateSaveProfile | None = None,
) -> None:
    """Save one State API root while honoring profile inheritance rules."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if not is_empty_directory(path):
        raise DirectoryNotEmptyError(
            f"Path {path} already exists and is not empty."
        )

    from robo_orchard_lab.utils.state.save_profile import resolve_save_profile

    resolve_save_profile(
        state.save_profile,
        inherited_profile=inherited_save_profile,
    ).save(
        state,
        path=path,
        protocol=protocol,
    )
