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

"""Internal graph-aware hierarchical persistence for the State API.

This module is intentionally private. The public contract stays on
``robo_orchard_lab.utils.state`` through ``State.save(...)``,
``State.load(...)``, ``load(...)``, and ``State.save_profile``.

Graph artifact contract:

- A graph-aware root still writes State API ``meta.json``. This keeps generic
  ``load(...)`` dispatch compatible with the regular State API.
- ``graph_manifest.json`` records the graph profile, format version, root
  node id, and the canonical owner path for every graph node.
- Every graph appearance writes ``entry.json``. Owner entries contain payload
  files; ref entries point back to the owner and are not independently
  loadable.
- Only entries whose reachable closure is inside their directory get a local
  ``graph_manifest.json`` and may be loaded directly.
- Manifest and entry schema changes must bump
  ``GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION``.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictInt,
    StrictStr,
    ValidationError,
)

from robo_orchard_lab.utils.state.save_profile import register_save_profile

if TYPE_CHECKING:
    from robo_orchard_lab.utils.state.core import (
        ConstructableStateApplyMode,
        StateConfig,
    )

ENTRY_FILE_NAME = "entry.json"
"""Per-appearance metadata file written in every graph node directory."""

GRAPH_MANIFEST_FILE_NAME = "graph_manifest.json"
"""Graph registry file written at graph roots and self-contained subroots."""

GRAPH_PROFILE = "graph"
"""Explicit ``State.save_profile`` value for this layout."""

GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION = 1
"""Current on-disk schema version for graph-aware State artifacts."""

GRAPH_AWARE_HIERARCHICAL_SUPPORTED_FORMAT_VERSIONS = frozenset(
    {GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION}
)
"""On-disk schema versions this implementation can read."""

__all__ = [
    "GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION",
    "GRAPH_AWARE_HIERARCHICAL_SUPPORTED_FORMAT_VERSIONS",
    "GRAPH_PROFILE",
    "GraphStateCorruptionError",
    "GraphStateUnsupportedFormatVersionError",
    "NonSelfContainedEntryError",
    "GRAPH_AWARE_STATE_PROFILE",
]


class GraphNodeType(str, Enum):
    """Logical type of a graph node."""

    # ``State`` container. Metadata is in ``state_meta.pkl`` and runtime
    # payload is the child directory ``state/``.
    STATE = "state"

    # ``StateSequence`` container. Metadata is in ``sequence_meta.pkl`` and
    # elements live under ``items/``.
    STATE_SEQUENCE = "state_sequence"

    # ``StateList`` container. Elements live under ``items/``.
    STATE_LIST = "state_list"

    # Plain dictionary container. Children use dict keys as path segments.
    DICT = "dict"

    # Plain list container. Children live under numeric ``items/`` segments.
    LIST = "list"

    # Plain tuple container. Children live under numeric ``items/`` segments.
    TUPLE = "tuple"

    # ``CustomizedSaveLoadMixin`` leaf that owns its directory.
    CUSTOM_LEAF = "custom_leaf"

    # Arbitrary leaf value stored in ``value.pkl``.
    OPAQUE_LEAF = "opaque_leaf"


class GraphStorageKind(str, Enum):
    """Storage family used to decide how owner payload files are written."""

    # Container payload: children are represented as graph nodes.
    GRAPH_CONTAINER = "graph_container"

    # Custom save/load leaf: traversal stops and object-owned save logic runs.
    CUSTOM_LEAF = "custom_leaf"

    # Opaque pickle leaf: traversal stops and the value is pickled directly.
    OPAQUE_LEAF = "opaque_leaf"


class GraphLoadScope(str, Enum):
    """Whether an owner entry can be loaded directly from its directory."""

    # Reachable closure is inside this directory; direct load is allowed.
    SELF_CONTAINED = "self_contained"

    # Reachable closure escapes this directory; load requires an outer graph.
    NON_SELF_CONTAINED = "non_self_contained"


class GraphEntryKind(str, Enum):
    """Whether an ``entry.json`` is an owner or reference alias."""

    # Canonical appearance. This directory owns the node payload files.
    OWNER = "owner"

    # Alias appearance. This directory points back to the owner.
    REF = "ref"


class GraphRecord(BaseModel):
    """Base Pydantic model for lightweight graph artifact records."""

    model_config = ConfigDict(frozen=True)


GraphRecordT = TypeVar("GraphRecordT", bound=GraphRecord)


def _validate_graph_record(
    record_type: type[GraphRecordT],
    payload: Any,
    *,
    context: str,
) -> GraphRecordT:
    try:
        return record_type.model_validate(payload)
    except ValidationError as exc:
        raise GraphStateCorruptionError(f"{context} is invalid.") from exc


class GraphEntryRecord(GraphRecord):
    """Parsed ``entry.json`` contract for one graph appearance."""

    kind: GraphEntryKind
    """``owner`` stores payload files; ``ref`` points back to the owner."""

    node_id: StrictStr
    """Stable node id in the containing ``graph_manifest.json``."""

    node_type: GraphNodeType
    """Logical node type; must match the manifest node record."""

    storage_kind: GraphStorageKind
    """Payload storage family; must match the manifest node record."""

    owner_relpath: StrictStr
    """Relative path from this appearance directory to the owner directory."""

    load_scope: GraphLoadScope
    """Whether this appearance can be loaded as a self-contained graph."""


class GraphManifestNodeRecord(GraphRecord):
    """Per-node registry entry inside ``graph_manifest.json``."""

    owner_path: StrictStr
    """Path from manifest root to the canonical owner directory, or ``.``."""

    node_type: GraphNodeType
    """Logical node type used by the loader to choose payload decoding."""

    storage_kind: GraphStorageKind
    """Storage family written by the saver and validated against ``entry``."""

    load_scope: GraphLoadScope
    """Whether this node's owner directory has a complete reachable closure."""


class GraphManifestHeaderRecord(GraphRecord):
    """Manifest header fields needed before version-specific parsing."""

    profile: StrictStr
    """Profile name; currently ``graph``."""

    format_version: StrictInt
    """On-disk schema version interpreted by a version-specific parser."""


class GraphManifestRecord(GraphManifestHeaderRecord):
    """Parsed ``graph_manifest.json`` contract for one graph root."""

    root_node_id: StrictStr
    """Node id to materialize when loading this manifest root."""

    nodes: dict[StrictStr, GraphManifestNodeRecord]
    """All nodes reachable from this manifest root, keyed by node id."""


class GraphChildMetaRecord(GraphRecord):
    """Pickled metadata that maps a container edge to a child directory."""

    segment: StrictStr
    """Filesystem path segment from owner directory to child entry."""

    key: Any
    """Original dict key or sequence index represented by this edge."""


@dataclass(frozen=True, slots=True)
class GraphAwareStateProfile:
    """Adapter surface used by ``state.py`` for this persistence profile.

    ``state.py`` should treat this object as the only callable surface for the
    graph-aware hierarchical profile. The details below are the private
    artifact contract this adapter owns:

    - The root directory contains normal State API ``meta.json`` plus
      ``graph_manifest.json``.
    - Every graph appearance is represented by a directory with ``entry.json``.
      Owner entries store payload files; reference entries only point at the
      owner node.
    - Container node payloads use child directories plus metadata pickles
      such as ``children_meta.pkl``, ``state_meta.pkl``, or
      ``sequence_meta.pkl``.
    - Opaque leaves use ``value.pkl``. Custom save/load leaves delegate to the
      object's own ``save(...)`` method and therefore keep their own State API
      directory format.
    - Direct loading is allowed only from graph roots or self-contained owner
      entries that include a local ``graph_manifest.json``.
    - ``format_version`` is the on-disk schema version. This adapter writes
      ``GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION`` and rejects manifests whose
      version is not in
      ``GRAPH_AWARE_HIERARCHICAL_SUPPORTED_FORMAT_VERSIONS``.
    """

    name: str
    """Profile name stored in ``State.save_profile``."""

    root_save_priority: int
    """Root-save priority when no explicit profile is selected."""

    format_version: int
    """On-disk schema version written by this implementation."""

    supported_format_versions: frozenset[int]
    """Manifest versions this implementation can read."""

    load_priority: int
    """Load dispatch priority; graph manifests must win over ``meta.json``."""

    preserve_identity_during_apply: bool
    """Whether nested apply decode keeps shared identity and cycles."""

    def save(
        self,
        state: Any,
        *,
        path: str,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> None:
        """Save ``state`` using the graph-aware hierarchical layout.

        The caller must provide an empty target directory, matching
        ``State.save(...)``'s precondition. This method writes:

        - root ``meta.json`` for normal State API dispatch;
        - root ``graph_manifest.json`` with node registry, format version, and
          root node id;
        - one ``entry.json`` per graph appearance, marking it as ``owner`` or
          ``ref``;
        - owner payload files for each node, such as ``state_meta.pkl``,
          ``sequence_meta.pkl``, ``children_meta.pkl``, ``value.pkl``, or a
          delegated custom save directory;
        - local ``graph_manifest.json`` files for self-contained owner entries
          that may be loaded independently.
        """
        _save_graph_aware_state(state, path=path, protocol=protocol)

    def load(
        self,
        path: str,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> Any:
        """Load a graph-aware artifact from ``path``.

        ``path`` may be the graph root or a self-contained owner entry with a
        local ``graph_manifest.json``. Loading a reference entry, or an owner
        whose reachable closure depends on nodes outside its directory, raises
        ``NonSelfContainedEntryError``. Corrupt or unsupported manifests raise
        ``GraphStateCorruptionError``.
        """
        return _load_graph_aware_state(path, protocol=protocol)

    def decode_payload_for_apply(
        self,
        obj: Any,
        *,
        active_paths: dict[int, str],
        path: str,
        constructable_state_apply_mode: "ConstructableStateApplyMode",
    ) -> Any:
        """Decode graph-loaded payloads for live ``load_state(...)`` paths.

        Unlike normal State payload decoding, this path preserves supported
        object identity and cycles by threading profile-owned memo state
        through recursive containers. It decodes nested State API
        containers with owner-selected handling for constructable nested
        ``State`` payloads, so the resulting payload can be passed to a live
        object's ``_set_state(...)`` method.
        """
        from robo_orchard_lab.utils.state.conversion import (
            _decode_state_payload,
        )

        del active_paths
        return _decode_state_payload(
            obj,
            path=path,
            materialize_nested_state=False,
            preserve_identity=True,
            constructable_state_apply_mode=constructable_state_apply_mode,
        )

    def has_artifact(self, path: str) -> bool:
        """Return whether ``path`` looks like any graph-aware graph entry.

        This is used by generic path loading before ``meta.json`` exists at a
        sub-entry path. It intentionally accepts either ``graph_manifest.json``
        or ``entry.json``; the subsequent ``load(...)`` call performs the full
        self-contained/corruption checks.
        """
        return _path_has_graph_manifest(path) or _path_has_graph_entry(path)

    def has_manifest(self, path: str) -> bool:
        """Return whether ``path`` has a graph-aware manifest.

        ``State.load(...)`` uses this stricter check at normal State roots,
        where ``meta.json`` is present and profile dispatch must be based on
        the root graph manifest rather than an arbitrary entry marker.
        """
        return _path_has_graph_manifest(path)

    def load_path(self, path: str) -> Any:
        """Load a graph-aware artifact path with the generic path protocol."""
        return self.load(path, protocol="cloudpickle")


GRAPH_AWARE_STATE_PROFILE = GraphAwareStateProfile(
    name=GRAPH_PROFILE,
    root_save_priority=0,
    format_version=GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION,
    supported_format_versions=GRAPH_AWARE_HIERARCHICAL_SUPPORTED_FORMAT_VERSIONS,
    load_priority=100,
    preserve_identity_during_apply=True,
)


class NonSelfContainedEntryError(ValueError):
    """Raised when a graph-aware entry cannot be loaded by itself."""


class GraphStateCorruptionError(ValueError):
    """Raised when a graph-aware State artifact is internally inconsistent."""


class GraphStateUnsupportedFormatVersionError(GraphStateCorruptionError):
    """Raised when a graph-aware State artifact uses an unsupported version."""


@dataclass(slots=True)
class _GraphChild:
    """In-memory edge from a graph container to a child graph node."""

    segment: str
    """Filesystem segment used for the child appearance path."""

    key: Any
    """Original container key or sequence index."""

    node_id: str
    """Target graph node id."""


@dataclass(frozen=True, slots=True)
class _GraphChildSpec:
    """Child edge requested by a graph node codec during graph building."""

    segment: str
    """Filesystem segment used for the child appearance path."""

    key: Any
    """Original container key or sequence index."""

    value: Any
    """Original child value to add to the graph."""

    path_parts: tuple[str, ...]
    """Child appearance path parts relative to the graph root."""

    inherited_hierarchical_save: bool | None = None
    """Hierarchical-save value inherited by this child, if any."""


@dataclass(frozen=True, slots=True)
class _GraphSaveContext:
    """Shared save context passed to node codecs."""

    owner_path: str
    """Absolute directory path for the node's canonical owner entry."""

    protocol: Literal["pickle", "cloudpickle"]
    """Pickle backend selected by the caller."""


@dataclass(slots=True)
class _GraphLoadContext:
    """Shared load context passed to node codecs."""

    root: str
    """Graph artifact root directory."""

    owner_path: str
    """Absolute directory path for the node's canonical owner entry."""

    node_id: str
    """Node id being loaded."""

    manifest: GraphManifestRecord
    """Manifest that owns the reachable node registry."""

    memo: dict[str, Any]
    """Materialized nodes keyed by node id, used to preserve cycles."""

    protocol: Literal["pickle", "cloudpickle"]
    """Pickle backend selected by the caller."""


class _GraphNodeCodec:
    """Codec that owns one graph node type's build, save, and load policy."""

    node_type: GraphNodeType
    storage_kind: GraphStorageKind

    def matches(self, obj: Any) -> bool:
        """Return whether this codec should own ``obj``."""
        raise NotImplementedError

    def iter_children(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        effective_hierarchical_save: bool | None,
    ) -> tuple[_GraphChildSpec, ...]:
        """Return graph children owned by ``obj``."""
        return ()

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        """Write owner payload files for ``node``."""
        raise NotImplementedError

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        """Load the node payload and update ``ctx.memo`` when needed."""
        raise NotImplementedError

    def build_root_state_config(
        self,
        obj: Any,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> StateConfig | None:
        """Return the root ``meta.json`` contract for supported root types."""
        del obj, protocol
        return None


def _build_graph_root_state_config(
    *,
    root_class: type[Any],
    protocol: Literal["pickle", "cloudpickle"],
    kind: str | None = None,
    state_class_type: Any = None,
    state_class_config: Any = None,
) -> StateConfig:
    from robo_orchard_lab.utils.state.core import StateConfig

    load_kwargs = {"protocol": protocol}
    if kind is not None:
        load_kwargs["kind"] = kind
    return StateConfig(
        class_type=root_class,
        load_kwargs=load_kwargs,
        state_class_type=state_class_type,
        state_class_config=state_class_config,
    )


def _indexed_child_specs(
    values: Any,
    path_parts: tuple[str, ...],
    effective_hierarchical_save: bool | None = True,
) -> tuple[_GraphChildSpec, ...]:
    if effective_hierarchical_save is not True:
        return ()
    specs: list[_GraphChildSpec] = []
    for idx, value in enumerate(values):
        segment = str(idx)
        specs.append(
            _GraphChildSpec(
                segment=os.path.join("items", segment),
                key=idx,
                value=value,
                path_parts=path_parts + ("items", segment),
                inherited_hierarchical_save=True,
            )
        )
    return tuple(specs)


def _effective_hierarchical_save(
    obj: Any,
    inherited_hierarchical_save: bool | None,
) -> bool | None:
    own_value = getattr(obj, "hierarchical_save", None)
    if own_value is None and inherited_hierarchical_save is True:
        return True
    if own_value in {True, False}:
        return own_value
    return None


def _should_split_state_item(value: Any) -> bool:
    from robo_orchard_lab.utils.state.core import (
        State,
        StateList,
        StateSequence,
    )
    from robo_orchard_lab.utils.state.mixin import CustomizedSaveLoadMixin

    if isinstance(value, CustomizedSaveLoadMixin):
        return True
    if isinstance(value, (State, StateList, StateSequence)):
        return value.hierarchical_save is True
    return False


def _state_child_specs(
    state: dict[str, Any],
    path_parts: tuple[str, ...],
    effective_hierarchical_save: bool | None,
) -> tuple[_GraphChildSpec, ...]:
    split_all = effective_hierarchical_save is True
    specs: list[_GraphChildSpec] = []
    for key, value in state.items():
        if not split_all and not _should_split_state_item(value):
            continue
        segment = _graph_segment_for_dict_key(key)
        specs.append(
            _GraphChildSpec(
                segment=os.path.join("state", segment),
                key=key,
                value=value,
                path_parts=path_parts + ("state", segment),
                inherited_hierarchical_save=True if split_all else None,
            )
        )
    return tuple(specs)


class _StateNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.STATE
    storage_kind = GraphStorageKind.GRAPH_CONTAINER

    def matches(self, obj: Any) -> bool:
        from robo_orchard_lab.utils.state.core import State

        return isinstance(obj, State)

    def iter_children(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        effective_hierarchical_save: bool | None,
    ) -> tuple[_GraphChildSpec, ...]:
        return _state_child_specs(
            obj.state,
            path_parts,
            effective_hierarchical_save,
        )

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        _dump_graph_pickle(
            os.path.join(ctx.owner_path, "state_meta.pkl"),
            {
                "class_type": node.obj.class_type,
                "config": node.obj.config,
                "parameters": node.obj.parameters,
                "hierarchical_save": node.obj.hierarchical_save,
                "save_profile": (node.obj.save_profile or GRAPH_PROFILE),
            },
            protocol=ctx.protocol,
        )
        if node.children:
            _dump_graph_children_payload(
                ctx.owner_path,
                node.children,
                protocol=ctx.protocol,
            )
        split_keys = {child.key for child in node.children}
        remaining_state = {
            key: value
            for key, value in node.obj.state.items()
            if key not in split_keys
        }
        if remaining_state:
            _dump_graph_pickle(
                os.path.join(ctx.owner_path, "state.pkl"),
                remaining_state,
                protocol=ctx.protocol,
            )

    def build_root_state_config(
        self,
        obj: Any,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> StateConfig:
        from robo_orchard_lab.utils.state.core import State

        return _build_graph_root_state_config(
            root_class=State,
            protocol=protocol,
            state_class_type=obj.class_type,
            state_class_config=obj.config,
        )

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        from robo_orchard_lab.utils.state.core import State

        meta = _load_graph_meta_dict(
            ctx.owner_path,
            filename="state_meta.pkl",
            protocol=ctx.protocol,
            error_label="State metadata",
        )
        state_value = State(
            state={},
            class_type=meta.get("class_type"),
            config=meta.get("config"),
            parameters=meta.get("parameters"),
            hierarchical_save=meta.get("hierarchical_save"),
            save_profile=(meta.get("save_profile") or GRAPH_PROFILE),
        )
        ctx.memo[ctx.node_id] = state_value
        state_payload: dict[str, Any] = {}
        state_path = os.path.join(ctx.owner_path, "state.pkl")
        if os.path.exists(state_path):
            loaded_state = _load_graph_pickle(
                state_path,
                protocol=ctx.protocol,
            )
            if not isinstance(loaded_state, dict):
                raise GraphStateCorruptionError(
                    f"Invalid State payload in {state_path}."
                )
            state_payload.update(loaded_state)

        children_meta_path = os.path.join(ctx.owner_path, "children_meta.pkl")
        if os.path.exists(children_meta_path):
            state_payload.update(_iter_loaded_graph_children(ctx))
        else:
            legacy_state_path = os.path.join(ctx.owner_path, "state")
            if _path_has_graph_entry(legacy_state_path):
                state_payload.update(
                    _load_graph_child(ctx, child_path=legacy_state_path)
                )

        state_value.state = state_payload
        return state_value


class _StateSequenceNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.STATE_SEQUENCE
    storage_kind = GraphStorageKind.GRAPH_CONTAINER

    def matches(self, obj: Any) -> bool:
        from robo_orchard_lab.utils.state.core import StateSequence

        return isinstance(obj, StateSequence)

    def iter_children(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        effective_hierarchical_save: bool | None,
    ) -> tuple[_GraphChildSpec, ...]:
        return _indexed_child_specs(
            obj.items,
            path_parts,
            effective_hierarchical_save,
        )

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        _dump_graph_pickle(
            os.path.join(ctx.owner_path, "sequence_meta.pkl"),
            {
                "kind": node.obj.kind,
                "hierarchical_save": node.obj.hierarchical_save,
                "save_profile": (node.obj.save_profile or GRAPH_PROFILE),
            },
            protocol=ctx.protocol,
        )
        _dump_graph_items_or_children(
            ctx.owner_path,
            node.children,
            node.obj.items,
            protocol=ctx.protocol,
        )

    def build_root_state_config(
        self,
        obj: Any,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> StateConfig:
        from robo_orchard_lab.utils.state.core import StateSequence

        return _build_graph_root_state_config(
            root_class=StateSequence,
            protocol=protocol,
            kind=obj.kind,
        )

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        from robo_orchard_lab.utils.state.core import StateSequence

        meta = _load_graph_meta_dict(
            ctx.owner_path,
            filename="sequence_meta.pkl",
            protocol=ctx.protocol,
            error_label="StateSequence metadata",
        )
        sequence_value = StateSequence(
            kind=meta["kind"],
            items=[],
            hierarchical_save=meta.get("hierarchical_save"),
            save_profile=(meta.get("save_profile") or GRAPH_PROFILE),
        )
        ctx.memo[ctx.node_id] = sequence_value
        sequence_value.items = _load_graph_items_or_children(
            ctx,
            error_label="sequence items",
        )
        return sequence_value


class _StateListNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.STATE_LIST
    storage_kind = GraphStorageKind.GRAPH_CONTAINER

    def matches(self, obj: Any) -> bool:
        from robo_orchard_lab.utils.state.core import StateList

        return isinstance(obj, StateList)

    def iter_children(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        effective_hierarchical_save: bool | None,
    ) -> tuple[_GraphChildSpec, ...]:
        return _indexed_child_specs(
            obj,
            path_parts,
            effective_hierarchical_save,
        )

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        _dump_graph_pickle(
            os.path.join(ctx.owner_path, "list_meta.pkl"),
            {
                "hierarchical_save": node.obj.hierarchical_save,
                "save_profile": (node.obj.save_profile or GRAPH_PROFILE),
            },
            protocol=ctx.protocol,
        )
        _dump_graph_items_or_children(
            ctx.owner_path,
            node.children,
            list(node.obj),
            protocol=ctx.protocol,
        )

    def build_root_state_config(
        self,
        obj: Any,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> StateConfig:
        from robo_orchard_lab.utils.state.core import StateList

        del obj
        return _build_graph_root_state_config(
            root_class=StateList,
            protocol=protocol,
        )

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        from robo_orchard_lab.utils.state.core import StateList

        meta = _load_graph_meta_dict(
            ctx.owner_path,
            filename="list_meta.pkl",
            protocol=ctx.protocol,
            error_label="StateList metadata",
            required=False,
        )
        state_list_value = StateList(
            [],
            hierarchical_save=meta.get("hierarchical_save"),
            save_profile=(meta.get("save_profile") or GRAPH_PROFILE),
        )
        ctx.memo[ctx.node_id] = state_list_value
        state_list_value.extend(
            _load_graph_items_or_children(
                ctx,
                error_label="StateList items",
            )
        )
        return state_list_value


class _DictNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.DICT
    storage_kind = GraphStorageKind.GRAPH_CONTAINER

    def matches(self, obj: Any) -> bool:
        return isinstance(obj, dict)

    def iter_children(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        effective_hierarchical_save: bool | None,
    ) -> tuple[_GraphChildSpec, ...]:
        specs: list[_GraphChildSpec] = []
        for key, value in obj.items():
            segment = _graph_segment_for_dict_key(key)
            specs.append(
                _GraphChildSpec(
                    segment=segment,
                    key=key,
                    value=value,
                    path_parts=path_parts + (segment,),
                )
            )
        return tuple(specs)

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        _dump_graph_children_payload(
            ctx.owner_path,
            node.children,
            protocol=ctx.protocol,
        )

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        dict_value: dict[Any, Any] = {}
        ctx.memo[ctx.node_id] = dict_value
        dict_value.update(_iter_loaded_graph_children(ctx))
        return dict_value


class _ListNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.LIST
    storage_kind = GraphStorageKind.GRAPH_CONTAINER

    def matches(self, obj: Any) -> bool:
        return isinstance(obj, list)

    def iter_children(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        effective_hierarchical_save: bool | None,
    ) -> tuple[_GraphChildSpec, ...]:
        return _indexed_child_specs(obj, path_parts)

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        _dump_graph_children_payload(
            ctx.owner_path,
            node.children,
            protocol=ctx.protocol,
        )

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        list_value = []
        ctx.memo[ctx.node_id] = list_value
        list_value.extend(
            value for _, value in _iter_loaded_graph_children(ctx)
        )
        return list_value


class _TupleNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.TUPLE
    storage_kind = GraphStorageKind.GRAPH_CONTAINER

    def matches(self, obj: Any) -> bool:
        return isinstance(obj, tuple)

    def iter_children(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        effective_hierarchical_save: bool | None,
    ) -> tuple[_GraphChildSpec, ...]:
        return _indexed_child_specs(obj, path_parts)

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        _dump_graph_children_payload(
            ctx.owner_path,
            node.children,
            protocol=ctx.protocol,
        )

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        items = [value for _, value in _iter_loaded_graph_children(ctx)]
        value_tuple = tuple(items)
        ctx.memo[ctx.node_id] = value_tuple
        return value_tuple


class _CustomLeafNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.CUSTOM_LEAF
    storage_kind = GraphStorageKind.CUSTOM_LEAF

    def matches(self, obj: Any) -> bool:
        from robo_orchard_lab.utils.state.mixin import CustomizedSaveLoadMixin

        return isinstance(obj, CustomizedSaveLoadMixin)

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        node.obj.save(ctx.owner_path, protocol=ctx.protocol)

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        from robo_orchard_lab.utils.state.save_profile import (
            resolve_save_profile,
        )

        # Custom leaves own a normal State API directory inside the graph
        # owner path, so load them through the tree profile's ``meta.json``
        # dispatcher instead of redispatching on the surrounding graph entry.
        custom_value = resolve_save_profile("tree").load_path(ctx.owner_path)
        ctx.memo[ctx.node_id] = custom_value
        return custom_value


class _OpaqueLeafNodeCodec(_GraphNodeCodec):
    node_type = GraphNodeType.OPAQUE_LEAF
    storage_kind = GraphStorageKind.OPAQUE_LEAF

    def matches(self, obj: Any) -> bool:
        return True

    def save_payload(
        self,
        node: _GraphNode,
        ctx: _GraphSaveContext,
    ) -> None:
        _dump_graph_pickle(
            os.path.join(ctx.owner_path, "value.pkl"),
            node.obj,
            protocol=ctx.protocol,
        )

    def load_payload(
        self,
        ctx: _GraphLoadContext,
    ) -> Any:
        value = _load_graph_pickle(
            os.path.join(ctx.owner_path, "value.pkl"),
            protocol=ctx.protocol,
        )
        ctx.memo[ctx.node_id] = value
        return value


_GRAPH_NODE_CODECS: tuple[_GraphNodeCodec, ...] = (
    _StateNodeCodec(),
    _StateSequenceNodeCodec(),
    _StateListNodeCodec(),
    _DictNodeCodec(),
    _ListNodeCodec(),
    _TupleNodeCodec(),
    _CustomLeafNodeCodec(),
    _OpaqueLeafNodeCodec(),
)

_GRAPH_NODE_CODECS_BY_TYPE = {
    codec.node_type: codec for codec in _GRAPH_NODE_CODECS
}


def _graph_codec_for_obj(obj: Any) -> _GraphNodeCodec:
    for codec in _GRAPH_NODE_CODECS:
        if codec.matches(obj):
            return codec
    raise GraphStateCorruptionError(
        f"No graph-aware codec registered for {type(obj).__name__}."
    )


def _graph_codec_for_node_type(
    node_type: GraphNodeType,
) -> _GraphNodeCodec:
    try:
        return _GRAPH_NODE_CODECS_BY_TYPE[node_type]
    except KeyError as exc:
        raise GraphStateCorruptionError(
            f"No graph-aware codec registered for node type {node_type!r}."
        ) from exc


@dataclass(slots=True)
class _GraphNode:
    """In-memory graph node discovered before writing artifacts."""

    node_id: str
    """Stable id used in manifests and entries for this save operation."""

    obj: Any
    """Original Python object represented by this graph node."""

    codec: _GraphNodeCodec
    """Codec that owns this node's child, save, and load policy."""

    effective_hierarchical_save: bool | None
    """Hierarchical-save value after parent inheritance is applied."""

    owner_parts: tuple[str, ...]
    """Canonical owner path parts relative to the graph root."""

    appearances: list[tuple[str, ...]] = field(default_factory=list)
    """All path appearances for this object, including refs."""

    children: list[_GraphChild] = field(default_factory=list)
    """Outgoing child edges for graph containers."""

    @property
    def node_type(self) -> GraphNodeType:
        """Logical node type used by manifests and entries."""
        return self.codec.node_type

    @property
    def storage_kind(self) -> GraphStorageKind:
        """Payload storage family for this node."""
        return self.codec.storage_kind


class _GraphBuilder:
    def __init__(self) -> None:
        self.nodes: dict[str, _GraphNode] = {}
        self._object_to_node_id: dict[int, str] = {}

    def add(
        self,
        obj: Any,
        path_parts: tuple[str, ...],
        *,
        inherited_hierarchical_save: bool | None = None,
    ) -> str:
        effective_hierarchical_save = _effective_hierarchical_save(
            obj,
            inherited_hierarchical_save,
        )
        obj_id = id(obj)
        existing_node_id = self._object_to_node_id.get(obj_id)
        if existing_node_id is not None:
            node = self.nodes[existing_node_id]
            node.appearances.append(path_parts)
            if (
                effective_hierarchical_save is True
                and node.effective_hierarchical_save is not True
            ):
                node.effective_hierarchical_save = True
                node.children.clear()
                self._populate_children(node)
            return existing_node_id

        node_id = f"n{len(self.nodes)}"
        codec = _graph_codec_for_obj(obj)
        node = _GraphNode(
            node_id=node_id,
            obj=obj,
            codec=codec,
            effective_hierarchical_save=effective_hierarchical_save,
            owner_parts=path_parts,
            appearances=[path_parts],
        )
        self.nodes[node_id] = node
        self._object_to_node_id[obj_id] = node_id
        self._populate_children(node)

        return node_id

    def _populate_children(self, node: _GraphNode) -> None:
        for child in node.codec.iter_children(
            node.obj,
            node.owner_parts,
            node.effective_hierarchical_save,
        ):
            child_id = self.add(
                child.value,
                child.path_parts,
                inherited_hierarchical_save=child.inherited_hierarchical_save,
            )
            node.children.append(
                _GraphChild(
                    segment=child.segment,
                    key=child.key,
                    node_id=child_id,
                )
            )


def _graph_segment_for_dict_key(key: Any) -> str:
    if not isinstance(key, str):
        raise TypeError(
            "Graph-aware State persistence supports string dictionary keys "
            f"only. Got {type(key).__name__}."
        )
    if key in {"", ".", ".."} or os.sep in key:
        raise ValueError(
            "Graph-aware State persistence cannot use dictionary key "
            f"{key!r} as a path segment."
        )
    return key


def _graph_path(parts: tuple[str, ...]) -> str:
    if not parts:
        return "."
    return "/".join(parts)


def _graph_abs_path(root: str, parts: tuple[str, ...]) -> str:
    if not parts:
        return root
    return os.path.join(root, *parts)


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise GraphStateCorruptionError(
            f"Expected {path} to contain a JSON object."
        )
    return payload


def _read_graph_entry(path: str) -> GraphEntryRecord:
    entry = _read_json(path)
    return _validate_graph_record(
        GraphEntryRecord,
        entry,
        context=f"Graph entry {path}",
    )


def _path_has_graph_manifest(path: str) -> bool:
    return os.path.exists(os.path.join(path, GRAPH_MANIFEST_FILE_NAME))


def _path_has_graph_entry(path: str) -> bool:
    return os.path.exists(os.path.join(path, ENTRY_FILE_NAME))


def _save_graph_aware_state(
    state: Any,
    *,
    path: str,
    protocol: Literal["pickle", "cloudpickle"],
) -> None:
    from robo_orchard_lab.utils.state.core import META_FILE_NAME

    root_codec = _graph_codec_for_obj(state)
    state_config = root_codec.build_root_state_config(
        state,
        protocol=protocol,
    )
    if state_config is None:
        raise TypeError(
            "GraphAwareStateProfile.save(...) expected State, "
            "StateSequence, or StateList. "
            f"Got {type(state).__name__}."
        )

    builder = _GraphBuilder()
    root_node_id = builder.add(state, ())
    closure_by_node = {
        node_id: _collect_graph_closure(builder, node_id, set())
        for node_id in builder.nodes
    }
    self_contained_by_node = {
        node_id: _is_graph_closure_self_contained(
            builder=builder,
            root_node_id=node_id,
            closure_node_ids=closure_node_ids,
        )
        for node_id, closure_node_ids in closure_by_node.items()
    }
    with open(os.path.join(path, META_FILE_NAME), "w") as f:
        f.write(state_config.to_str(format="json", indent=2))

    manifest_nodes: dict[str, GraphManifestNodeRecord] = {}
    for node in builder.nodes.values():
        load_scope: GraphLoadScope = (
            GraphLoadScope.SELF_CONTAINED
            if self_contained_by_node[node.node_id]
            else GraphLoadScope.NON_SELF_CONTAINED
        )
        manifest_nodes[node.node_id] = GraphManifestNodeRecord(
            owner_path=_graph_path(node.owner_parts),
            node_type=node.node_type,
            storage_kind=node.storage_kind,
            load_scope=load_scope,
        )

    manifest = GraphManifestRecord(
        profile=GRAPH_PROFILE,
        format_version=GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION,
        root_node_id=root_node_id,
        nodes=manifest_nodes,
    )
    _write_json(
        os.path.join(path, GRAPH_MANIFEST_FILE_NAME),
        manifest.model_dump(mode="json"),
    )

    for node in builder.nodes.values():
        for appearance_parts in node.appearances:
            _write_graph_entry(
                root=path,
                node=node,
                appearance_parts=appearance_parts,
                is_owner=appearance_parts == node.owner_parts,
                is_self_contained=(
                    appearance_parts == node.owner_parts
                    and self_contained_by_node[node.node_id]
                ),
            )

    for node in builder.nodes.values():
        node.codec.save_payload(
            node,
            _GraphSaveContext(
                owner_path=_graph_abs_path(path, node.owner_parts),
                protocol=protocol,
            ),
        )

    for node in builder.nodes.values():
        if (
            node.node_id == root_node_id
            or not self_contained_by_node[node.node_id]
        ):
            continue
        _write_local_graph_manifest(
            root=path,
            local_root_node=node,
            closure_node_ids=closure_by_node[node.node_id],
            builder=builder,
        )


def _write_graph_entry(
    *,
    root: str,
    node: _GraphNode,
    appearance_parts: tuple[str, ...],
    is_owner: bool,
    is_self_contained: bool,
) -> None:
    entry_path = _graph_abs_path(root, appearance_parts)
    os.makedirs(entry_path, exist_ok=True)
    owner_path = _graph_abs_path(root, node.owner_parts)
    load_scope: GraphLoadScope = (
        GraphLoadScope.SELF_CONTAINED
        if is_self_contained
        else GraphLoadScope.NON_SELF_CONTAINED
    )
    entry = GraphEntryRecord(
        kind=GraphEntryKind.OWNER if is_owner else GraphEntryKind.REF,
        node_id=node.node_id,
        node_type=node.node_type,
        storage_kind=node.storage_kind,
        owner_relpath=os.path.relpath(owner_path, entry_path),
        load_scope=load_scope,
    )
    _write_json(
        os.path.join(entry_path, ENTRY_FILE_NAME),
        entry.model_dump(mode="json"),
    )


def _collect_graph_closure(
    builder: _GraphBuilder,
    node_id: str,
    seen: set[str],
) -> set[str]:
    if node_id in seen:
        return seen
    seen.add(node_id)
    for child in builder.nodes[node_id].children:
        _collect_graph_closure(builder, child.node_id, seen)
    return seen


def _is_graph_closure_self_contained(
    *,
    builder: _GraphBuilder,
    root_node_id: str,
    closure_node_ids: set[str],
) -> bool:
    root_parts = builder.nodes[root_node_id].owner_parts
    return all(
        _is_graph_path_within(builder.nodes[node_id].owner_parts, root_parts)
        for node_id in closure_node_ids
    )


def _is_graph_path_within(
    path_parts: tuple[str, ...],
    root_parts: tuple[str, ...],
) -> bool:
    return path_parts[: len(root_parts)] == root_parts


def _relative_graph_parts(
    path_parts: tuple[str, ...],
    root_parts: tuple[str, ...],
) -> tuple[str, ...]:
    if not _is_graph_path_within(path_parts, root_parts):
        raise GraphStateCorruptionError(
            f"Path {_graph_path(path_parts)!r} is outside graph root "
            f"{_graph_path(root_parts)!r}."
        )
    return path_parts[len(root_parts) :]


def _write_local_graph_manifest(
    *,
    root: str,
    local_root_node: _GraphNode,
    closure_node_ids: set[str],
    builder: _GraphBuilder,
) -> None:
    local_root_parts = local_root_node.owner_parts
    nodes: dict[str, GraphManifestNodeRecord] = {}
    for node_id in closure_node_ids:
        node = builder.nodes[node_id]
        nodes[node_id] = GraphManifestNodeRecord(
            owner_path=_graph_path(
                _relative_graph_parts(node.owner_parts, local_root_parts)
            ),
            node_type=node.node_type,
            storage_kind=node.storage_kind,
            load_scope=GraphLoadScope.SELF_CONTAINED,
        )

    manifest = GraphManifestRecord(
        profile=GRAPH_PROFILE,
        format_version=GRAPH_AWARE_HIERARCHICAL_FORMAT_VERSION,
        root_node_id=local_root_node.node_id,
        nodes=nodes,
    )
    _write_json(
        os.path.join(
            _graph_abs_path(root, local_root_parts),
            GRAPH_MANIFEST_FILE_NAME,
        ),
        manifest.model_dump(mode="json"),
    )


def _dump_graph_items_or_children(
    owner_path: str,
    children: list[_GraphChild],
    items: list[Any],
    *,
    protocol: Literal["pickle", "cloudpickle"],
) -> None:
    if children:
        _dump_graph_children_payload(
            owner_path,
            children,
            protocol=protocol,
        )
        return
    _dump_graph_pickle(
        os.path.join(owner_path, "items.pkl"),
        items,
        protocol=protocol,
    )


def _dump_graph_children_payload(
    owner_path: str,
    children: list[_GraphChild],
    *,
    protocol: Literal["pickle", "cloudpickle"],
) -> None:
    children_meta = [
        GraphChildMetaRecord(
            segment=child.segment,
            key=child.key,
        ).model_dump(mode="python")
        for child in children
    ]
    _dump_graph_pickle(
        os.path.join(owner_path, "children_meta.pkl"),
        children_meta,
        protocol=protocol,
    )


def _dump_graph_pickle(
    path: str,
    payload: Any,
    *,
    protocol: Literal["pickle", "cloudpickle"],
) -> None:
    from robo_orchard_lab.utils.state.core import _protocol2module

    pickle_module = _protocol2module[protocol]
    with open(path, "wb") as f:
        pickle_module.dump(payload, f)


def _load_graph_pickle(
    path: str,
    *,
    protocol: Literal["pickle", "cloudpickle"],
) -> Any:
    from robo_orchard_lab.utils.state.core import _protocol2module

    pickle_module = _protocol2module[protocol]
    with open(path, "rb") as f:
        return pickle_module.load(f)


def _load_graph_aware_state(
    path: str,
    protocol: Literal["pickle", "cloudpickle"],
) -> Any:
    if _path_has_graph_entry(path):
        entry = _read_graph_entry(os.path.join(path, ENTRY_FILE_NAME))
        if entry.kind is GraphEntryKind.REF:
            raise NonSelfContainedEntryError("reference_entry")
        if (
            entry.load_scope is not GraphLoadScope.SELF_CONTAINED
            or not _path_has_graph_manifest(path)
        ):
            raise NonSelfContainedEntryError("external_closure_required")

    manifest = _read_graph_manifest(path)
    memo: dict[str, Any] = {}
    return _load_graph_node(
        root=path,
        node_id=manifest.root_node_id,
        manifest=manifest,
        memo=memo,
        protocol=protocol,
    )


def _read_graph_manifest(path: str) -> GraphManifestRecord:
    manifest_path = os.path.join(path, GRAPH_MANIFEST_FILE_NAME)
    if not os.path.exists(manifest_path):
        raise GraphStateCorruptionError(
            f"Graph manifest {manifest_path} does not exist."
        )

    manifest = _read_json(manifest_path)
    header = _validate_graph_record(
        GraphManifestHeaderRecord,
        manifest,
        context=f"Graph manifest {manifest_path}",
    )
    if header.profile != GRAPH_PROFILE:
        raise GraphStateCorruptionError(
            f"Unsupported graph-aware State profile: {header.profile!r}."
        )
    if (
        header.format_version
        not in GRAPH_AWARE_STATE_PROFILE.supported_format_versions
    ):
        raise GraphStateUnsupportedFormatVersionError(
            "Unsupported graph-aware State format version: "
            f"{header.format_version!r}. Supported versions: "
            f"{sorted(GRAPH_AWARE_STATE_PROFILE.supported_format_versions)}."
        )

    if header.format_version == 1:
        return _validate_graph_record(
            GraphManifestRecord,
            manifest,
            context="Graph manifest v1",
        )

    raise GraphStateUnsupportedFormatVersionError(
        "Unsupported graph-aware State format version: "
        f"{header.format_version!r}."
    )


def _load_graph_node(
    *,
    root: str,
    node_id: str,
    manifest: GraphManifestRecord,
    memo: dict[str, Any],
    protocol: Literal["pickle", "cloudpickle"],
) -> Any:
    if node_id in memo:
        return memo[node_id]

    node_info = manifest.nodes.get(node_id)
    if node_info is None:
        raise GraphStateCorruptionError(
            f"Graph manifest is missing node {node_id!r}."
        )

    owner_relpath = node_info.owner_path
    owner_path = (
        root
        if owner_relpath == "."
        else os.path.join(root, *owner_relpath.split("/"))
    )
    _validate_graph_owner_entry(owner_path, node_id, node_info)

    codec = _graph_codec_for_node_type(node_info.node_type)
    if node_info.storage_kind is not codec.storage_kind:
        raise GraphStateCorruptionError(
            f"Graph node {node_id!r} has storage_kind "
            f"{node_info.storage_kind!r}, expected {codec.storage_kind!r}."
        )
    return codec.load_payload(
        _GraphLoadContext(
            root=root,
            owner_path=owner_path,
            node_id=node_id,
            manifest=manifest,
            memo=memo,
            protocol=protocol,
        )
    )


def _validate_graph_owner_entry(
    owner_path: str,
    node_id: str,
    node_info: GraphManifestNodeRecord,
) -> None:
    entry = _read_graph_entry(os.path.join(owner_path, ENTRY_FILE_NAME))
    if entry.kind is not GraphEntryKind.OWNER:
        raise GraphStateCorruptionError(
            f"Graph node {node_id!r} owner path is not an owner entry."
        )
    if entry.node_id != node_id:
        raise GraphStateCorruptionError(
            f"Graph node {node_id!r} has inconsistent node_id."
        )
    if entry.node_type != node_info.node_type:
        raise GraphStateCorruptionError(
            f"Graph node {node_id!r} has inconsistent node_type."
        )
    if entry.storage_kind != node_info.storage_kind:
        raise GraphStateCorruptionError(
            f"Graph node {node_id!r} has inconsistent storage_kind."
        )


def _load_graph_children_meta(
    owner_path: str,
    *,
    protocol: Literal["pickle", "cloudpickle"],
) -> list[GraphChildMetaRecord]:
    children = _load_graph_pickle(
        os.path.join(owner_path, "children_meta.pkl"),
        protocol=protocol,
    )
    if not isinstance(children, list):
        raise GraphStateCorruptionError(
            f"Invalid children metadata in {owner_path}."
        )
    return [
        _validate_graph_record(
            GraphChildMetaRecord,
            child,
            context=f"Child metadata in {owner_path}",
        )
        for child in children
    ]


def _load_graph_meta_dict(
    owner_path: str,
    *,
    filename: str,
    protocol: Literal["pickle", "cloudpickle"],
    error_label: str,
    required: bool = True,
) -> dict[str, Any]:
    meta_path = os.path.join(owner_path, filename)
    if not os.path.exists(meta_path):
        if required:
            raise GraphStateCorruptionError(
                f"Missing {error_label} in {meta_path}."
            )
        return {}
    meta = _load_graph_pickle(meta_path, protocol=protocol)
    if not isinstance(meta, dict):
        raise GraphStateCorruptionError(
            f"Invalid {error_label} in {meta_path}."
        )
    return meta


def _iter_loaded_graph_children(
    ctx: _GraphLoadContext,
) -> Iterator[tuple[Any, Any]]:
    for child in _load_graph_children_meta(
        ctx.owner_path,
        protocol=ctx.protocol,
    ):
        yield (
            child.key,
            _load_graph_child(
                ctx,
                child_path=os.path.join(ctx.owner_path, child.segment),
            ),
        )


def _load_graph_items_or_children(
    ctx: _GraphLoadContext,
    *,
    error_label: str,
) -> list[Any]:
    items_path = os.path.join(ctx.owner_path, "items.pkl")
    if os.path.exists(items_path):
        loaded_items = _load_graph_pickle(
            items_path,
            protocol=ctx.protocol,
        )
        if not isinstance(loaded_items, list):
            raise GraphStateCorruptionError(
                f"Invalid {error_label} in {items_path}."
            )
        return loaded_items
    return [value for _, value in _iter_loaded_graph_children(ctx)]


def _load_graph_child(
    ctx: _GraphLoadContext,
    *,
    child_path: str,
) -> Any:
    entry = _read_graph_entry(os.path.join(child_path, ENTRY_FILE_NAME))
    return _load_graph_node(
        root=ctx.root,
        node_id=entry.node_id,
        manifest=ctx.manifest,
        memo=ctx.memo,
        protocol=ctx.protocol,
    )


register_save_profile(GRAPH_AWARE_STATE_PROFILE)
