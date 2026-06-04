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

"""Mixin and adapter classes for objects that participate in State recovery."""

from __future__ import annotations
import copy
import os
from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping, MutableSequence, MutableSet
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin as HuggingFaceProcessorMixin,
)
from typing_extensions import Self

from robo_orchard_lab.utils.state.conversion import (
    _canonicalize_get_state_result,
    _validate_recovery_state,
    decode_state_payload_for_apply,
    obj2state,
)
from robo_orchard_lab.utils.state.core import (
    META_FILE_NAME,
    ConstructableStateApplyMode,
    State,
    StateConfig,
    _load_state_from_path,
)

__all__ = [
    "ConstructableStateApplyMode",
    "CustomizedSaveLoadMixin",
    "HuggingFacePreTrainedObj",
    "StateMaterializeProtocol",
    "StateMaterializeMixin",
    "StatePersistenceMixin",
    "StateRuntimeMixin",
    "StateRuntimeProtocol",
    "StateSaveLoadMixin",
    "WrappedHuggingFaceObj",
]

StatePayloadDict: TypeAlias = dict[str, object]


@runtime_checkable
class StateMaterializeProtocol(Protocol):
    """Two-phase protocol for fresh-object materialization from ``State``.

    ``state2obj(...)`` and constructable nested apply decode use this protocol
    when a ``State.class_type`` should materialize into a new live object. The
    two-phase shape lets graph-aware decode allocate a placeholder before
    recursively decoding children, so shared identity and cycles can be
    restored without requiring ``StateSaveLoadMixin``.
    """

    @classmethod
    def allocate_state_instance(cls) -> Self:
        """Allocate an instance before decoded state is available."""
        ...

    def apply_decoded_state(self, state: State) -> None:
        """Hydrate this instance from an already-decoded ``State``."""
        ...


@runtime_checkable
class StateRuntimeProtocol(Protocol):
    """Narrow live-object State capture and apply contract."""

    def get_state(self) -> State:
        """Capture the current live runtime state."""
        ...

    def load_state(self, state: State) -> None:
        """Apply a State payload to this live object."""
        ...


class StateRuntimeMixin:
    """Mixin for live-object State capture and runtime recovery.

    Use this mixin when a live object can export a canonical ``State`` payload
    and later apply a compatible payload back to itself. Subclasses usually
    implement ``_get_state()`` and ``_set_state(...)``; callers should use
    ``get_state()`` and ``load_state(state)`` for runtime recovery.

    The default ``_get_state()`` / ``_set_state(...)`` fallback only supports
    attribute-backed objects. Live objects whose primary runtime state lives
    in a mutable container such as a ``dict`` / ``list`` / ``set`` subclass
    must provide explicit recovery hooks.
    """

    def _raise_unsupported_default_container_state(
        self,
        api_name: str,
    ) -> None:
        raise TypeError(
            "StateSaveLoadMixin default "
            f"`{api_name}` only supports attribute-backed objects. "
            "Mutable container-backed objects must override `_get_state()` / "
            "`_set_state(...)` or provide compatible `__getstate__()` / "
            f"`__setstate__(...)`. Got {type(self).__name__}."
        )

    def _uses_mutable_container_storage(self) -> bool:
        return isinstance(
            self,
            (MutableMapping, MutableSequence, MutableSet),
        )

    def _get_ignore_save_attributes(self) -> list[str]:
        return []

    def _get_state(self) -> State:
        # raise NotImplementedError
        # if has __getstate__ method, use it to get the state
        if hasattr(self, "__getstate__"):
            state_dict_obj = self.__getstate__()  # type: ignore
            if not isinstance(state_dict_obj, dict):
                raise TypeError(
                    "StateSaveLoadMixin default `_get_state()` expects "
                    "`__getstate__()` to return a dict payload. "
                    f"Got {type(state_dict_obj).__name__}."
                )
            state_dict = cast(StatePayloadDict, state_dict_obj)
        else:
            if self._uses_mutable_container_storage():
                self._raise_unsupported_default_container_state("_get_state()")
            state_dict = cast(StatePayloadDict, self.__dict__.copy())
        for key in self._get_ignore_save_attributes():
            state_dict.pop(key, None)
        return State(
            state=state_dict,
            class_type=type(self),
            config=None,
            hierarchical_save=None,
        )

    def get_state(self) -> State:
        """Capture this live object's runtime state as a State payload.

        The returned payload is caller-owned and can be passed to
        ``load_state(state)`` on a compatible live object. Nested State API
        values are normalized with ``obj2state(...)``. Circular references fail
        fast because State API v1 supports tree-shaped payloads only.

        Returns:
            State: Canonical runtime recovery payload.
        """
        state = _canonicalize_get_state_result(
            self._get_state(),
            owner=type(self).__name__,
            default_class_type=type(self),
            copy_payload=True,
        )
        state.state = cast(StatePayloadDict, obj2state(state.state))
        return state

    def _set_state(self, state: State) -> None:
        # raise NotImplementedError
        state_dict = state.state
        if hasattr(self, "__setstate__"):
            self.__setstate__(state_dict)  # type: ignore
        else:
            if self._uses_mutable_container_storage():
                self._raise_unsupported_default_container_state(
                    "_set_state(...)"
                )
            self.__dict__.update(state_dict)

    def load_state(
        self,
        state: State,
        *,
        constructable_state_apply_mode: ConstructableStateApplyMode
        | None = None,
    ) -> None:
        """Apply a State payload to this live object.

        Nested transport containers are decoded before ``_set_state(...)``
        receives the payload. By default, nested constructable ``State``
        payloads materialize into fresh live objects. Callers may pass
        ``constructable_state_apply_mode=ConstructableStateApplyMode.PRESERVE_STATE``
        to preserve constructable nested payloads as ``State`` for
        owner-directed apply.

        Args:
            state (State): Runtime ``State`` payload to apply.
            constructable_state_apply_mode (ConstructableStateApplyMode):
                Optional override for how nested constructable ``State``
                payloads are decoded during apply. ``None`` uses the default
                materialize behavior.
        """
        if not isinstance(state, State):
            raise TypeError(
                "StateRuntimeMixin.load_state(...) expects a State payload. "
                f"Got {type(state).__name__}."
            )
        state = copy.deepcopy(state)
        self._apply_state(
            state,
            constructable_state_apply_mode=constructable_state_apply_mode,
            context="StateRuntimeMixin.load_state(...)",
        )

    def _apply_state(
        self,
        state: State,
        *,
        constructable_state_apply_mode: ConstructableStateApplyMode
        | None = None,
        context: str,
    ) -> None:
        _validate_recovery_state(
            state,
            context=context,
        )
        if constructable_state_apply_mode is None:
            constructable_state_apply_mode = (
                ConstructableStateApplyMode.MATERIALIZE
            )
        elif not isinstance(
            constructable_state_apply_mode,
            ConstructableStateApplyMode,
        ):
            raise TypeError(
                "constructable_state_apply_mode must be "
                "ConstructableStateApplyMode or None."
            )
        state.state = cast(
            StatePayloadDict,
            decode_state_payload_for_apply(
                state.state,
                save_profile=state.save_profile,
                constructable_state_apply_mode=constructable_state_apply_mode,
            ),
        )
        self._set_state(state)


class StatePersistenceMixin(StateRuntimeMixin):
    """Filesystem persistence helpers layered over State runtime recovery."""

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
        hierarchical_save: bool | None = None,
    ) -> None:
        """Persist this object's current State payload to disk.

        This is a filesystem convenience around ``_get_state()``,
        ``obj2state(...)``, and ``State.save(...)``. For live recovery, prefer
        passing ``get_state()`` directly instead of round-tripping through a
        path.

        Args:
            path (str): Empty directory path to create or use for the payload.
            protocol (Literal["pickle", "cloudpickle"], optional): Pickle
                backend for non-tensor Python values. Default is
                ``"cloudpickle"``.
            hierarchical_save (bool | None, optional): Overrides the payload's
                hierarchical save behavior when not ``None``. Default is
                ``None``.
        """
        state = _canonicalize_get_state_result(
            self._get_state(),
            owner=type(self).__name__,
            default_class_type=type(self),
            copy_payload=False,
        )
        state.state = cast(StatePayloadDict, obj2state(state.state))
        if hierarchical_save is not None:
            state.hierarchical_save = hierarchical_save
        state.save(path, protocol=protocol)

    def load_state_from_path(
        self,
        path: str,
        *,
        constructable_state_apply_mode: ConstructableStateApplyMode
        | None = None,
    ) -> None:
        """Load a persisted State payload from disk and apply it."""

        state = _load_state_from_path(path)
        if not isinstance(state, State):
            raise TypeError(
                "StatePersistenceMixin.load_state_from_path(path) expected "
                f"the path to contain a State payload. Got "
                f"{type(state).__name__}."
            )
        self.load_state(
            state,
            constructable_state_apply_mode=constructable_state_apply_mode,
        )

    @classmethod
    def load(cls, path: str) -> Self:
        """Create a new instance from a persisted State directory.

        This classmethod allocates ``cls`` with
        ``allocate_state_instance()`` and then delegates to
        ``load_state_from_path(path)``. Use ``load_state(state)`` when
        applying runtime recovery to an existing live object.

        Args:
            path (str): Directory containing a persisted State payload.

        Returns:
            Self: Newly allocated object with the persisted state applied.
        """
        allocator = getattr(cls, "allocate_state_instance", None)
        if callable(allocator):
            obj = cast(Self, allocator())
        else:
            obj = cls.__new__(cls)
        obj.load_state_from_path(path)
        return obj


class StateMaterializeMixin(StateRuntimeMixin):
    """Two-phase materialization helpers layered over runtime apply hooks."""

    @classmethod
    def allocate_state_instance(cls) -> Self:
        """Allocate a fresh instance for ``state2obj(...)`` materialization."""
        return cls.__new__(cls)

    def apply_decoded_state(self, state: State) -> None:
        """Apply a decoded ``State`` payload during fresh materialization."""
        self._set_state(state)


class StateSaveLoadMixin(StatePersistenceMixin, StateMaterializeMixin):
    """Compatibility aggregate for State runtime, persistence, and materialize.

    New recovery boundaries should prefer the narrowest applicable surface:
    ``StateRuntimeProtocol`` / ``StateRuntimeMixin`` for live recovery,
    ``StatePersistenceMixin`` for filesystem persistence, and
    ``StateMaterializeProtocol`` / ``StateMaterializeMixin`` for
    ``state2obj(...)`` materialization. This aggregate keeps the historical
    ``save`` / ``load`` / ``load_state(path_or_state)`` behavior intact.
    """

    def load_state(
        self,
        path_or_state: str | State,
        *,
        constructable_state_apply_mode: ConstructableStateApplyMode
        | None = None,
    ) -> None:
        """Apply a State payload or persisted State directory to this object.

        Passing a ``State`` object is the canonical runtime recovery path. A
        string path is accepted only for compatibility; new code should use
        ``load_state_from_path(path)`` for persistence.
        """

        if isinstance(path_or_state, str):
            self.load_state_from_path(
                path_or_state,
                constructable_state_apply_mode=constructable_state_apply_mode,
            )
            return
        if isinstance(path_or_state, State):
            state = copy.deepcopy(path_or_state)
        else:
            raise TypeError(
                "StateSaveLoadMixin.load_state(...) expects a str path or "
                f"State payload. Got {type(path_or_state).__name__}."
            )
        self._apply_state(
            state,
            constructable_state_apply_mode=constructable_state_apply_mode,
            context="StateSaveLoadMixin.load_state(...)",
        )


class CustomizedSaveLoadMixin(metaclass=ABCMeta):
    """Adapter mixin for objects that own custom filesystem serialization.

    Use this when an object already has a real save/load format and should be
    stored inside State API directories without being converted into a
    ``State`` payload. Subclasses implement ``_save_impl(...)`` and
    ``load(...)``; this mixin writes the State API metadata that lets generic
    loaders redispatch to the subclass.
    """

    @abstractmethod
    def _save_impl(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> dict[str, object] | None: ...

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
        hierarchical_save: bool | None = None,
    ) -> None:
        """Save this object with its custom format and State API metadata.

        Args:
            path (str): Directory path owned by the custom serializer.
            protocol (Literal["pickle", "cloudpickle"], optional): Protocol
                forwarded to ``_save_impl(...)``. Default is ``"cloudpickle"``.
            hierarchical_save (bool | None, optional): Accepted for State API
                call-site compatibility and ignored by the base implementation.
                Default is ``None``.
        """
        load_kwargs: dict[str, object] | None = self._save_impl(
            path,
            protocol,
        )
        if load_kwargs is None:
            load_kwargs = {}
        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(
                StateConfig(
                    class_type=type(self),
                    load_kwargs=load_kwargs,
                ).to_str(format="json", indent=2)
            )

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> Self:
        """Load an object from the custom directory format.

        Args:
            path (str): Directory previously written by ``save(...)``.

        Returns:
            Self: Object reconstructed by the subclass-specific loader.
        """
        ...


HuggingFacePreTrainedObj: TypeAlias = (
    PreTrainedModel | PreTrainedTokenizerBase | HuggingFaceProcessorMixin
)


class WrappedHuggingFaceObj(CustomizedSaveLoadMixin):
    """State API adapter for Hugging Face pretrained objects.

    The wrapper does not own a new serialization format. It delegates saving
    and loading to ``save_pretrained(...)`` / ``from_pretrained(...)`` while
    recording the original Hugging Face class in State API metadata.
    """

    unwrapped_obj: HuggingFacePreTrainedObj

    def __init__(self, unwrapped_obj: HuggingFacePreTrainedObj) -> None:
        if not self.is_huggingface_pretrained(unwrapped_obj):
            raise TypeError(
                f"Expected a HuggingFace PreTrainedModel, "
                f"PreTrainedTokenizerBase or ProcessorMixin, "
                f"but got {type(unwrapped_obj)}."
            )

        self.unwrapped_obj = unwrapped_obj

    def _save_impl(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
    ) -> dict[str, object] | None:
        self.unwrapped_obj.save_pretrained(path)
        return None

    def save(
        self,
        path: str,
        protocol: Literal["pickle", "cloudpickle"] = "cloudpickle",
        hierarchical_save: bool | None = None,
    ) -> None:
        """Save the wrapped Hugging Face object with State API metadata.

        Args:
            path (str): Directory passed to ``save_pretrained(...)``.
            protocol (Literal["pickle", "cloudpickle"], optional): Accepted for
                State API compatibility; Hugging Face serialization owns the
                on-disk format. Default is ``"cloudpickle"``.
            hierarchical_save (bool | None, optional): Accepted for State API
                call-site compatibility and ignored. Default is ``None``.
        """
        load_kwargs: dict[str, object] | None = self._save_impl(
            path,
            protocol,
        )
        if load_kwargs is None:
            load_kwargs = {}
        with open(os.path.join(path, META_FILE_NAME), "w") as f:
            f.write(
                StateConfig(
                    class_type=type(self),
                    load_kwargs=load_kwargs,
                    state_class_type=type(self.unwrapped_obj),
                ).to_str(format="json", indent=2)
            )

    @classmethod
    def load(cls, path: str) -> HuggingFacePreTrainedObj:
        """Load the original Hugging Face object from a wrapped directory.

        Args:
            path (str): Directory containing ``meta.json`` and Hugging Face
                ``save_pretrained(...)`` files.

        Returns:
            HuggingFacePreTrainedObj: Object returned by the original class's
            ``from_pretrained(...)`` method.
        """
        meta_path = os.path.join(path, META_FILE_NAME)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file {meta_path} does not exist.")
        # load the meta.json file
        with open(meta_path, "r") as f:
            meta = f.read()
        type_config = StateConfig.from_str(meta, format="json")
        if type_config.state_class_type is None:
            raise ValueError(
                "WrappedHuggingFaceObj metadata is missing `state_class_type`."
            )
        if type_config.class_type != cls:
            raise TypeError(
                f"Type config class type {type_config.class_type} "
                f"does not match {cls}."
            )
        origin_type: HuggingFacePreTrainedObj = type_config.state_class_type  # type: ignore # noqa

        return origin_type.from_pretrained(path)  # type: ignore

    @staticmethod
    def is_huggingface_pretrained(obj: object) -> bool:
        return isinstance(
            obj,
            (
                PreTrainedModel,
                PreTrainedTokenizerBase,
                HuggingFaceProcessorMixin,
            ),
        )
