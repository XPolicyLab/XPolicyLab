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

"""State payload conversion and recovery helpers."""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import cast

from robo_orchard_lab.utils.state.core import (
    ConstructableStateApplyMode,
    State,
    StateList,
    StateSequence,
)

__all__ = [
    "decode_state_payload_for_apply",
    "obj2state",
    "state2obj",
    "validate_recovery_state",
]


def obj2state(obj: object) -> object:
    """Encode nested runtime values into State API transport containers.

    Use this before persistence or capture when payloads may contain nested
    ``StateSaveLoadMixin`` objects, ``State`` payloads, or Python containers.
    Lists and tuples become ``StateSequence`` records, nested stateful objects
    become ``State``, and plain scalar values pass through unchanged.

    Args:
        obj (object): Object or nested payload to encode.

    Returns:
        object: Encoded payload graph containing plain values plus any needed
        ``State`` / ``StateSequence`` transport containers.
    """
    return _obj2state(obj, active_paths={}, path="state")


def state2obj(obj: object) -> object:
    """Materialize State API payloads into fresh runtime objects.

    ``State`` payloads must include ``class_type`` so a fresh instance can be
    allocated and populated. The ``class_type`` must satisfy the two-phase
    ``StateMaterializeProtocol``; ``StateSaveLoadMixin`` already does.
    ``StateSequence`` restores list or tuple fidelity. Apply-only nested
    ``State`` payloads without ``class_type`` are rejected here and must go
    through ``decode_state_payload_for_apply(...)`` instead.

    Args:
        obj (object): State API payload or nested payload to materialize.

    Returns:
        object: Fresh live object or recursively decoded Python value.
    """
    return _decode_state_payload(
        obj,
        path="state",
        materialize_nested_state=True,
        active_paths={},
        profile_aware_nested_state=True,
    )


def decode_state_payload_for_apply(
    obj: object,
    *,
    save_profile: str | None = None,
    constructable_state_apply_mode: ConstructableStateApplyMode = (
        ConstructableStateApplyMode.MATERIALIZE
    ),
) -> object:
    """Decode State API transport containers for live-object apply paths.

    Unlike ``state2obj(...)``, apply-only nested ``State`` payloads stay as
    ``State`` so an existing live owner can apply them. Constructable nested
    ``State`` payloads whose ``class_type`` satisfies
    ``StateMaterializeProtocol`` may either materialize into fresh objects or
    remain ``State`` payloads, depending on
    ``constructable_state_apply_mode``.

    Args:
        obj (object): State API payload or nested payload to decode.
        save_profile (str | None, optional): Effective profile of the payload
            root. When set, graph-aware payloads are decoded with the
            graph-aware apply decoder. Default is ``None``.
        constructable_state_apply_mode (ConstructableStateApplyMode):
            Whether nested constructable ``State`` payloads should materialize
            into fresh objects or remain ``State`` payloads for the owner to
            apply. Default is ``ConstructableStateApplyMode.MATERIALIZE``.

    Returns:
        object: Payload decoded for live-object state application.
    """
    from robo_orchard_lab.utils.state.save_profile import resolve_save_profile

    return resolve_save_profile(save_profile).decode_payload_for_apply(
        obj,
        active_paths={},
        path="state",
        constructable_state_apply_mode=constructable_state_apply_mode,
    )


def _decode_state_payload(
    obj: object,
    *,
    path: str,
    materialize_nested_state: bool,
    active_paths: dict[int, str] | None = None,
    preserve_identity: bool = False,
    constructable_state_apply_mode: ConstructableStateApplyMode = (
        ConstructableStateApplyMode.MATERIALIZE
    ),
    profile_aware_nested_state: bool = False,
) -> object:
    return _StatePayloadDecoder(
        materialize_nested_state=materialize_nested_state,
        active_paths=active_paths,
        memo={} if preserve_identity else None,
        constructable_state_apply_mode=constructable_state_apply_mode,
        profile_aware_nested_state=profile_aware_nested_state,
    ).decode(obj, path=path)


def validate_recovery_state(
    state: State,
    *,
    require_class_type: bool = False,
    require_config: bool = False,
    context: str = "validate_recovery_state(...)",
) -> None:
    """Validate a State payload against the shared recovery contract.

    Args:
        state (State): State payload to validate.
        require_class_type (bool, optional): Whether materialization metadata
            must include ``class_type``. Default is ``False``.
        require_config (bool, optional): Whether recreate metadata must
            include ``config``. Default is ``False``.
        context (str, optional): Human-readable context used in error
            messages. Default is ``"validate_recovery_state(...)"``.
    """
    _validate_recovery_state(
        state,
        context=context,
        require_class_type=require_class_type,
        require_config=require_config,
    )


def _validate_recovery_state(
    state: State,
    *,
    context: str,
    require_class_type: bool = False,
    require_config: bool = False,
) -> None:
    if not isinstance(state, State):
        raise TypeError(
            f"{context} expects a State payload. Got {type(state).__name__}."
        )
    if not isinstance(state.state, dict):
        raise TypeError(
            f"{context} expects State.state to be a dict. "
            f"Got {type(state.state).__name__}."
        )
    if require_class_type and state.class_type is None:
        raise ValueError(
            f"{context} requires State.class_type for materialization."
        )
    if require_config and state.config is None:
        raise ValueError(f"{context} requires State.config for recreation.")


def _enter_state_path(
    obj: object,
    active_paths: dict[int, str],
    path: str,
) -> int:
    obj_id = id(obj)
    existing_path = active_paths.get(obj_id)
    if existing_path is not None:
        raise ValueError(
            "Circular reference detected in State payload at "
            f"`{path}`. The same object was already seen at "
            f"`{existing_path}`."
        )
    active_paths[obj_id] = path
    return obj_id


def _exit_state_path(active_paths: dict[int, str], obj_id: int) -> None:
    active_paths.pop(obj_id, None)


def _format_state_path(parent: str, child: str) -> str:
    if parent:
        return f"{parent}.{child}"
    return child


def _canonicalize_get_state_result(
    state: State | dict[str, object],
    *,
    owner: str,
    default_class_type: type[object],
    copy_payload: bool,
) -> State:
    if isinstance(state, State):
        ret = copy.deepcopy(state) if copy_payload else state
    elif isinstance(state, dict):
        ret = State(
            state=cast(
                dict[str, object],
                copy.deepcopy(state) if copy_payload else state,
            ),
            class_type=default_class_type,
            config=None,
            parameters=None,
            hierarchical_save=None,
        )
    else:
        raise TypeError(
            f"{owner}._get_state() must return a State or dict payload. "
            f"Got {type(state).__name__}."
        )
    _validate_recovery_state(
        ret,
        context=f"{owner}._get_state()",
    )
    return ret


def _copy_state_shell(obj: State, *, empty_state: bool) -> State:
    state = obj.model_copy(deep=False)
    state.config = copy.deepcopy(obj.config)
    state.parameters = copy.deepcopy(obj.parameters)
    if empty_state:
        state.state = {}
    return state


_DECODE_MEMO_MISS = object()


@dataclass(slots=True)
class _StatePayloadDecoder:
    materialize_nested_state: bool
    active_paths: dict[int, str] | None = None
    memo: dict[int, object] | None = None
    constructable_state_apply_mode: ConstructableStateApplyMode = (
        ConstructableStateApplyMode.MATERIALIZE
    )
    profile_aware_nested_state: bool = False

    def _memo_value(self, obj: object) -> object:
        if self.memo is None:
            return _DECODE_MEMO_MISS
        return self.memo.get(id(obj), _DECODE_MEMO_MISS)

    def _enter_path(self, obj: object, path: str) -> int | None:
        if self.active_paths is None:
            return None
        return _enter_state_path(obj, self.active_paths, path)

    def _exit_path(self, obj_id: int | None) -> None:
        if obj_id is None or self.active_paths is None:
            return
        _exit_state_path(self.active_paths, obj_id)

    def _decoder_for_nested_state(
        self,
        state: State,
    ) -> "_StatePayloadDecoder":
        if self.memo is not None or not self.profile_aware_nested_state:
            return self

        from robo_orchard_lab.utils.state.save_profile import (
            resolve_save_profile,
        )

        if not resolve_save_profile(
            state.save_profile
        ).preserve_identity_during_apply:
            return self

        return _StatePayloadDecoder(
            materialize_nested_state=self.materialize_nested_state,
            memo={},
            constructable_state_apply_mode=(
                self.constructable_state_apply_mode
            ),
            profile_aware_nested_state=True,
        )

    def _decode_indexed_items(
        self,
        items: StateList | list[object] | tuple[object, ...],
        *,
        path: str,
    ) -> list[object]:
        return [
            self.decode(
                item,
                path=f"{path}[{idx}]",
            )
            for idx, item in enumerate(items)
        ]

    def _decode_state_value(self, obj: State, *, path: str) -> object:
        from robo_orchard_lab.utils.state.mixin import StateMaterializeProtocol

        memo_value = self._memo_value(obj)
        if memo_value is not _DECODE_MEMO_MISS:
            return memo_value

        obj_id = self._enter_path(obj, path)
        try:
            payload_decoder = self._decoder_for_nested_state(obj)
            payload_path = _format_state_path(path, "state")
            memo_owner = payload_decoder.memo
            preserve_state_payload = not self.materialize_nested_state and (
                obj.class_type is None
                or self.constructable_state_apply_mode
                is ConstructableStateApplyMode.PRESERVE_STATE
            )

            if preserve_state_payload:
                state = _copy_state_shell(
                    obj,
                    empty_state=memo_owner is not None,
                )
                if memo_owner is not None:
                    memo_owner[id(obj)] = state
                state.state = cast(
                    dict[str, object],
                    payload_decoder.decode(
                        obj.state,
                        path=payload_path,
                    ),
                )
                return state

            _validate_recovery_state(
                obj,
                context="state2obj(...)",
                require_class_type=True,
            )
            class_type = obj.class_type
            if class_type is None:
                raise ValueError(
                    "state2obj(...) requires State.class_type for "
                    "materialization."
                )
            if not issubclass(class_type, StateMaterializeProtocol):
                raise TypeError(
                    f"Class type {class_type} does not implement "
                    "StateMaterializeProtocol."
                )

            stateful_type = cast(type[StateMaterializeProtocol], class_type)
            new_obj: StateMaterializeProtocol | None = None
            if memo_owner is not None:
                new_obj = stateful_type.allocate_state_instance()
                memo_owner[id(obj)] = new_obj

            state = _copy_state_shell(
                obj,
                empty_state=memo_owner is not None,
            )
            state.state = cast(
                dict[str, object],
                payload_decoder.decode(
                    obj.state,
                    path=payload_path,
                ),
            )
        finally:
            self._exit_path(obj_id)

        if new_obj is not None:
            new_obj.apply_decoded_state(state)
            return new_obj

        new_obj = stateful_type.allocate_state_instance()
        new_obj.apply_decoded_state(state)
        return new_obj

    def _decode_list_value(
        self,
        obj: StateList | list[object],
        *,
        path: str,
    ) -> list[object]:
        memo_value = self._memo_value(obj)
        if memo_value is not _DECODE_MEMO_MISS:
            return cast(list[object], memo_value)

        obj_id = self._enter_path(obj, path)
        try:
            items: list[object]
            if isinstance(obj, StateList) and self.memo is not None:
                items = StateList(
                    [],
                    hierarchical_save=obj.hierarchical_save,
                    save_profile=obj.save_profile,
                )
            else:
                items = []
            if self.memo is not None:
                self.memo[id(obj)] = items
            items.extend(self._decode_indexed_items(obj, path=path))
            return items
        finally:
            self._exit_path(obj_id)

    def _decode_state_sequence_value(
        self,
        obj: StateSequence,
        *,
        path: str,
    ) -> list[object] | tuple[object, ...]:
        memo_value = self._memo_value(obj)
        if memo_value is not _DECODE_MEMO_MISS:
            return cast(list[object] | tuple[object, ...], memo_value)

        obj_id = self._enter_path(obj, path)
        try:
            if obj.kind == "tuple":
                return tuple(self._decode_indexed_items(obj.items, path=path))

            items: list[object] = []
            if self.memo is not None:
                self.memo[id(obj)] = items
            items.extend(self._decode_indexed_items(obj.items, path=path))
            return items
        finally:
            self._exit_path(obj_id)

    def _decode_tuple_value(
        self,
        obj: tuple[object, ...],
        *,
        path: str,
    ) -> tuple[object, ...]:
        obj_id = self._enter_path(obj, path)
        try:
            return tuple(self._decode_indexed_items(obj, path=path))
        finally:
            self._exit_path(obj_id)

    def _decode_dict_value(
        self,
        obj: dict[object, object],
        *,
        path: str,
    ) -> dict[object, object]:
        memo_value = self._memo_value(obj)
        if memo_value is not _DECODE_MEMO_MISS:
            return cast(dict[object, object], memo_value)

        obj_id = self._enter_path(obj, path)
        try:
            decoded: dict[object, object] = {}
            if self.memo is not None:
                self.memo[id(obj)] = decoded
            for key, value in obj.items():
                decoded[key] = self.decode(
                    value,
                    path=_format_state_path(path, str(key)),
                )
            return decoded
        finally:
            self._exit_path(obj_id)

    def decode(self, obj: object, *, path: str) -> object:
        from robo_orchard_lab.utils.state.mixin import WrappedHuggingFaceObj

        if isinstance(obj, State):
            return self._decode_state_value(obj, path=path)

        if isinstance(obj, (StateList, list)):
            return self._decode_list_value(obj, path=path)

        if isinstance(obj, StateSequence):
            return self._decode_state_sequence_value(obj, path=path)

        if isinstance(obj, tuple):
            return self._decode_tuple_value(obj, path=path)

        if isinstance(obj, dict):
            return self._decode_dict_value(obj, path=path)

        if isinstance(obj, WrappedHuggingFaceObj):
            return obj.unwrapped_obj

        return obj


def _obj2state(
    obj: object,
    *,
    active_paths: dict[int, str],
    path: str,
) -> object:
    from robo_orchard_lab.utils.state.mixin import (
        HuggingFacePreTrainedObj,
        StateSaveLoadMixin,
        WrappedHuggingFaceObj,
    )

    if isinstance(obj, State):
        obj_id = _enter_state_path(obj, active_paths, path)
        try:
            obj.state = cast(
                dict[str, object],
                _obj2state(
                    obj.state,
                    active_paths=active_paths,
                    path=_format_state_path(path, "state"),
                ),
            )
            return obj
        finally:
            _exit_state_path(active_paths, obj_id)
    elif isinstance(obj, StateList):
        obj_id = _enter_state_path(obj, active_paths, path)
        try:
            return StateSequence(
                kind="list",
                items=[
                    _obj2state(
                        item,
                        active_paths=active_paths,
                        path=f"{path}[{idx}]",
                    )
                    for idx, item in enumerate(obj)
                ],
                hierarchical_save=obj.hierarchical_save,
            )
        finally:
            _exit_state_path(active_paths, obj_id)
    elif isinstance(obj, StateSequence):
        obj_id = _enter_state_path(obj, active_paths, path)
        try:
            return obj.model_copy(
                update={
                    "items": [
                        _obj2state(
                            item,
                            active_paths=active_paths,
                            path=f"{path}[{idx}]",
                        )
                        for idx, item in enumerate(obj.items)
                    ]
                }
            )
        finally:
            _exit_state_path(active_paths, obj_id)
    elif isinstance(obj, (list, tuple)):
        obj_id = _enter_state_path(obj, active_paths, path)
        try:
            kind = "tuple" if isinstance(obj, tuple) else "list"
            return StateSequence(
                kind=kind,
                items=[
                    _obj2state(
                        item,
                        active_paths=active_paths,
                        path=f"{path}[{idx}]",
                    )
                    for idx, item in enumerate(obj)
                ],
            )
        finally:
            _exit_state_path(active_paths, obj_id)
    elif isinstance(obj, dict):
        obj_id = _enter_state_path(obj, active_paths, path)
        try:
            return {
                k: _obj2state(
                    v,
                    active_paths=active_paths,
                    path=_format_state_path(path, str(k)),
                )
                for k, v in obj.items()
            }
        finally:
            _exit_state_path(active_paths, obj_id)
    elif isinstance(obj, StateSaveLoadMixin):
        obj_id = _enter_state_path(obj, active_paths, path)
        state = _canonicalize_get_state_result(
            obj._get_state(),
            owner=type(obj).__name__,
            default_class_type=type(obj),
            copy_payload=False,
        )
        try:
            state.state = cast(
                dict[str, object],
                _obj2state(
                    state.state,
                    active_paths=active_paths,
                    path=_format_state_path(path, "state"),
                ),
            )
            return state
        finally:
            _exit_state_path(active_paths, obj_id)
    elif WrappedHuggingFaceObj.is_huggingface_pretrained(obj):
        wrapped_obj = WrappedHuggingFaceObj(
            unwrapped_obj=cast(HuggingFacePreTrainedObj, obj)
        )
        return wrapped_obj
    else:
        return obj
