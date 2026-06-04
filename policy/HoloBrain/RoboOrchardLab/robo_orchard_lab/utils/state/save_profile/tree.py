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

"""Default tree-shaped State persistence profile."""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import safetensors
import torch
from safetensors import (
    numpy as safetensors_numpy,
    torch as safetensors_pytorch,
)

from robo_orchard_lab.utils.state.core import (
    META_FILE_NAME,
    ConstructableStateApplyMode,
    State,
    StateConfig,
    StateList,
    StateSequence,
    _protocol2module,
    _save_state_api_root,
)
from robo_orchard_lab.utils.state.mixin import (
    CustomizedSaveLoadMixin,
    StateSaveLoadMixin,
)
from robo_orchard_lab.utils.state.save_profile import register_save_profile


@dataclass(frozen=True, slots=True)
class TreeStateSaveProfile:
    """Backward-compatible tree-shaped State API directory profile."""

    name: str = "tree"
    root_save_priority: int = 100
    load_priority: int = 0
    preserve_identity_during_apply: bool = False

    def save(
        self,
        state: State | StateSequence | StateList,
        *,
        path: str,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> None:
        """Save a State API root using the default tree-shaped layout."""
        if isinstance(state, State):
            self._save_state(state, path=path, protocol=protocol)
        elif isinstance(state, StateSequence):
            self._save_sequence(state, path=path, protocol=protocol)
        elif isinstance(state, StateList):
            self._save_list(state, path=path, protocol=protocol)
        else:
            raise TypeError(
                "TreeStateSaveProfile.save(...) expected State, "
                "StateSequence, or StateList. "
                f"Got {type(state).__name__}."
            )

    def load(
        self,
        path: str,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> State | StateSequence | StateList:
        """Load a State API root from the default tree-shaped layout."""
        type_config = _load_state_config(path)
        if type_config.class_type is State:
            return self._load_state(
                path,
                protocol=protocol,
                type_config=type_config,
            )
        if type_config.class_type is StateSequence:
            kind = type_config.load_kwargs.get("kind", "list")
            return self._load_sequence(
                path,
                protocol=protocol,
                kind=kind,
            )
        if type_config.class_type is StateList:
            return self._load_list(path, protocol=protocol)

        raise TypeError(
            "TreeStateSaveProfile.load(...) expected a State API root. "
            f"Got {type_config.class_type}."
        )

    def _save_state(
        self,
        state: State,
        *,
        path: str,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> None:
        _write_state_config(
            path=path,
            state_config=StateConfig(
                class_type=State,
                load_kwargs={"protocol": protocol},
                state_class_type=state.class_type,
                state_class_config=state.config,
            ),
        )

        if state.parameters is not None:
            np_tensors = {}
            pt_tensors = {}
            for k, v in state.parameters.items():
                if isinstance(v, np.ndarray):
                    np_tensors[k] = v
                elif isinstance(v, torch.Tensor):
                    pt_tensors[k] = v
                else:
                    raise ValueError(
                        f"Unsupported tensor type for parameter {k}: {type(v)}"
                    )
            if len(np_tensors) > 0:
                safetensors_numpy.save_file(
                    np_tensors,
                    os.path.join(path, "parameters.safetensors.np"),
                )
            if len(pt_tensors) > 0:
                safetensors_pytorch.save_file(
                    pt_tensors,
                    os.path.join(path, "parameters.safetensors.pt"),
                )

        _save_state_dict(
            path=path,
            name="state",
            states=state.state.copy(),
            protocol=protocol,
            hierarchical_save=state.hierarchical_save,
        )

    def _load_state(
        self,
        path: str,
        *,
        protocol: Literal["pickle", "cloudpickle"],
        type_config: StateConfig,
    ) -> State:
        state = _load_state_dict(path=path, name="state", protocol=protocol)

        return State(
            state=state,
            config=type_config.state_class_config,  # type: ignore
            parameters=_load_state_parameters(path),
            class_type=type_config.state_class_type,
            save_profile=self.name,
        )

    def _save_sequence(
        self,
        state: StateSequence,
        *,
        path: str,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> None:
        _write_state_config(
            path=path,
            state_config=StateConfig(
                class_type=StateSequence,
                load_kwargs={
                    "protocol": protocol,
                    "kind": state.kind,
                },
            ),
        )

        _save_state_dict(
            path=path,
            name="items",
            states={str(i): item for i, item in enumerate(state.items)},
            protocol=protocol,
            hierarchical_save=state.hierarchical_save,
        )

    def _load_sequence(
        self,
        path: str,
        *,
        protocol: Literal["pickle", "cloudpickle"],
        kind: Any,
    ) -> StateSequence:
        if kind == "list":
            sequence_kind: Literal["list", "tuple"] = "list"
        elif kind == "tuple":
            sequence_kind = "tuple"
        else:
            raise ValueError(f"Unsupported State sequence kind: {kind!r}.")

        data_dict = _load_indexed_state_dict(
            path=path,
            name="items",
            protocol=protocol,
            error_label="state sequence",
        )
        return StateSequence(
            kind=sequence_kind,
            items=[data_dict[i] for i in range(len(data_dict))],
            hierarchical_save=None,
            save_profile=self.name,
        )

    def _save_list(
        self,
        state: StateList,
        *,
        path: str,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> None:
        _write_state_config(
            path=path,
            state_config=StateConfig(
                class_type=StateList,
                load_kwargs={"protocol": protocol},
            ),
        )

        _save_state_dict(
            path=path,
            name="all",
            states={str(i): item for i, item in enumerate(state)},
            protocol=protocol,
            hierarchical_save=state.hierarchical_save,
        )

    def _load_list(
        self,
        path: str,
        *,
        protocol: Literal["pickle", "cloudpickle"],
    ) -> StateList:
        data_dict = _load_indexed_state_dict(
            path=path,
            name="all",
            protocol=protocol,
            error_label="state list",
        )
        return StateList(
            [data_dict[i] for i in range(len(data_dict))],
            hierarchical_save=None,
            save_profile=self.name,
        )

    def has_manifest(self, path: str) -> bool:
        """Return whether ``path`` has the default ``meta.json`` manifest."""
        return os.path.exists(os.path.join(path, META_FILE_NAME))

    def has_artifact(self, path: str) -> bool:
        """Return whether ``path`` is a default State API artifact."""
        return self.has_manifest(path)

    def load_path(
        self,
        path: str,
    ) -> (
        State
        | StateList
        | StateSequence
        | StateSaveLoadMixin
        | CustomizedSaveLoadMixin
    ):
        """Load a State API directory using its ``meta.json`` dispatcher."""
        type_config = _load_state_config(path)
        return type_config.class_type.load(path, **type_config.load_kwargs)

    def decode_payload_for_apply(
        self,
        obj: Any,
        *,
        active_paths: dict[int, str],
        path: str,
        constructable_state_apply_mode: ConstructableStateApplyMode,
    ) -> Any:
        """Decode tree-profile payloads for live ``load_state(...)`` paths."""
        from robo_orchard_lab.utils.state.conversion import (
            _decode_state_payload,
        )

        return _decode_state_payload(
            obj,
            path=path,
            materialize_nested_state=False,
            active_paths=active_paths,
            constructable_state_apply_mode=constructable_state_apply_mode,
            profile_aware_nested_state=True,
        )


_TREE_STATE_PROFILE = TreeStateSaveProfile()


def _write_state_config(path: str, state_config: StateConfig) -> None:
    with open(os.path.join(path, META_FILE_NAME), "w") as f:
        f.write(state_config.to_str(format="json", indent=2))


def _load_state_config(path: str) -> StateConfig:
    meta_path = os.path.join(path, META_FILE_NAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file {meta_path} does not exist.")
    with open(meta_path, "r") as f:
        meta = f.read()
    type_config = StateConfig.from_str(meta, format="json")
    if not isinstance(type_config, StateConfig):
        raise TypeError(f"Invalid type config: {type_config}")
    return type_config


def _load_state_parameters(path: str) -> dict[str, Any] | None:
    parameters = {}
    np_tensors_path = os.path.join(path, "parameters.safetensors.np")
    pt_tensors_path = os.path.join(path, "parameters.safetensors.pt")
    if os.path.exists(np_tensors_path):
        np_tensors = safetensors.safe_open(np_tensors_path, framework="numpy")
        for key in np_tensors.keys():
            parameters[key] = np_tensors.get_tensor(key)
    if os.path.exists(pt_tensors_path):
        pt_tensors = safetensors.safe_open(
            pt_tensors_path,
            framework="pt",
            device="cpu",
        )
        for key in pt_tensors.keys():
            parameters[key] = pt_tensors.get_tensor(key)
    if len(parameters) == 0:
        return None
    return parameters


def _load_indexed_state_dict(
    *,
    path: str,
    name: str,
    protocol: Literal["pickle", "cloudpickle"],
    error_label: str,
) -> dict[int, Any]:
    data_dict = _load_state_dict(
        path=path,
        name=name,
        protocol=protocol,
    )
    data_dict = {int(k): v for k, v in data_dict.items()}
    key_list = sorted(data_dict)
    if key_list != list(range(len(key_list))):
        raise ValueError(
            f"Missing keys in the {error_label}: {key_list}. "
            "The keys should be continuous from 0 to n-1."
        )
    return data_dict


def _save_state_dict(
    path: str,
    name: str,
    states: dict[str, Any],
    protocol: Literal["pickle", "cloudpickle"],
    hierarchical_save: bool | None,
) -> None:
    pickle_module = _protocol2module[protocol]
    folder = os.path.join(path, name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    remaining_states: dict[str, Any] = {}
    for k, v in states.items():
        if isinstance(v, (State, StateList, StateSequence)):
            if v.hierarchical_save is True or (
                v.hierarchical_save is None and hierarchical_save is True
            ):
                v = v.model_copy()
                v.hierarchical_save = True
                _save_state_api_root(
                    v,
                    path=os.path.join(folder, f"{k}"),
                    protocol=protocol,
                    inherited_save_profile=_TREE_STATE_PROFILE,
                )
                continue
        elif isinstance(v, CustomizedSaveLoadMixin):
            v.save(os.path.join(folder, f"{k}"), protocol=protocol)
            continue
        remaining_states[k] = v

    if hierarchical_save is True:
        for k, item in remaining_states.items():
            if isinstance(item, (State, StateList, StateSequence)):
                _save_state_api_root(
                    item,
                    path=os.path.join(folder, f"{k}"),
                    protocol=protocol,
                    inherited_save_profile=_TREE_STATE_PROFILE,
                )
            else:
                item_path = os.path.join(folder, f"{k}.pkl")
                with open(item_path, "wb") as f:
                    pickle_module.dump(item, f)
    else:
        if len(remaining_states) == 0:
            return
        state_path = os.path.join(path, f"{name}.pkl")
        with open(state_path, "wb") as f:
            pickle_module.dump(remaining_states, f)


def _load_state_dict(
    path: str,
    name: str,
    protocol: Literal["pickle", "cloudpickle"],
) -> dict[str, Any]:
    state = {}
    protocol_module = _protocol2module[protocol]
    if os.path.exists(os.path.join(path, f"{name}.pkl")):
        with open(os.path.join(path, f"{name}.pkl"), "rb") as f:
            state.update(protocol_module.load(f))

    state_folder = os.path.join(path, name)
    if os.path.exists(state_folder):
        for file in os.listdir(state_folder):
            if file.endswith(".pkl"):
                n = file[:-4]
                with open(os.path.join(state_folder, file), "rb") as f:
                    state[n] = protocol_module.load(f)
            elif os.path.isdir(os.path.join(state_folder, file)):
                from robo_orchard_lab.utils.state.save_profile import (
                    load_state_artifact,
                )

                n = file
                state[n] = load_state_artifact(
                    os.path.join(state_folder, file)
                )
            else:
                raise ValueError(
                    f"Unknown state file format: {file}. "
                    "Only .pkl files and folders are supported."
                )
    return state


__all__ = [
    "TreeStateSaveProfile",
    "_TREE_STATE_PROFILE",
    "_load_state_dict",
    "_save_state_dict",
]

register_save_profile(_TREE_STATE_PROFILE)
