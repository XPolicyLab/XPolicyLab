# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the VLM subgoal strategy."""

from __future__ import annotations

from collections.abc import Mapping
from vlm_orchestrator.strategies.base import OrchestrationStrategy, SessionState
from vlm_orchestrator.vlm.api import (
    apply_coordinate_variables,
    extract_arm,
    extract_coordinates_from_text,
    infer_atomic_action,
    normalize_atomic_action,
    normalize_coordinates,
    vla_coordinate_payload,
)


def parse_subgoals(value) -> tuple[
    list[str], bool, list[str], list[str], list[list[list[int]]], list[str]
]:
    """Normalize VLM decomposition results into parallel arrays."""
    if hasattr(value, "subgoals"):
        raw = getattr(value, "subgoals", [])
        ordered = bool(getattr(value, "ordered", True))
        adapters = list(getattr(value, "adapter_ids", []) or [])
        actions = list(getattr(value, "atomic_actions", []) or [])
        coordinates = list(getattr(value, "coordinates", []) or [])
        arms = list(getattr(value, "arms", []) or [])
    elif isinstance(value, Mapping):
        raw = value.get("subgoals", [])
        ordered = bool(value.get("ordered", True))
        adapters = []
        actions = []
        coordinates = []
        arms = []
    else:
        raw, ordered = value, True
        adapters = actions = coordinates = arms = []
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    result: list[str] = []
    out_adapters: list[str] = []
    out_actions: list[str] = []
    out_coordinates: list[list[list[int]]] = []
    out_arms: list[str] = []
    for index, item in enumerate(raw):
        mapping = item if isinstance(item, Mapping) else {}
        if mapping:
            text = mapping.get("subtask_prompt") or mapping.get("text") or mapping.get("command")
            if not text:
                text = " ".join(str(mapping.get(key, "")).strip() for key in ("subtask_description", "combination_description") if mapping.get(key))
            text = text or mapping.get("instruction", "")
            adapter = mapping.get("adapter_id", "")
            action = mapping.get("atomic_action", "")
            raw_points = mapping.get("coordinates", mapping.get("coordinate"))
            raw_arm = mapping.get("arm", "")
        else:
            text, adapter, action = item, "", ""
            raw_points = coordinates[index] if index < len(coordinates) else None
            raw_arm = arms[index] if index < len(arms) else ""
        text = str(text or "").strip()
        if not text or text.lower().startswith(("motion:", "point:", "trace:")):
            continue
        points = normalize_coordinates(raw_points) or extract_coordinates_from_text(text)
        arm = extract_arm(raw_arm, text)
        text = apply_coordinate_variables(text, points, arm)
        result.append(text)
        out_adapters.append(str(adapter or "").strip())
        normalized = normalize_atomic_action(action)
        out_actions.append(normalized or infer_atomic_action(text))
        out_coordinates.append(points)
        out_arms.append(arm)
    return result, ordered, out_adapters, out_actions, out_coordinates, out_arms


class SubgoalBaseStrategy(OrchestrationStrategy):
    @staticmethod
    def _current_grounding(state: SessionState) -> tuple[list[list[int]], str]:
        index = state.current_subgoal_idx
        points = state.subgoal_coordinates[index] if 0 <= index < len(state.subgoal_coordinates) else []
        arm = state.subgoal_arms[index] if 0 <= index < len(state.subgoal_arms) else ""
        text = state.subgoals[index] if 0 <= index < len(state.subgoals) else (state.original_instruction or "")
        return normalize_coordinates(points) or extract_coordinates_from_text(text), extract_arm(arm, text)

    @classmethod
    def _attach_coordinates(cls, obs: dict, state: SessionState) -> dict:
        result = dict(obs)
        points, _ = cls._current_grounding(state)
        payload = vla_coordinate_payload(points)
        if payload is None:
            result.pop("coordinates", None)
        else:
            result["coordinates"] = payload
        result.pop("arm", None)
        return result

    def _set_subgoal_prompt(self, obs: dict, state: SessionState, phase: str = "subgoal") -> dict:
        if 0 <= state.current_subgoal_idx < len(state.subgoals):
            prompt = state.subgoals[state.current_subgoal_idx]
        else:
            prompt = state.original_instruction or ""
        prompt, metadata = self.ctx.transform_prompt(prompt, obs, state, phase, "vla")
        if metadata:
            state.transform_metadata.extend(metadata)
            state.log({"type": "transform", "phase": phase, "items": metadata})
        state.vla_prompt = state.rewritten_instruction = prompt
        return self._attach_coordinates(self.ctx.set_prompt(obs, prompt), state)
