"""Incremental joint controller for three-tube insertion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contract import ROBOT_KEYS, TASK_ID, normalize_insert_tubes_envelope


@dataclass(slots=True)
class InsertTubesSlot:
    skill_choice: str
    stages: tuple[dict[str, Any], ...]
    stage_index: int = 0
    stage_actions: int = 0
    stage_start: dict[str, list[float]] | None = None


def _copy_action(robot: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    return {key: [float(item) for item in robot[key]] for key in ROBOT_KEYS}


class InsertTubesController:
    def __init__(self, model_cfg: Mapping[str, object] | None = None) -> None:
        cfg = {} if model_cfg is None else model_cfg
        if cfg.get("task_name") != TASK_ID:
            raise ValueError(f"task_name must be {TASK_ID!r}")
        if cfg.get("action_type") != "joint":
            raise ValueError("insert_tubes policy requires action_type='joint'")
        if not isinstance(cfg.get("env_cfg_type"), str) or not cfg["env_cfg_type"]:
            raise ValueError("env_cfg_type must be a non-empty string")
        self._slots: dict[int, InsertTubesSlot] = {}
        self._observations: dict[int, dict[str, Any]] = {}
        self._latest_batch_indices: tuple[int, ...] = ()
        self._single_ready = False

    def reset(self) -> None:
        self._slots.clear()
        self._observations.clear()
        self._latest_batch_indices = ()
        self._single_ready = False

    def prepare_case(self, case_meta=None) -> None:
        self.reset()

    def on_trial_end(self, result=None) -> None:
        self.reset()

    def update_obs(self, obs: Mapping[str, object]) -> None:
        normalized = normalize_insert_tubes_envelope(obs)
        if normalized["env_idx"] != 0:
            raise ValueError("single-environment observation must use env_idx=0")
        self._ingest(normalized)
        self._latest_batch_indices = ()
        self._single_ready = True

    def update_obs_batch(self, obs_list: Sequence[Mapping[str, object]]) -> None:
        normalized = tuple(normalize_insert_tubes_envelope(item) for item in obs_list)
        indices = tuple(int(item["env_idx"]) for item in normalized)
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("obs_list must contain unique environments")
        for observation in normalized:
            self._ingest(observation)
        self._latest_batch_indices = indices
        self._single_ready = False

    def get_action(self) -> list[dict[str, list[float]]]:
        if not self._single_ready:
            raise RuntimeError("get_action requires update_obs")
        self._single_ready = False
        return [self._action(0)]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict[str, list[float]]]]:
        if not self._latest_batch_indices:
            raise RuntimeError("get_action_batch requires update_obs_batch")
        indices = (
            self._latest_batch_indices
            if env_idx_list is None
            else tuple(int(item) for item in env_idx_list)
        )
        if set(indices) - set(self._latest_batch_indices):
            raise ValueError("requested environment is absent from latest observation")
        self._latest_batch_indices = ()
        return [[self._action(index)] for index in indices]

    def _ingest(self, observation: dict[str, Any]) -> None:
        env_idx = int(observation["env_idx"])
        self._observations[env_idx] = observation
        slot = self._slots.get(env_idx)
        if slot is None:
            self._slots[env_idx] = InsertTubesSlot(
                skill_choice=str(observation["skill_choice"]),
                stages=tuple(observation["stages"]),
                stage_start=_copy_action(observation["robot"]),
            )
            return
        slot.stages = tuple(observation["stages"])
        if slot.stage_index < len(slot.stages) and slot.stage_actions >= int(
            slot.stages[slot.stage_index]["steps"]
        ):
            slot.stage_index += 1
            slot.stage_actions = 0
            slot.stage_start = _copy_action(observation["robot"])

    def _action(self, env_idx: int) -> dict[str, list[float]]:
        observation = self._observations[env_idx]
        slot = self._slots[env_idx]
        if slot.skill_choice == "infeasible" or slot.stage_index >= len(slot.stages):
            return _copy_action(observation["robot"])
        stage = slot.stages[slot.stage_index]
        start = slot.stage_start or _copy_action(observation["robot"])
        fraction = min(1.0, (slot.stage_actions + 1) / int(stage["steps"]))
        target = stage["target"]
        action = {
            key: [
                float(left) + fraction * (float(right) - float(left))
                for left, right in zip(start[key], target[key], strict=True)
            ]
            for key in ROBOT_KEYS
        }
        slot.stage_actions += 1
        return action


__all__ = ["InsertTubesController"]
