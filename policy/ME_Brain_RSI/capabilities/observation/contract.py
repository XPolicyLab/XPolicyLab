"""Exact wire contract for ``insert_tubes`` code-policy controller."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral
from typing import Any

TASK_ID = "insert_tubes"
OFFICIAL_MAXIMUM_ACTIONS = 500
OPEN_GRIPPER = 1.0
CLOSED_GRIPPER = 0.0
ROBOT_KEYS = (
    "left_arm_joint_state",
    "right_arm_joint_state",
    "left_ee_joint_state",
    "right_ee_joint_state",
)
ROBOT_KEY_SET = frozenset(ROBOT_KEYS)
ENVELOPE_KEYS = frozenset({"env_idx", "skill_choice", "stages", "robot"})
STAGE_KEYS = frozenset({"name", "steps", "target"})
skill_choices = frozenset({"direct", "partial", "infeasible"})


class InsertTubesEnvelopeError(ValueError):
    pass


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise InsertTubesEnvelopeError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise InsertTubesEnvelopeError(f"{name} must be finite") from error
    if not isfinite(result):
        raise InsertTubesEnvelopeError(f"{name} must be finite")
    return result


def _index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise InsertTubesEnvelopeError(f"{name} must be a non-negative integer")
    return int(value)


def _vector(value: object, size: int, name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise InsertTubesEnvelopeError(f"{name} must be numeric")
    try:
        items = list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise InsertTubesEnvelopeError(f"{name} must be numeric") from error
    if len(items) != size:
        raise InsertTubesEnvelopeError(f"{name} must contain {size} values")
    return [_number(item, name) for item in items]


def _action(value: object, name: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or set(value) != ROBOT_KEY_SET:
        raise InsertTubesEnvelopeError(f"{name} keys must be exactly {sorted(ROBOT_KEY_SET)!r}")
    return {
        "left_arm_joint_state": _vector(value["left_arm_joint_state"], 6, name),
        "right_arm_joint_state": _vector(value["right_arm_joint_state"], 6, name),
        "left_ee_joint_state": _vector(value["left_ee_joint_state"], 1, name),
        "right_ee_joint_state": _vector(value["right_ee_joint_state"], 1, name),
    }


def normalize_insert_tubes_envelope(value: Mapping[str, object]) -> dict[str, Any]:
    if set(value) != ENVELOPE_KEYS:
        raise InsertTubesEnvelopeError(
            f"insert_tubes envelope keys must be exactly {sorted(ENVELOPE_KEYS)!r}"
        )
    skill_choice = value["skill_choice"]
    if skill_choice not in skill_choices:
        raise InsertTubesEnvelopeError(f"skill_choice must be one of {sorted(skill_choices)!r}")
    raw_stages = value["stages"]
    if isinstance(raw_stages, (str, bytes, bytearray, Mapping)):
        raise InsertTubesEnvelopeError("stages must be a sequence")
    stages = []
    for index, raw_stage in enumerate(raw_stages):  # type: ignore[union-attr]
        if not isinstance(raw_stage, Mapping) or set(raw_stage) != STAGE_KEYS:
            raise InsertTubesEnvelopeError(
                f"stages[{index}] keys must be exactly {sorted(STAGE_KEYS)!r}"
            )
        name = raw_stage["name"]
        if not isinstance(name, str) or not name:
            raise InsertTubesEnvelopeError("stage name must be non-empty")
        steps = _index(raw_stage["steps"], f"stages[{index}].steps")
        if steps < 1:
            raise InsertTubesEnvelopeError("stage steps must be positive")
        stages.append(
            {
                "name": name,
                "steps": steps,
                "target": _action(raw_stage["target"], f"stages[{index}].target"),
            }
        )
    if (skill_choice == "infeasible") != (not stages):
        raise InsertTubesEnvelopeError("only infeasible skill_choice may have no stages")
    if sum(stage["steps"] for stage in stages) > OFFICIAL_MAXIMUM_ACTIONS:
        raise InsertTubesEnvelopeError("stage plan exceeds the native 500-action limit")
    return {
        "env_idx": _index(value["env_idx"], "env_idx"),
        "skill_choice": skill_choice,
        "stages": stages,
        "robot": _action(value["robot"], "robot"),
    }


def validate_insert_tubes_action(value: Mapping[str, object]) -> dict[str, list[float]]:
    return _action(value, "insert_tubes action")
