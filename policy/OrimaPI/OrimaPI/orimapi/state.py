# coding=utf-8
"""Owned observation snapshots and per-episode working memory."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


def _json_convert(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_convert(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_convert(child) for child in value]
    return value


def json_value(value: Any) -> Any:
    """Copy JSON-compatible data, including array/scalar values from a backend."""
    return json.loads(json.dumps(_json_convert(value), allow_nan=False))


def image_media_type(image: bytes) -> str:
    if not isinstance(image, bytes):
        raise ValueError("Images must be encoded PNG or JPEG bytes")
    if image.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    raise ValueError("Images must be encoded PNG or JPEG bytes")


def image_data_url(image: bytes) -> str:
    if isinstance(image, RGBImage):
        image = image.to_png()
    return f"data:{image_media_type(image)};base64,{base64.b64encode(image).decode()}"


@dataclass(frozen=True)
class RGBImage:
    """Immutable decoded RGB snapshot for local inference, without PNG round trips."""

    pixels: bytes
    width: int
    height: int

    def __post_init__(self):
        if (
            type(self.width) is not int
            or type(self.height) is not int
            or min(self.width, self.height) < 1
            or not isinstance(self.pixels, bytes)
            or len(self.pixels) != self.width * self.height * 3
        ):
            raise ValueError("RGB image requires exact width*height*3 immutable bytes")

    def to_png(self):
        import io
        from PIL import Image

        stream = io.BytesIO()
        Image.frombytes("RGB", (self.width, self.height), self.pixels).save(
            stream, format="PNG"
        )
        return stream.getvalue()

    def __deepcopy__(self, memo):
        return self


class Mode(StrEnum):
    DIRECT = "direct"
    ONCE = "once"
    CLOSED_LOOP = "closed_loop"


class DecisionKind(StrEnum):
    KEEP = "keep"
    REPLACE = "replace"
    RETRY = "retry"


@dataclass(frozen=True)
class Observation:
    """Public robot inputs; step counts actions actually executed by the simulator.

    The future simulator adapter must supply only public camera/proprioception
    data here. Evaluator rewards and hidden object poses are not policy inputs.
    """

    env_id: int
    step: int
    instruction: str
    images: dict[str, bytes | RGBImage]
    robot_state: Any

    def __post_init__(self) -> None:
        for name in ("env_id", "step"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("An original task instruction is required")
        if not isinstance(self.images, dict) or not self.images:
            raise ValueError("At least one camera image is required")
        for name, image in self.images.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Camera names must be nonempty strings")
            if not isinstance(image, RGBImage):
                image_media_type(image)
        object.__setattr__(self, "images", dict(self.images))
        object.__setattr__(self, "robot_state", json_value(self.robot_state))

    def __deepcopy__(self, memo):
        clone = object.__new__(type(self))
        object.__setattr__(clone, "env_id", self.env_id)
        object.__setattr__(clone, "step", self.step)
        object.__setattr__(clone, "instruction", self.instruction)
        object.__setattr__(clone, "images", dict(self.images))
        state = self.robot_state
        if isinstance(state, dict):
            state = {
                k: list(v) if isinstance(v, list) else deepcopy(v, memo)
                for k, v in state.items()
            }
        else:
            state = deepcopy(state, memo)
        object.__setattr__(clone, "robot_state", state)
        memo[id(self)] = clone
        return clone


@dataclass(frozen=True)
class Decision:
    decision: DecisionKind
    subgoal: str
    reason: str
    previous_subgoal_completed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", DecisionKind(self.decision))
        for name in ("subgoal", "reason"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Decision {name} must be a nonempty string")
        if type(self.previous_subgoal_completed) is not bool:
            raise ValueError("previous_subgoal_completed must be a boolean")
        if self.previous_subgoal_completed and self.decision != DecisionKind.REPLACE:
            raise ValueError("Only replace can mark the previous subgoal completed")

    @classmethod
    def from_dict(cls, value: Any) -> Decision:
        required = {"decision", "subgoal", "reason"}
        if (
            not isinstance(value, dict)
            or not required <= value.keys()
            or value.keys() - required - {"previous_subgoal_completed"}
        ):
            raise ValueError("Unexpected supervisor decision fields")
        return cls(**value)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "subgoal": self.subgoal,
            "reason": self.reason,
            "previous_subgoal_completed": self.previous_subgoal_completed,
        }


@dataclass(frozen=True)
class PolicySelection:
    policy_id: str
    reason: str

    def __post_init__(self) -> None:
        for value in (self.policy_id, self.reason):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Policy selection requires nonempty policy_id and reason"
                )

    @classmethod
    def from_dict(cls, value: Any) -> PolicySelection:
        if not isinstance(value, dict) or set(value) != {"policy_id", "reason"}:
            raise ValueError("Unexpected policy selection fields")
        return cls(**value)

    def to_dict(self) -> dict:
        return {"policy_id": self.policy_id, "reason": self.reason}


@dataclass(frozen=True)
class SupervisorRequest:
    """One request shape: candidates selects a policy; None requests a subgoal."""

    request_id: str
    episode_id: str
    plan_version: int
    observation: Observation
    memory: dict
    policy: dict | None = None
    candidates: dict | None = None


# Preserve the old constructor/import without a second request class.
PolicySelectionRequest = SupervisorRequest
