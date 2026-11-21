"""JSON-serializable contracts shared by agents and simulator workers."""

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class Observation:
    episode_id: str
    step: int
    instruction: str
    state: dict[str, Any]
    # Camera name -> image data URL. This keeps remote traces self-contained.
    images: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Action:
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Transition:
    observation: Observation
    reward: float = 0.0
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    # Camera name -> MP4 data URL of the motion produced by this action.
    # Still images stay on the observation. Video is not sent to the model.
    videos: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Transition":
        data = dict(data)
        data["observation"] = Observation(**data["observation"])
        return cls(**data)


@dataclass
class Decision:
    action: Action
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str | None = None


class Environment(Protocol):
    """Only actor-visible observations belong here; native evaluation stays in the worker.

    execute must reject stale episode/step pairs and return a newer observation.
    Motion units, frames and limits must be documented in each tool schema.
    """

    name: str

    def tools(self) -> list[dict]: ...
    def reset(self, task: str, seed: int) -> Observation: ...
    def execute(self, action: Action, episode_id: str, step: int) -> Transition: ...
    def close(self) -> None: ...


class Agent(Protocol):
    def decide(self, request: dict) -> Decision: ...
