"""Configuration for one EmbodiedRSI Test session."""

from dataclasses import dataclass
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent
SESSION_PROTOCOL = "workspace-session-v1"


def setting(name, default=None):
    return os.environ.get("EMBODIEDRSI_" + name, default)


@dataclass(frozen=True)
class Simulator:
    backend: str
    api_surface: str
    interpreter: str
    env_task: str


@dataclass(frozen=True)
class Agent:
    preset: str
    model: str
    reasoning_effort: str
    timeout_sec: float | None
    isolation: str
    image: str
    network: str
    cpus: int
    memory_mb: int


@dataclass(frozen=True)
class Test:
    scenes: int
    submission_timeout_sec: int
    generation_timeout_sec: float | None
    budget: int
    feedback_fps: int
    expose_success_feedback: bool


@dataclass(frozen=True)
class TaskConfig:
    root: Path
    name: str
    description: str
    simulator: Simulator
    agent: Agent
    test: Test
    workspace_source: str = "inputs"
    protocol: str = SESSION_PROTOCOL

    @property
    def feedback_fps(self):
        return self.test.feedback_fps


def config_from_dict(data):
    data = dict(data)
    data["root"] = Path(data["root"])
    data["simulator"] = Simulator(**data["simulator"])
    data["agent"] = Agent(**data["agent"])
    data["test"] = Test(**data["test"])
    cfg = TaskConfig(**data)
    if cfg.protocol != SESSION_PROTOCOL or cfg.agent.preset != "codex":
        raise ValueError("EmbodiedRSI requires the Codex workspace-session-v1 protocol")
    return cfg
