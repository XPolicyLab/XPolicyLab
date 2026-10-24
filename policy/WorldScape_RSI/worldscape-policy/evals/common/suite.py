from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationTask:
    """One benchmark task and its backend-specific construction metadata."""

    task_id: str
    instruction: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.instruction:
            raise ValueError("instruction must be non-empty")


@dataclass(frozen=True)
class EpisodeSpec:
    task: EvaluationTask
    episode_index: int
    seed: int

    @property
    def episode_id(self) -> str:
        return f"{self.task.task_id}-{self.episode_index:04d}"


@dataclass(frozen=True)
class TaskSuite:
    """A deterministic task/episode schedule."""

    tasks: tuple[EvaluationTask, ...]
    episodes_per_task: int = 1
    seed: int = 0
    suite_id: str = "custom"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        tasks: Iterable[EvaluationTask],
        *,
        episodes_per_task: int = 1,
        seed: int = 0,
        suite_id: str = "custom",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        values = tuple(tasks)
        if not values:
            raise ValueError("A task suite must contain at least one task")
        if episodes_per_task < 1:
            raise ValueError("episodes_per_task must be positive")
        ids = [task.task_id for task in values]
        if len(ids) != len(set(ids)):
            raise ValueError("Task IDs must be unique")
        object.__setattr__(self, "tasks", values)
        object.__setattr__(self, "episodes_per_task", episodes_per_task)
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "suite_id", str(suite_id))
        object.__setattr__(self, "metadata", dict(metadata or {}))

    def episodes(self) -> Iterator[EpisodeSpec]:
        offset = 0
        for task in self.tasks:
            for episode_index in range(self.episodes_per_task):
                yield EpisodeSpec(task, episode_index, self.seed + offset)
                offset += 1

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> TaskSuite:
        raw_suite = config.get("suite", {})
        if not isinstance(raw_suite, Mapping):
            raise TypeError("Evaluation config suite must be a mapping")
        suite_metadata = raw_suite.get("metadata", {})
        if not isinstance(suite_metadata, Mapping):
            raise TypeError("Evaluation suite metadata must be a mapping")
        raw_tasks = raw_suite.get("tasks", config.get("tasks"))
        if not isinstance(raw_tasks, list):
            raise TypeError("Evaluation config must contain a tasks or suite.tasks list")
        tasks = []
        for raw in raw_tasks:
            if not isinstance(raw, Mapping):
                raise TypeError("Each task config must be a mapping")
            task_id = raw.get("id", raw.get("task_id"))
            instruction = raw.get("instruction")
            metadata = raw.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise TypeError("Task metadata must be a mapping")
            tasks.append(
                EvaluationTask(
                    task_id=str(task_id or ""),
                    instruction=str(instruction or ""),
                    metadata=dict(metadata),
                )
            )
        return cls(
            tasks,
            episodes_per_task=int(config.get("episodes_per_task", 1)),
            seed=int(config.get("seed", 0)),
            suite_id=str(raw_suite.get("id", raw_suite.get("name", "custom"))),
            metadata=dict(suite_metadata),
        )
