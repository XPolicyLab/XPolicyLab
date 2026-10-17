# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark-neutral episode runner for user-defined embodied agents."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from roborsi.evaluation.protocol import (
    RESULT_SCHEMA_VERSION,
    RunProvenance,
    TaskIdentity,
)
from roborsi.evaluation.result import write_json_atomic

if TYPE_CHECKING:
    from roborsi.embodied_agent import EmbodiedAgent


@dataclass(frozen=True, slots=True)
class NativeScore:
    """A score read from the benchmark's own episode result."""

    success: bool
    score: float
    source: str | Path | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.success) is not bool:
            raise TypeError("native success must be a bool")
        if type(self.score) is bool:
            raise TypeError("native score must be a number")
        numeric_score = float(self.score)
        if not math.isfinite(numeric_score):
            raise ValueError("native score must be finite")
        object.__setattr__(self, "score", numeric_score)
        if not isinstance(self.details, Mapping):
            raise TypeError("native details must be a mapping")
        try:
            encoded = json.dumps(dict(self.details), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "native details must contain only finite JSON data"
            ) from exc
        decoded = json.loads(encoded)
        if decoded != self.details:
            raise ValueError("native details must use JSON object keys and values")
        object.__setattr__(self, "details", decoded)


class BenchmarkEpisode(Protocol):
    """One isolated benchmark episode; it owns reset, execution, and scoring."""

    def run(self, agent: EmbodiedAgent, output_dir: Path) -> NativeScore:
        """Execute and return the benchmark's native score."""
        ...


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One agent, benchmark, task, and optional seed to evaluate once."""

    agent_name: str
    benchmark_name: str
    task_name: str
    make_agent: Callable[[Path], EmbodiedAgent]
    make_episode: Callable[[], BenchmarkEpisode] | None = None
    seed: int | None = None
    provenance: RunProvenance = field(default_factory=RunProvenance)
    make_episode_for_task: Callable[[TaskIdentity], BenchmarkEpisode] | None = None

    def __post_init__(self) -> None:
        TaskIdentity(self.benchmark_name, self.task_name, self.seed)
        if not isinstance(self.provenance, RunProvenance):
            raise TypeError("provenance must be RunProvenance")
        if self.make_episode is None and self.make_episode_for_task is None:
            raise ValueError("an episode factory is required")
        if self.make_episode is not None and self.make_episode_for_task is not None:
            raise ValueError("provide only one episode factory")
        if self.seed is not None and self.make_episode_for_task is None:
            raise ValueError("seed requires make_episode_for_task")

    @property
    def task(self) -> TaskIdentity:
        return TaskIdentity(self.benchmark_name, self.task_name, self.seed)

    @property
    def key(self) -> str:
        base = f"{self.benchmark_name}/{self.agent_name}/{self.task_name}"
        return f"{base}/s{self.seed}" if self.seed is not None else base


def _preserve_native_score(source: str | Path, destination: Path) -> tuple[str, str]:
    """Copy a local native result before publishing the standardized result."""
    path = Path(source)
    if not path.is_absolute():
        path = destination / path
    if not path.is_file():
        raise FileNotFoundError(f"native score source does not exist: {path}")
    relative = Path("native") / path.name
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if path.resolve() != target.resolve():
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            shutil.copyfile(path, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    checksum = hashlib.sha256()
    with target.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk)
    return relative.as_posix(), checksum.hexdigest()


def evaluate_cases(
    cases: Sequence[EvaluationCase],
    output_dir: str | Path,
    *,
    max_workers: int = 1,
) -> list[dict[str, Any]]:
    """Run independent cases concurrently and save one native result per case.

    ``make_episode`` must create an isolated environment for each legacy case.
    Seeded cases use ``make_episode_for_task`` to pass the task identity. GPU
    benchmarks should use a process or cluster worker per case; a shared Isaac
    environment must not be driven by multiple Python threads.
    """
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    keys = [case.key for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("evaluation cases must have unique keys")
    if any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", part)
        for key in keys
        for part in key.split("/")
    ):
        raise ValueError("case names must contain only letters, digits, _ or -")
    root = Path(output_dir)
    existing = [key for key in keys if (root / key).exists()]
    if existing:
        raise FileExistsError(f"evaluation case output already exists: {existing[0]}")

    def one(case: EvaluationCase) -> dict[str, Any]:
        destination = root / case.key
        destination.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        common = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "case": case.key,
            "agent": case.agent_name,
            "task": case.task.as_dict(),
            "provenance": case.provenance.as_dict(),
        }
        try:
            agent = case.make_agent(destination)
            if case.make_episode_for_task is not None:
                episode = case.make_episode_for_task(case.task)
            else:
                assert case.make_episode is not None
                episode = case.make_episode()
            score = episode.run(agent, destination)
            native_source, native_sha256 = (
                _preserve_native_score(score.source, destination)
                if score.source is not None
                else (None, None)
            )
            record: dict[str, Any] = {
                **common,
                "status": "completed",
                "success": score.success,
                "score": score.score,
                "native_score_source": native_source,
                "native_score_sha256": native_sha256,
                "details": dict(score.details),
                "elapsed_s": time.monotonic() - started,
                "error": None,
            }
        except Exception as exc:
            record = {
                **common,
                "status": "error",
                "success": None,
                "score": None,
                "native_score_source": None,
                "native_score_sha256": None,
                "details": {},
                "elapsed_s": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        trace_manifest = destination / "trace" / "manifest.json"
        record["trace_manifest"] = (
            "trace/manifest.json" if trace_manifest.is_file() else None
        )
        write_json_atomic(destination / "result.json", record)
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(one, case): case.key for case in cases}
        records = {futures[future]: future.result() for future in as_completed(futures)}
    return [records[key] for key in keys]
