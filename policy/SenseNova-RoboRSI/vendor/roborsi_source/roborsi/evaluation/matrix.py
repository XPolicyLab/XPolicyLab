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

"""Compose Agent presets with Benchmark cases through existing batch runs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roborsi.agents.composition import (
    json_value,
    normalize_overrides,
    reject_secrets,
)
from roborsi.evaluation.batch import BatchCase, BatchEvaluator
from roborsi.evaluation.result import write_json_atomic
from roborsi.evaluation.store import ResultStore

_NAME = re.compile(r"[A-Za-z0-9_-]+\Z")


@dataclass(frozen=True)
class MatrixSelection:
    """One named preset and its optional deployment and Benchmark selection."""

    name: str
    config: str
    overrides: tuple[str, ...]
    image: str | None = None
    python: str | None = None
    benchmarks: tuple[str, ...] | None = None


def load_matrix(path: str | Path) -> tuple[MatrixSelection, ...]:
    """Read a matrix without generating or copying per-Benchmark Agent YAMLs."""
    import yaml

    path = Path(path).resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("invalid Agent matrix YAML") from exc
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "agents"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or not isinstance(data["agents"], dict)
        or not data["agents"]
    ):
        raise ValueError("Agent matrix requires schema_version: 1 and named agents")
    json_value(data)
    reject_secrets(data)
    result = []
    for name, row in data["agents"].items():
        if not _NAME.fullmatch(name):
            raise ValueError("matrix Agent names must use letters, numbers, '_' or '-'")
        if not isinstance(row, dict) or set(row) - {
            "config",
            "overrides",
            "image",
            "python",
            "benchmarks",
        }:
            raise ValueError(f"unknown Agent matrix fields: {name}")
        config = row.get("config")
        if not isinstance(config, str) or not config.strip():
            raise ValueError(f"matrix Agent requires config: {name}")
        for field in ("image", "python"):
            if field in row and (
                not isinstance(row[field], str) or not row[field].strip()
            ):
                raise ValueError(f"matrix {field} must be nonempty text")
        benchmarks = row.get("benchmarks")
        if benchmarks is not None and (
            not isinstance(benchmarks, list)
            or not benchmarks
            or any(not isinstance(b, str) or not b.strip() for b in benchmarks)
            or len(set(benchmarks)) != len(benchmarks)
        ):
            raise ValueError("matrix benchmarks must be a nonempty unique list")
        result.append(
            MatrixSelection(
                name=name,
                config=config
                if config.startswith("bundle:")
                else str((path.parent / config).absolute()),
                overrides=normalize_overrides(row.get("overrides", [])),
                image=row.get("image"),
                python=row.get("python"),
                benchmarks=tuple(benchmarks) if benchmarks is not None else None,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class MatrixAgent:
    """A validated launcher/evaluator applied to one selected case set."""

    name: str
    version: str
    evaluator: BatchEvaluator
    cases: tuple[BatchCase, ...]


@contextmanager
def _worker(store: ResultStore, run_id: str, *, create: bool = False) -> Iterator[None]:
    """Use the same nonblocking POSIX worker-lock convention as cases and RSI."""
    import fcntl

    if create:
        store.root.mkdir(parents=True, exist_ok=True)
    with (store.root / f".{run_id}.matrix.lock").open(
        "a+b" if create else "rb"
    ) as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"matrix has an active worker: {run_id}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def matrix_status(store: ResultStore, run_id: str) -> dict[str, Any]:
    """Report the parent state and ordinary case states for every Agent run."""
    plan = store.load_batch_plan(run_id)
    if plan.get("kind") != "agent_matrix":
        raise ValueError("run is not an Agent matrix")
    try:
        report = json.loads((store.root / run_id / "matrix.json").read_text())
    except FileNotFoundError:
        report = {"run_id": run_id, "status": "running"}
    if report["status"] == "running":
        try:
            with _worker(store, run_id):
                report["status"] = "interrupted"
        except ValueError:
            pass
    return {
        "run_id": run_id,
        "status": report["status"],
        "agents": {
            name: {
                "run_id": child,
                "cases": {
                    case: store.status(child, case).as_dict()
                    for case in store.list_cases(child)
                },
            }
            for name, child in plan["runs"].items()
        },
    }


def run_matrix(
    store: ResultStore,
    run_id: str,
    agents: Sequence[MatrixAgent],
    *,
    max_concurrent: int = 1,
    timeout_s: float = 30.0,
    case_timeout_s: float = 3600.0,
    resume_from: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Preflight every Agent, then run sequential batches with bounded case concurrency.

    Each Agent keeps a separate ordinary batch run. Resume locates the latest
    started batch for each Agent, including interrupted multi-stage resumes.
    """
    if not _NAME.fullmatch(run_id) or (
        resume_from is not None and not _NAME.fullmatch(resume_from)
    ):
        raise ValueError("matrix run IDs must be path-safe identifiers")
    if (
        not agents
        or len({a.name for a in agents}) != len(agents)
        or any(not _NAME.fullmatch(a.name) for a in agents)
    ):
        raise ValueError("matrix agents must have unique path-safe names")
    if (store.root / run_id).exists():
        raise FileExistsError(f"run already exists: {run_id}")
    options = {
        "max_concurrent": max_concurrent,
        "timeout_s": timeout_s,
        "case_timeout_s": case_timeout_s,
    }
    runs = {a.name: f"{run_id}--{a.name}" for a in agents}
    expected = {
        a.name: a.evaluator.preflight(runs[a.name], a.cases, a.version, **options)
        for a in agents
    }
    previous = {}
    ancestor = resume_from
    seen = {run_id}
    while ancestor is not None:
        if ancestor in seen:
            raise ValueError("matrix resume lineage contains a cycle")
        seen.add(ancestor)
        with _worker(store, ancestor):
            plan = store.load_batch_plan(ancestor)
        if plan.get("kind") != "agent_matrix" or plan.get("agents") != expected:
            raise ValueError("matrix inputs differ from previous run")
        if plan.get("runs") != {a.name: f"{ancestor}--{a.name}" for a in agents}:
            raise ValueError("invalid matrix child run IDs")
        for agent in agents:
            candidate = plan["runs"][agent.name]
            if agent.name not in previous and (store.root / candidate).exists():
                previous[agent.name] = candidate
        ancestor = plan.get("resume_from")
        if ancestor is not None and (
            not isinstance(ancestor, str) or not _NAME.fullmatch(ancestor)
        ):
            raise ValueError("invalid matrix resume lineage")
    for agent in agents:
        if agent.name in previous:
            agent.evaluator.preflight(
                runs[agent.name],
                agent.cases,
                agent.version,
                resume_from=previous[agent.name],
                **options,
            )
    plan = {
        "schema_version": 1,
        "kind": "agent_matrix",
        "agents": expected,
        "runs": runs,
        "resume_from": resume_from,
    }
    if dry_run:
        return {
            "run_id": run_id,
            "total_cases": sum(len(a.cases) for a in agents),
            **plan,
        }
    with ExitStack() as workers:
        workers.enter_context(_worker(store, run_id, create=True))
        for old_run in sorted(seen - {run_id}):
            workers.enter_context(_worker(store, old_run))
        directory = store.reserve_batch_run(run_id, plan)
        report = {
            "run_id": run_id,
            "resume_from": resume_from,
            "status": "running",
            "runs": runs,
            "agents": {},
        }
        write_json_atomic(directory / "matrix.json", report)
        try:
            for agent in agents:
                batch = agent.evaluator.run(
                    runs[agent.name],
                    agent.cases,
                    agent.version,
                    resume_from=previous.get(agent.name),
                    **options,
                )
                report["agents"][agent.name] = {
                    "agent_version": agent.version,
                    **batch.as_dict(),
                }
                write_json_atomic(directory / "matrix.json", report)
            report["status"] = "completed"
        except BaseException:
            report["status"] = "interrupted"
            raise
        finally:
            write_json_atomic(directory / "matrix.json", report)
        return report
