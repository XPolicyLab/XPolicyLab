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

"""Expand a locked benchmark matrix and run independent RoboRSI cases."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from roborsi.benchmarks.registry import BenchmarkRegistry
from roborsi.evaluation.messages import EvaluationResult, Provenance
from roborsi.evaluation.orchestration import CaseOrchestrator
from roborsi.evaluation.protocol import TaskIdentity
from roborsi.evaluation.store import ResultStore


@dataclass(frozen=True, slots=True)
class BatchCase:
    """One declared benchmark task and seed."""

    benchmark: str
    task_id: str
    seed: int

    def __post_init__(self) -> None:
        TaskIdentity(self.benchmark, self.task_id, self.seed)

    @property
    def case_id(self) -> str:
        """Return a path-safe stable ID without losing native task spelling."""
        key = json.dumps(
            [self.benchmark, self.task_id, self.seed], ensure_ascii=False
        ).encode("utf-8")
        return "case-" + hashlib.sha256(key).hexdigest()


def expand_cases(
    registry: BenchmarkRegistry,
    benchmarks: Sequence[str],
    *,
    task_ids: Mapping[str, Sequence[str]] | None = None,
    seeds: Sequence[int] | None = None,
) -> tuple[BatchCase, ...]:
    """Resolve requested cells and reject undeclared task/seed combinations."""
    if not benchmarks or len(set(benchmarks)) != len(benchmarks):
        raise ValueError("benchmarks must be non-empty and unique")
    if task_ids is not None and set(task_ids) - set(benchmarks):
        raise ValueError("task filter names an unselected benchmark")
    if seeds is not None:
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be non-empty and unique")
        for seed in seeds:
            TaskIdentity("validation", "validation", seed)
    cases: list[BatchCase] = []
    for benchmark in benchmarks:
        lock = registry.lock(benchmark)
        declared = lock.list_tasks()
        selected_tasks = (
            tuple(task_ids[benchmark])
            if task_ids is not None and benchmark in task_ids
            else tuple(dict.fromkeys(item.task_id for item in declared))
        )
        if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks):
            raise ValueError(f"task IDs for {benchmark} must be non-empty and unique")
        for task_id in selected_tasks:
            available = tuple(item.seed for item in declared if item.task_id == task_id)
            if not available:
                raise ValueError(f"unknown task for {benchmark}: {task_id}")
            for seed in seeds if seeds is not None else available:
                lock.task(task_id, seed)
                cases.append(BatchCase(benchmark, task_id, seed))
    return tuple(cases)


@dataclass(frozen=True, slots=True)
class CaseRecord:
    """A published result tied to one matrix cell."""

    case: BatchCase
    result: EvaluationResult
    source_run_id: str

    def as_dict(self) -> dict:
        return {
            "case_id": self.case.case_id,
            "source_run_id": self.source_run_id,
            "result": self.result.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Counts and manifest-defined aggregate for one benchmark only."""

    planned: int
    scored: int
    successes: int
    task_failures: int
    agent_errors: int
    infrastructure_errors: int
    cancelled: int
    error_denominator: str
    aggregation: str
    success_rate: float | None
    mean_native_score: float | None
    aggregate_score: float | None

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BatchReport:
    """Published cases and separate manifest aggregates by benchmark."""

    run_id: str
    cases: tuple[CaseRecord, ...]
    summaries: dict[str, BenchmarkSummary]
    resume_from: str | None = None

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "resume_from": self.resume_from,
            "cases": [record.as_dict() for record in self.cases],
            "benchmarks": {
                name: summary.as_dict() for name, summary in self.summaries.items()
            },
        }


def summarize_cases(
    records: Sequence[CaseRecord], evaluation: dict
) -> BenchmarkSummary:
    """Compute one Benchmark's declared aggregate from its published cases."""
    planned = len(records)
    scored = sum(record.result.score is not None for record in records)
    successes = sum(record.result.status == "success" for record in records)
    task_failures = sum(record.result.status == "task_failure" for record in records)
    agent_errors = sum(record.result.status == "agent_error" for record in records)
    infrastructure_errors = sum(
        record.result.status == "infrastructure_error" for record in records
    )
    cancelled = sum(record.result.status == "cancelled" for record in records)
    denominator = planned if evaluation["error_denominator"] == "expected" else scored
    success_rate = successes / denominator if denominator else None
    native_total = sum(record.result.score or 0.0 for record in records)
    mean_native_score = native_total / denominator if denominator else None
    aggregation = evaluation["aggregation"]
    if aggregation == "case_mean":
        aggregate_score = mean_native_score
    else:
        by_task: dict[str, list[CaseRecord]] = defaultdict(list)
        for record in records:
            aggregation_task = record.result.task.native.get("aggregation_task")
            key = (
                aggregation_task
                if isinstance(aggregation_task, str) and aggregation_task
                else record.case.task_id
            )
            by_task[key].append(record)
        task_scores = []
        for task_records in by_task.values():
            task_denominator = (
                len(task_records)
                if evaluation["error_denominator"] == "expected"
                else sum(record.result.score is not None for record in task_records)
            )
            if task_denominator:
                task_scores.append(
                    sum(
                        (record.result.score or 0.0)
                        if aggregation == "task_weighted_mean"
                        else float(record.result.status == "success")
                        for record in task_records
                    )
                    / task_denominator
                )
        aggregate_score = sum(task_scores) / len(task_scores) if task_scores else None
    return BenchmarkSummary(
        planned=planned,
        scored=scored,
        successes=successes,
        task_failures=task_failures,
        agent_errors=agent_errors,
        infrastructure_errors=infrastructure_errors,
        cancelled=cancelled,
        error_denominator=evaluation["error_denominator"],
        aggregation=evaluation["aggregation"],
        success_rate=success_rate,
        mean_native_score=mean_native_score,
        aggregate_score=aggregate_score,
    )


class BatchEvaluator:
    """Bound concurrent case orchestration while preserving each case result."""

    def __init__(
        self,
        registry: BenchmarkRegistry,
        store: ResultStore,
        orchestrator: CaseOrchestrator,
    ) -> None:
        self.registry = registry
        self.store = store
        self.orchestrator = orchestrator

    def _lineage(self, resume_from: str | None, expected: dict) -> tuple[str, ...]:
        if resume_from is None:
            return ()
        try:
            latest = self.store.load_batch_plan(resume_from)
        except FileNotFoundError as exc:
            raise ValueError(
                f"cannot resume a run without a batch plan: {resume_from}"
            ) from exc
        prior = latest.get("prior_runs")
        if (
            not isinstance(prior, list)
            or any(not isinstance(item, str) for item in prior)
            or len(set(prior)) != len(prior)
            or resume_from in prior
        ):
            raise ValueError("batch plan has an invalid run lineage")
        lineage = (*prior, resume_from)
        for index, old_run in enumerate(lineage):
            try:
                plan = self.store.load_batch_plan(old_run)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"cannot resume a run without a batch plan: {old_run}"
                ) from exc
            if set(plan) != set(expected) | {"prior_runs"}:
                raise ValueError(f"batch plan fields differ: {old_run}")
            base = {name: plan[name] for name in expected}
            if json.dumps(base, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise ValueError(f"batch inputs differ from previous run: {old_run}")
            if plan["prior_runs"] != list(lineage[:index]):
                raise ValueError(f"batch run lineage differs: {old_run}")
        return lineage

    def _reusable_records(
        self,
        cells: tuple[BatchCase, ...],
        locks: dict,
        agent_version: str,
        lineage: tuple[str, ...],
    ) -> dict[BatchCase, CaseRecord]:
        attempts: dict[BatchCase, CaseRecord | None] = {}
        selected = {cell.case_id: cell for cell in cells}
        for old_run in lineage:
            states = self.store.list_cases(old_run)
            if set(states) - set(selected):
                raise ValueError(f"previous run contains undeclared cases: {old_run}")
            for case_id, state in states.items():
                cell = selected[case_id]
                previous = attempts.get(cell)
                if previous is not None and (
                    previous.result.error is None or not previous.result.error.retryable
                ):
                    raise ValueError(
                        f"nonretryable case was rerun in previous lineage: {case_id}"
                    )
                lock = locks[cell.benchmark]
                task = lock.task_message(cell.task_id, cell.seed)
                provenance = Provenance.from_dict(
                    lock.provenance(agent_version).as_dict()
                )
                if state == "claimed":
                    if self.store.status(old_run, case_id).state == "running":
                        raise ValueError(
                            f"previous case still has an active worker: {old_run}/{case_id}"
                        )
                    try:
                        self.store.verify_claim(old_run, case_id, task, provenance)
                    except FileNotFoundError:
                        # A crash between directory claim and plan publication
                        # leaves no result to reuse. The next run is separate.
                        pass
                    attempts[cell] = None
                    continue
                result = self.store.load(old_run, case_id)
                if result.task != task or result.provenance != provenance:
                    raise ValueError(
                        f"published case differs from locked batch inputs: {case_id}"
                    )
                attempts[cell] = CaseRecord(cell, result, old_run)
        return {
            cell: record
            for cell, record in attempts.items()
            if record is not None
            and (record.result.error is None or not record.result.error.retryable)
        }

    def _prepare(
        self,
        run_id: str,
        cases: Sequence[BatchCase],
        agent_version: str,
        *,
        max_concurrent: int = 1,
        timeout_s: float = 30.0,
        case_timeout_s: float = 3600.0,
        resume_from: str | None = None,
    ) -> tuple:
        """Validate one batch without creating results or starting services."""
        cells = tuple(cases)
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("cases must be non-empty and unique")
        if type(max_concurrent) is not int or max_concurrent <= 0:
            raise ValueError("max_concurrent must be a positive integer")
        if not isinstance(agent_version, str) or not agent_version.strip():
            raise ValueError("agent_version must be non-empty text")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in (timeout_s, case_timeout_s)
        ):
            raise ValueError("case and RPC timeouts must be positive")
        locks = {cell.benchmark: self.registry.lock(cell.benchmark) for cell in cells}
        for cell in cells:
            locks[cell.benchmark].task(cell.task_id, cell.seed)
        expected = {
            "schema_version": 1,
            "cases": [
                {
                    "benchmark": cell.benchmark,
                    "task_id": cell.task_id,
                    "seed": cell.seed,
                    "case_id": cell.case_id,
                }
                for cell in cells
            ],
            "agent_version": agent_version,
            "timeout_s": float(timeout_s),
            "case_timeout_s": float(case_timeout_s),
            "manifest_sha256": {
                name: lock.manifest_sha256 for name, lock in locks.items()
            },
        }
        lineage = self._lineage(resume_from, expected)
        if run_id in lineage:
            raise ValueError("resume needs a new run ID")
        reused = self._reusable_records(cells, locks, agent_version, lineage)
        return cells, locks, expected, lineage, reused

    def preflight(
        self,
        run_id: str,
        cases: Sequence[BatchCase],
        agent_version: str,
        *,
        max_concurrent: int = 1,
        timeout_s: float = 30.0,
        case_timeout_s: float = 3600.0,
        resume_from: str | None = None,
    ) -> dict:
        """Validate inputs and resume state, returning a plan without side effects."""
        import re

        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("invalid run ID")
        if (self.store.root / run_id).exists():
            raise FileExistsError(f"run already exists: {run_id}")
        verify = (
            getattr(self.orchestrator.agent_launcher, "verify_version", None)
            if hasattr(self.orchestrator, "agent_launcher")
            else None
        )
        if verify is not None:
            verify(agent_version)
        _, _, plan, _, _ = self._prepare(
            run_id,
            cases,
            agent_version,
            max_concurrent=max_concurrent,
            timeout_s=timeout_s,
            case_timeout_s=case_timeout_s,
            resume_from=resume_from,
        )
        return plan

    def run(
        self,
        run_id: str,
        cases: Sequence[BatchCase],
        agent_version: str,
        *,
        max_concurrent: int = 1,
        timeout_s: float = 30.0,
        case_timeout_s: float = 3600.0,
        resume_from: str | None = None,
    ) -> BatchReport:
        """Start a fresh run, optionally reusing terminal results from a prior run."""
        cells, locks, expected, lineage, reused = self._prepare(
            run_id,
            cases,
            agent_version,
            max_concurrent=max_concurrent,
            timeout_s=timeout_s,
            case_timeout_s=case_timeout_s,
            resume_from=resume_from,
        )
        try:
            self.store.reserve_batch_run(
                run_id, {**expected, "prior_runs": list(lineage)}
            )
        except FileExistsError as exc:
            raise FileExistsError(f"run already exists: {run_id}") from exc

        def execute(cell: BatchCase) -> CaseRecord:
            self.orchestrator.run(
                run_id,
                cell.case_id,
                cell.case_id,
                cell.benchmark,
                cell.task_id,
                cell.seed,
                agent_version,
                timeout_s=timeout_s,
                case_timeout_s=case_timeout_s,
            )
            result = self.store.load(run_id, cell.case_id)
            lock = locks[cell.benchmark]
            declared_source = Provenance.from_dict(
                lock.provenance(agent_version).as_dict()
            )
            if (
                result.task != lock.task_message(cell.task_id, cell.seed)
                or result.provenance != declared_source
            ):
                raise ValueError("published case differs from locked batch inputs")
            return CaseRecord(cell, result, run_id)

        completed: dict[BatchCase, CaseRecord] = dict(reused)
        pending_cells = [cell for cell in cells if cell not in reused]
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {executor.submit(execute, cell): cell for cell in pending_cells}
            for future in as_completed(futures):
                try:
                    completed[futures[future]] = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
        records = tuple(completed[cell] for cell in cells)
        for lock in locks.values():
            lock.verify()
        summaries = {
            benchmark: summarize_cases(
                [record for record in records if record.case.benchmark == benchmark],
                lock.as_dict()["evaluation"],
            )
            for benchmark, lock in locks.items()
        }
        return BatchReport(run_id, records, summaries, resume_from)
