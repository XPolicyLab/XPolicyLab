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

"""Evaluate, improve and compare saved Agents through public Runner interfaces."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol

from pydantic import Field, model_validator

from roborsi.evaluation.agent_artifact import AgentArtifact, AgentArtifactStore
from roborsi.evaluation.batch import BatchCase, BatchReport, summarize_cases
from roborsi.evaluation.case_status import timestamp
from roborsi.evaluation.messages import RoboRSIMessage, ServiceError
from roborsi.evaluation.result import write_json_atomic
from roborsi.evaluation.store import (
    ComparisonCheck,
    ComparisonPolicy,
    ResultStore,
    compare_results,
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")


class EvaluationSettings(RoboRSIMessage):
    """Identical Runner settings used for both versions."""

    max_concurrent: int = Field(default=1, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)
    case_timeout_s: float = Field(default=3600.0, gt=0)

    def arguments(self) -> dict[str, Any]:
        """Return the keyword arguments accepted by BatchEvaluator.run."""
        return self.model_dump(exclude={"schema_version"})


class CandidateProposal(RoboRSIMessage):
    """Describe edits made in the supplied workspace and optional entry changes."""

    reason: str = Field(min_length=1)
    factory: str | None = None
    components: dict[str, list[str]] | None = None
    environment_version: str | None = None


@dataclass(frozen=True)
class ImprovementContext:
    """Inputs to a trusted improver; source edits belong in workspace."""

    baseline: AgentArtifact
    workspace: Path
    report: BatchReport
    results: ResultStore

    def trace(self, case: BatchCase) -> list[dict[str, Any]]:
        """Read a baseline trace through the Store's integrity checks."""
        record = next((r for r in self.report.cases if r.case == case), None)
        if record is None:
            raise ValueError("case is not part of this baseline")
        return self.results.trace_events(record.source_run_id, case.case_id)


class Improver(Protocol):
    """Replaceable method that edits source after examining evaluation feedback."""

    def describe(self) -> dict[str, Any]: ...

    def propose(self, context: ImprovementContext) -> CandidateProposal: ...


class BatchRunner(Protocol):
    """The public BatchEvaluator surface required by RSI."""

    def run(
        self,
        run_id: str,
        cases: Sequence[BatchCase],
        agent_version: str,
        **settings: Any,
    ) -> BatchReport: ...


class CaseComparison(RoboRSIMessage):
    """One checked case pair and the reason if it cannot be compared."""

    case_id: str
    benchmark: str
    comparable: bool
    reason: str | None
    score_delta: float | None
    checks: tuple[ComparisonCheck, ...] = ()


class BenchmarkChange(RoboRSIMessage):
    """Separate aggregate scores; scores across Benchmarks are never averaged."""

    baseline: float | None
    candidate: float | None


class Comparison(RoboRSIMessage):
    """Evidence passed to a version retention rule."""

    cases: tuple[CaseComparison, ...]
    benchmarks: dict[str, BenchmarkChange]

    @property
    def comparable(self) -> bool:
        return bool(self.cases) and all(c.comparable for c in self.cases)


class RetentionDecision(RoboRSIMessage):
    """Keep the candidate or continue from the baseline, with an explicit reason."""

    retain: bool
    reason: str = Field(min_length=1)


class DecisionRule(Protocol):
    """Choose among comparable versions; cannot override the comparability gate."""

    def describe(self) -> dict[str, Any]: ...

    def decide(self, comparison: Comparison) -> RetentionDecision: ...


class NoRegressionRule:
    """Require no Benchmark regression and a gain greater than min_gain on one."""

    def __init__(self, min_gain: float = 0.0) -> None:
        if isinstance(min_gain, bool) or not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and nonnegative")
        self.min_gain = min_gain

    def describe(self) -> dict[str, Any]:
        return {
            "name": "no_regression",
            "min_gain": self.min_gain,
            "direction": "higher",
        }

    def decide(self, comparison: Comparison) -> RetentionDecision:
        if not comparison.comparable:
            return RetentionDecision(
                retain=False, reason="one or more cases are not comparable"
            )
        changes = comparison.benchmarks.values()
        if not changes or any(
            c.baseline is None or c.candidate is None for c in changes
        ):
            return RetentionDecision(
                retain=False, reason="a Benchmark has no aggregate score"
            )
        deltas = {
            name: c.candidate - c.baseline for name, c in comparison.benchmarks.items()
        }
        regressions = [name for name, delta in deltas.items() if delta < 0]
        if regressions:
            return RetentionDecision(
                retain=False,
                reason="Benchmark score decreased: " + ", ".join(regressions),
            )
        if not any(delta > self.min_gain for delta in deltas.values()):
            return RetentionDecision(
                retain=False, reason="no Benchmark improved beyond min_gain"
            )
        return RetentionDecision(
            retain=True, reason="no Benchmark regressed and at least one improved"
        )


class RSIRun(RoboRSIMessage):
    """Persistent state and evidence links for one improvement round."""

    run_id: str
    state: Literal[
        "baseline",
        "improving",
        "candidate",
        "comparing",
        "completed",
        "failed",
        "interrupted",
    ]
    updated_at: str
    baseline_version: str
    candidate_version: str | None = None
    selected_version: str | None = None
    baseline_run: str
    candidate_run: str
    cases: tuple[dict[str, Any], ...]
    settings: EvaluationSettings
    improver: dict[str, Any]
    rule: dict[str, Any]
    # Missing policy denotes a historical round using the former fixed checks.
    comparison_policy: ComparisonPolicy = Field(
        default_factory=ComparisonPolicy.strict_runtime
    )
    proposal: CandidateProposal | None = None
    comparison: Comparison | None = None
    decision: RetentionDecision | None = None
    reports: dict[str, str] = Field(default_factory=dict)
    error: ServiceError | None = None

    @model_validator(mode="after")
    def _state_evidence(self) -> RSIRun:
        if self.state == "completed":
            if (
                self.candidate_version is None
                or self.proposal is None
                or self.comparison is None
                or self.decision is None
                or set(self.reports) != {"baseline", "candidate"}
                or self.error is not None
            ):
                raise ValueError("completed RSI round requires comparison evidence")
            expected = (
                self.candidate_version
                if self.decision.retain
                else self.baseline_version
            )
            if self.selected_version != expected:
                raise ValueError("selected Agent version differs from decision")
            if self.decision.retain and not self.comparison.comparable:
                raise ValueError("incomparable candidate cannot be selected")
        elif self.selected_version is not None or self.decision is not None:
            raise ValueError("unfinished RSI round cannot select an Agent version")
        if self.state == "failed" and self.error is None:
            raise ValueError("failed RSI round requires an error")
        return self


class RSIStore:
    """Own round state and report snapshots on a local POSIX filesystem."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, run_id: str) -> Path:
        """Resolve a round name within this Store."""
        if not isinstance(run_id, str) or not _IDENTIFIER.fullmatch(run_id):
            raise ValueError(
                "RSI run_id must contain letters, numbers, underscores or hyphens"
            )
        return self.root / run_id

    @contextmanager
    def create(self, record: RSIRun) -> Iterator[Path]:
        """Claim a new round and hold its worker lock until execution exits."""
        import fcntl

        directory = self.directory(record.run_id)
        directory.mkdir()
        with (directory / "worker.lock").open("xb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                self.write(record)
                yield directory
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def write(self, record: RSIRun) -> None:
        """Atomically replace the current round state."""
        write_json_atomic(
            self.directory(record.run_id) / "record.json", record.as_dict()
        )

    def status(self, run_id: str) -> RSIRun:
        """Read state and report checksums; identify an exited active worker."""
        import fcntl

        directory = self.directory(run_id)
        record = RSIRun.from_dict(json.loads((directory / "record.json").read_text()))
        if record.run_id != run_id:
            raise ValueError("RSI record identity differs")
        for name, digest in record.reports.items():
            if (
                name not in {"baseline", "candidate"}
                or hashlib.sha256((directory / f"{name}.json").read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError("RSI report checksum differs")
        if record.state in {"baseline", "improving", "candidate", "comparing"}:
            with (directory / "worker.lock").open("rb") as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return record
                fcntl.flock(stream, fcntl.LOCK_UN)
            return record.model_copy(update={"state": "interrupted"})
        return record


class RSIController:
    """Run one improvement round without importing Benchmark implementations.

    runner_factory returns a fresh BatchEvaluator (or compatible Runner) for an
    Agent artifact. Improvers and custom decision rules execute trusted Python.
    Each call starts fresh baseline/candidate runs; prior rounds stay intact.
    """

    def __init__(
        self,
        artifacts: AgentArtifactStore,
        results: ResultStore,
        runs: RSIStore,
        runner_factory: Callable[[AgentArtifact], BatchRunner],
    ) -> None:
        self.artifacts, self.results, self.runs = artifacts, results, runs
        self.runner_factory = runner_factory

    def _evaluate(
        self,
        run_id: str,
        artifact: AgentArtifact,
        cases: tuple[BatchCase, ...],
        settings: EvaluationSettings,
    ) -> BatchReport:
        artifact.verify()
        report = self.runner_factory(artifact).run(
            run_id, cases, artifact.version_id, **settings.arguments()
        )
        if (
            report.run_id != run_id
            or report.resume_from is not None
            or tuple(r.case for r in report.cases) != cases
        ):
            raise ValueError(
                "RSI Runner returned a different task matrix or reused a run"
            )
        for record in report.cases:
            result = self.results.load(run_id, record.case.case_id)
            identity = BatchCase(
                result.task.benchmark, result.task.task_id, result.task.seed
            )
            if (
                identity != record.case
                or record.source_run_id != run_id
                or record.result != result
                or result.provenance.agent_version != artifact.version_id
            ):
                raise ValueError(
                    "RSI Runner result differs from the stored Agent evaluation"
                )
        if set(report.summaries) != {c.benchmark for c in cases}:
            raise ValueError("RSI Runner returned a different Benchmark summary set")
        for name, summary in report.summaries.items():
            records = [r for r in report.cases if r.case.benchmark == name]
            evaluation = {
                "aggregation": summary.aggregation,
                "error_denominator": summary.error_denominator,
            }
            for entry in records:
                if entry.result.runtime is not None and any(
                    entry.result.runtime.conditions.evaluation.get(key) != value
                    for key, value in evaluation.items()
                ):
                    raise ValueError(
                        "RSI aggregation differs from recorded evaluation rules"
                    )
            if summarize_cases(records, evaluation) != summary:
                raise ValueError("RSI Runner aggregate differs from its case results")
        artifact.verify()
        return report

    def _compare(
        self, baseline: BatchReport, candidate: BatchReport, policy: ComparisonPolicy
    ) -> Comparison:
        comparisons = []
        for a, b in zip(baseline.cases, candidate.cases, strict=True):
            checked = compare_results(
                self.results.load(a.source_run_id, a.case.case_id),
                self.results.load(b.source_run_id, b.case.case_id),
                policy=policy,
            )
            comparisons.append(
                CaseComparison(
                    case_id=a.case.case_id,
                    benchmark=a.case.benchmark,
                    comparable=checked.comparable,
                    reason=checked.reason,
                    score_delta=checked.score_delta,
                    checks=checked.checks,
                )
            )
        if baseline.summaries.keys() != candidate.summaries.keys():
            raise ValueError("RSI Benchmark summaries differ")
        return Comparison(
            cases=tuple(comparisons),
            benchmarks={
                name: BenchmarkChange(
                    baseline=summary.aggregate_score,
                    candidate=candidate.summaries[name].aggregate_score,
                )
                for name, summary in baseline.summaries.items()
            },
        )

    def run(
        self,
        run_id: str,
        baseline_version: str,
        cases: Sequence[BatchCase],
        improver: Improver,
        *,
        settings: EvaluationSettings | None = None,
        rule: DecisionRule | None = None,
        comparison_policy: ComparisonPolicy | None = None,
    ) -> RSIRun:
        """Evaluate two fresh versions and record the checked retention decision."""
        cells = tuple(cases)
        if not cells or len(set(cells)) != len(cells):
            raise ValueError("RSI cases must be non-empty and unique")
        baseline = self.artifacts.load(baseline_version)
        settings = settings or EvaluationSettings()
        rule = rule or NoRegressionRule()
        record = RSIRun(
            run_id=run_id,
            state="baseline",
            updated_at=timestamp(),
            baseline_version=baseline_version,
            baseline_run=f"{run_id}-baseline",
            candidate_run=f"{run_id}-candidate",
            cases=tuple(
                {"benchmark": c.benchmark, "task_id": c.task_id, "seed": c.seed}
                for c in cells
            ),
            settings=settings,
            improver=improver.describe(),
            rule=rule.describe(),
            comparison_policy=comparison_policy or ComparisonPolicy(),
        )

        def update(**values: Any) -> None:
            nonlocal record
            updated = RSIRun.from_dict(
                {**record.as_dict(), **values, "updated_at": timestamp()}
            )
            self.runs.write(updated)
            record = updated

        def save_report(name: str, report: BatchReport) -> None:
            path = self.runs.directory(run_id) / f"{name}.json"
            write_json_atomic(path, report.as_dict())
            update(
                reports={
                    **record.reports,
                    name: hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )

        with self.runs.create(record) as directory:
            try:
                first = self._evaluate(record.baseline_run, baseline, cells, settings)
                save_report("baseline", first)
                update(state="improving")
                workspace = baseline.checkout(directory / "source")
                context = ImprovementContext(
                    self.artifacts.load(baseline_version),
                    workspace,
                    copy.deepcopy(first),
                    self.results,
                )
                proposal = improver.propose(context)
                if not isinstance(proposal, CandidateProposal):
                    raise TypeError("improver must return CandidateProposal")
                baseline.verify()
                candidate = self.artifacts.create(
                    workspace,
                    proposal.factory or baseline.document["factory"],
                    components=proposal.components
                    if proposal.components is not None
                    else baseline.document["components"],
                    parent_version=baseline_version,
                    environment_version=proposal.environment_version,
                )
                update(
                    state="candidate",
                    candidate_version=candidate.version_id,
                    proposal=proposal.as_dict(),
                )
                second = self._evaluate(
                    record.candidate_run, candidate, cells, settings
                )
                save_report("candidate", second)
                update(state="comparing")
                comparison = self._compare(first, second, record.comparison_policy)
                decision = (
                    rule.decide(copy.deepcopy(comparison))
                    if comparison.comparable
                    else RetentionDecision(
                        retain=False, reason="one or more cases are not comparable"
                    )
                )
                if not isinstance(decision, RetentionDecision):
                    raise TypeError("decision rule must return RetentionDecision")
                baseline.verify()
                candidate.verify()
                update(
                    state="completed",
                    comparison=comparison.as_dict(),
                    decision=decision.as_dict(),
                    selected_version=candidate.version_id
                    if decision.retain
                    else baseline_version,
                )
            except BaseException as exc:
                update(
                    state="failed",
                    error=ServiceError(
                        code="infrastructure_error",
                        message=f"RSI {record.state} failed",
                        details={
                            "stage": record.state,
                            "exception": type(exc).__name__,
                        },
                    ).as_dict(),
                )
                raise
        return record
