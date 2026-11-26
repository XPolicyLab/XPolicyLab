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

"""Case-scoped RoboRSI result publication, evidence verification, and comparison."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from roborsi.evaluation.case_status import (
    CaseJournal,
    CasePublicationError,
    CaseStatus,
    timestamp,
)
from roborsi.evaluation.case_trace import CaseTrace, read_case_trace
from roborsi.evaluation.messages import (
    ContainerAgentRuntimeRecord,
    EvaluationResult,
    Provenance,
    RoboRSIMessage,
    ServiceError,
    Task,
)
from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.logging import get_logger

logger = get_logger("evaluation.store")

_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")
_REPRODUCIBILITY_FIELDS = (
    "benchmark_version",
    "benchmark_commit",
    "assets_version",
    "adapter_version",
)


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonstandard JSON constant: {value}")

    document = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def _sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


@dataclass(frozen=True, slots=True)
class TraceState:
    """Completion metadata read from a case's trace manifest."""

    status: str
    complete: bool


class ComparisonPolicy(RoboRSIMessage):
    """Choose deployment checks in addition to required evaluation conditions.

    Task, scoring rules, timeouts, verified Agent identity, Benchmark/Adapter
    versions and asset contents are always checked. Deployment evidence remains
    visible even when it is not required. This policy never queries a platform.
    """

    require_image_digest: bool = False
    compare_benchmark_configuration: bool = False
    compare_host_platform: bool = False
    compare_agent_configuration: bool = False

    @classmethod
    def strict_runtime(cls) -> ComparisonPolicy:
        """Require all deployment checks, matching the former fixed policy."""
        return cls(
            require_image_digest=True,
            compare_benchmark_configuration=True,
            compare_host_platform=True,
            compare_agent_configuration=True,
        )


class ComparisonCheck(RoboRSIMessage):
    """One observable condition, including optional deployment evidence."""

    name: str
    required: bool
    status: Literal["passed", "differs", "unknown"]
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ResultComparison:
    """Score comparison with the selected policy and observable checks."""

    comparable: bool
    reason: str | None
    score_delta: float | None
    success_changed: bool | None
    policy: ComparisonPolicy = field(default_factory=ComparisonPolicy)
    checks: tuple[ComparisonCheck, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return the decision, chosen policy and all checks as JSON data."""
        return {
            "comparable": self.comparable,
            "reason": self.reason,
            "score_delta": self.score_delta,
            "success_changed": self.success_changed,
            "policy": self.policy.as_dict(),
            "checks": [check.as_dict() for check in self.checks],
        }


class ResultStore:
    """Provide immutable per-case result records under one storage root.

    ``claim`` creates the case directory exclusively, then writes its planned
    Task and provenance. A crash before publication leaves a visible claimed
    case, which a later batch run can inspect rather than overwrite.
    """

    def __init__(
        self, root: str | Path, *, status_root: str | Path | None = None
    ) -> None:
        self.root = Path(root).resolve()
        self.journal = CaseJournal(
            Path(status_root).resolve()
            if status_root is not None
            else self.root / ".case-status"
        )

    def reserve_batch_run(self, run_id: str, plan: dict[str, Any]) -> Path:
        """Atomically reserve a fresh batch run before any case starts."""
        if not isinstance(plan, dict):
            raise TypeError("batch plan must be a JSON object")
        directory = self._case_dir(run_id, "placeholder").parent
        self.root.mkdir(parents=True, exist_ok=True)
        directory.mkdir(exist_ok=False)
        try:
            write_json_atomic(directory / "batch_plan.json", plan)
        except Exception:
            try:
                directory.rmdir()
            except OSError:
                # Preserve a concurrently claimed case or another artifact.
                pass
            raise
        return directory

    def load_batch_plan(self, run_id: str) -> dict[str, Any]:
        """Read a previously reserved batch run's immutable input plan."""
        directory = self._case_dir(run_id, "placeholder").parent
        return _strict_json(directory / "batch_plan.json")

    def _case_dir(self, run_id: str, case_id: str) -> Path:
        if not isinstance(run_id, str) or not _IDENTIFIER.fullmatch(run_id):
            raise ValueError("run_id must contain only letters, digits, _ or -")
        if not isinstance(case_id, str) or not _IDENTIFIER.fullmatch(case_id):
            raise ValueError("case_id must contain only letters, digits, _ or -")
        return self.root / run_id / case_id

    def claim(
        self,
        run_id: str,
        case_id: str,
        task: Task,
        provenance: Provenance,
        *,
        episode_id: str | None = None,
    ) -> Path:
        """Atomically reserve a case and pin its declared evaluation inputs."""
        if not isinstance(task, Task) or not isinstance(provenance, Provenance):
            raise TypeError("claim requires Task and Provenance")
        destination = self._case_dir(run_id, case_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(exist_ok=False)
        try:
            self.journal.create(run_id, case_id, task, provenance, episode_id)
        except Exception:
            destination.rmdir()
            raise
        try:
            write_json_atomic(
                destination / "plan.json",
                {
                    "schema_version": 1,
                    "task": task.as_dict(),
                    "provenance": provenance.as_dict(),
                },
            )
        except (ValueError, OSError) as exc:
            self.journal.failure(run_id, case_id, exc, "store.claim")
            raise
        return destination

    @contextmanager
    def execution(self, run_id: str, case_id: str, episode_id: str) -> Iterator[None]:
        """Hold a process lock while an accepted case owns live runtime work."""
        directory = self._case_dir(run_id, case_id)
        try:
            self.journal.read(run_id, case_id)
        except FileNotFoundError:
            plan = _strict_json(directory / "plan.json")
            self.journal.create(
                run_id,
                case_id,
                Task.from_dict(plan["task"]),
                Provenance.from_dict(plan["provenance"]),
                episode_id,
            )
        if (directory / "result.json").exists():
            raise FileExistsError("case result is already published")
        with self.journal.execution(
            run_id,
            case_id,
            episode_id,
            published=lambda: (directory / "result.json").is_file(),
        ):
            yield

    def status(self, run_id: str, case_id: str) -> CaseStatus:
        """Read execution state and verify any published result without changing it."""
        directory = self._case_dir(run_id, case_id)
        try:
            saved = self.journal.status(run_id, case_id)
        except FileNotFoundError:
            saved = None
        if saved is not None and saved.state == "failed" and not saved.result_published:
            return saved
        if (directory / "result.json").is_file():
            updated_at = datetime.fromtimestamp(
                (directory / "result.json").stat().st_mtime, timezone.utc
            ).isoformat()
            try:
                result = self.load(run_id, case_id)
            except (ValueError, OSError) as exc:
                return CaseStatus(
                    run_id=run_id,
                    case_id=case_id,
                    state="failed",
                    updated_at=updated_at,
                    result_published=True,
                    error=ServiceError(
                        code="infrastructure_error",
                        message=str(exc),
                        details={"stage": "store.load"},
                    ),
                )
            state = (
                "cancelled"
                if result.status == "cancelled"
                else "failed"
                if result.error
                else "completed"
            )
            return CaseStatus(
                run_id=run_id,
                case_id=case_id,
                episode_id=result.episode_id,
                state=state,
                updated_at=updated_at,
                result_published=True,
                result_status=result.status,
                error=result.error,
            )
        if saved is not None:
            return saved
        if not directory.is_dir():
            raise FileNotFoundError(f"unknown case: {run_id}/{case_id}")
        return CaseStatus(
            run_id=run_id,
            case_id=case_id,
            state="unknown",
            updated_at=timestamp(),
            error=ServiceError(
                code="infrastructure_error",
                message="legacy case has no worker status record",
                retryable=True,
            ),
        )

    def verify_claim(
        self, run_id: str, case_id: str, task: Task, provenance: Provenance
    ) -> Path:
        """Validate a previously reserved case before launching its services."""
        directory = self._case_dir(run_id, case_id)
        plan = _strict_json(directory / "plan.json")
        if (
            plan.get("task") != task.as_dict()
            or plan.get("provenance") != provenance.as_dict()
        ):
            raise ValueError("claimed case inputs differ from requested evaluation")
        if (directory / "result.json").exists():
            raise FileExistsError(f"case result is already published: {directory}")
        return directory

    def _artifact(self, directory: Path, reference: str) -> Path:
        path = (directory / reference).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"case artifact unavailable or outside case: {reference}")
        return path

    def start_trace(self, run_id: str, case_id: str, episode_id: str) -> CaseTrace:
        """Start an exclusive Runner recording for an already claimed case."""
        directory = self._case_dir(run_id, case_id)
        plan = _strict_json(directory / "plan.json")
        task = Task.from_dict(plan["task"])
        provenance = Provenance.from_dict(plan["provenance"])
        self.verify_claim(run_id, case_id, task, provenance)
        return CaseTrace(directory, run_id, case_id, episode_id, task, provenance)

    def _trace_state(
        self, directory: Path, reference: str | None, result: EvaluationResult
    ) -> TraceState | None:
        if reference is None:
            return None
        manifest = _strict_json(self._artifact(directory, reference))
        if "format" in manifest:
            recording, _ = read_case_trace(
                directory,
                reference,
                task=result.task,
                provenance=result.provenance,
                episode_id=result.episode_id,
            )
            return TraceState(recording.status, recording.complete)
        if type(manifest.get("complete")) is not bool:
            raise ValueError("trace manifest needs a boolean complete field")
        status = manifest.get("status")
        if not isinstance(status, str) or not status:
            raise ValueError("trace manifest needs a non-empty status")
        return TraceState(status=status, complete=manifest["complete"])

    def _verify_artifacts(
        self, directory: Path, result: EvaluationResult
    ) -> TraceState | None:
        if result.native_score_source is not None:
            if not result.native_score_source.startswith("native/"):
                raise ValueError("native score must live under native/")
            native = self._artifact(directory, result.native_score_source)
            if _sha256(native) != result.native_score_sha256:
                raise ValueError("native score digest mismatch")
        seen = set()
        for artifact in result.benchmark_artifacts:
            if artifact.episode_id != result.episode_id or artifact.ref in seen:
                raise ValueError(
                    "Benchmark artifact belongs to another episode or repeats a ref"
                )
            seen.add(artifact.ref)
            path = self._artifact(directory, artifact.ref)
            if (
                path.stat().st_size != artifact.size_bytes
                or _sha256(path) != artifact.sha256
            ):
                raise ValueError(
                    "downloaded Benchmark artifact differs from HTTP metadata"
                )
        return self._trace_state(directory, result.trace_manifest, result)

    def verify_artifacts(
        self, run_id: str, case_id: str, result: EvaluationResult
    ) -> TraceState | None:
        """Check result files before finalizing the Runner's trace or publishing."""
        return self._verify_artifacts(self._case_dir(run_id, case_id), result)

    def publish(
        self, run_id: str, case_id: str, result: EvaluationResult
    ) -> TraceState | None:
        """Validate a claimed case and publish its result without replacement."""
        if not isinstance(result, EvaluationResult):
            raise TypeError("publish requires EvaluationResult")
        result = EvaluationResult.from_dict(result.as_dict())
        directory = self._case_dir(run_id, case_id)
        plan = _strict_json(directory / "plan.json")
        if plan.get("task") != result.task.as_dict():
            raise ValueError("result task differs from claimed task")
        if plan.get("provenance") != result.provenance.as_dict():
            raise ValueError("result provenance differs from claimed provenance")
        try:
            if self.journal.fallback(run_id, case_id) is not None:
                raise FileExistsError("case already has a final storage failure record")
        except FileNotFoundError:
            self.journal.create(
                run_id, case_id, result.task, result.provenance, result.episode_id
            )
        trace = self._verify_artifacts(directory, result)
        destination = directory / "result.json"
        temporary = directory / f".result-{uuid.uuid4().hex}.json"
        try:
            write_json_atomic(temporary, result.as_dict())
            os.link(temporary, destination)
        except FileExistsError:
            raise
        except OSError as exc:
            failure = self.journal.failure(
                run_id, case_id, exc, "store.publish", previous=result
            )
            assert failure is not None
            raise CasePublicationError(failure) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove temporary result file", exc_info=True)
        try:
            self.journal.completed(run_id, case_id, result)
        except OSError:
            # result.json is already committed and remains authoritative.
            logger.warning(
                "result saved but status journal update failed", exc_info=True
            )
        return trace

    def load(self, run_id: str, case_id: str) -> EvaluationResult:
        """Read and revalidate the published message and its evidence."""
        directory = self._case_dir(run_id, case_id)
        try:
            fallback = self.journal.fallback(run_id, case_id)
        except FileNotFoundError:
            fallback = None
        if fallback is not None:
            return fallback
        document = _strict_json(directory / "result.json")
        result = EvaluationResult.from_dict(document)
        plan = _strict_json(directory / "plan.json")
        if (
            plan.get("task") != result.task.as_dict()
            or plan.get("provenance") != result.provenance.as_dict()
        ):
            raise ValueError("stored result differs from claimed inputs")
        self._verify_artifacts(directory, result)
        return result

    def trace_state(self, run_id: str, case_id: str) -> TraceState | None:
        """Report whether a published case's referenced trace is complete."""
        result = self.load(run_id, case_id)
        if result.trace_manifest is None:
            return None
        manifest = _strict_json(
            self._artifact(self._case_dir(run_id, case_id), result.trace_manifest)
        )
        return TraceState(manifest["status"], manifest["complete"])

    def trace_events(self, run_id: str, case_id: str) -> list[dict[str, Any]]:
        """Read a verified Runner recording, including messages and media references."""
        result = self.load(run_id, case_id)
        if result.trace_manifest is None:
            raise ValueError("case has no trace")
        _, events = read_case_trace(
            self._case_dir(run_id, case_id),
            result.trace_manifest,
            task=result.task,
            provenance=result.provenance,
            episode_id=result.episode_id,
        )
        return events

    def list_cases(self, run_id: str) -> dict[str, str]:
        """Return claimed and published case states for one run."""
        parent = self._case_dir(run_id, "placeholder").parent
        names = (
            {
                path.name
                for path in parent.iterdir()
                if path.is_dir() and _IDENTIFIER.fullmatch(path.name)
            }
            if parent.is_dir()
            else set()
        )
        names.update(
            path.stem
            for path in (self.journal.root / run_id).glob("*.json")
            if _IDENTIFIER.fullmatch(path.stem)
        )
        return {
            name: "published"
            if (parent / name / "result.json").is_file()
            else "failed"
            if self.status(run_id, name).state == "failed"
            else "claimed"
            for name in sorted(names)
        }

    def compare(
        self,
        left_run: str,
        left_case: str,
        right_run: str,
        right_case: str,
        *,
        policy: ComparisonPolicy | None = None,
    ) -> ResultComparison:
        """Load checked evidence and compare under an explicit evaluation policy."""
        left = self.load(left_run, left_case)
        right = self.load(right_run, right_case)
        return compare_results(left, right, policy=policy)


def compare_results(
    left: EvaluationResult,
    right: EvaluationResult,
    *,
    policy: ComparisonPolicy | None = None,
) -> ResultComparison:
    """Compare validated results without deployment or storage I/O.

    Storage implementations must load and verify both records and their
    artifacts before calling this function. Optional checks report unknown or
    different deployment evidence without blocking the default policy.
    """
    policy = ComparisonPolicy.from_dict((policy or ComparisonPolicy()).as_dict())
    checks: list[ComparisonCheck] = []

    def check(
        name: str,
        a: Any,
        b: Any,
        reason: str,
        *,
        required: bool = True,
        presence_only: bool = False,
    ) -> None:
        status = (
            "unknown"
            if a is None or b is None
            else "passed"
            if presence_only or a == b
            else "differs"
        )
        checks.append(
            ComparisonCheck(
                name=name,
                required=required,
                status=status,
                reason=None if status == "passed" else reason,
            )
        )

    def result() -> ResultComparison:
        failure = next((c for c in checks if c.required and c.status != "passed"), None)
        return ResultComparison(
            failure is None,
            failure.reason if failure else None,
            right.score - left.score if failure is None else None,
            right.success != left.success if failure is None else None,
            policy,
            tuple(checks),
        )

    check("task", left.task, right.task, "task or run configuration differs")
    check(
        "runtime",
        left.runtime,
        right.runtime,
        "runtime versions were not verified",
        presence_only=True,
    )
    if left.runtime is None or right.runtime is None:
        return result()
    a_runtime, b_runtime = left.runtime, right.runtime
    check(
        "agent_version",
        a_runtime.observed.agent_version,
        b_runtime.observed.agent_version,
        "Agent source and environment were not verified",
        presence_only=True,
    )
    for name in (*_REPRODUCIBILITY_FIELDS, "docker_image_digest"):
        a, b = getattr(a_runtime.observed, name), getattr(b_runtime.observed, name)
        check(
            name,
            a,
            b,
            f"{name} is unknown" if a is None or b is None else f"{name} differs",
            required=name != "docker_image_digest" or policy.require_image_digest,
        )
    check(
        "conditions",
        a_runtime.conditions,
        b_runtime.conditions,
        "Runner settings or scoring rules differ",
    )
    a_agent, b_agent = a_runtime.agent, b_runtime.agent
    check(
        "agent_configuration",
        a_agent.configuration
        if isinstance(a_agent, ContainerAgentRuntimeRecord)
        else {},
        b_agent.configuration
        if isinstance(b_agent, ContainerAgentRuntimeRecord)
        else {},
        "Agent runtime settings differ",
        required=policy.compare_agent_configuration,
    )
    if isinstance(a_agent, ContainerAgentRuntimeRecord) != isinstance(
        b_agent, ContainerAgentRuntimeRecord
    ):
        checks[-1] = ComparisonCheck(
            name="agent_configuration",
            required=policy.compare_agent_configuration,
            status="differs",
            reason="Agent runtime settings differ",
        )
    a_benchmark, b_benchmark = a_runtime.benchmark, b_runtime.benchmark
    for name, attribute, required, reason in (
        ("assets", "assets", True, "asset contents differ"),
        (
            "benchmark_configuration",
            "configuration",
            policy.compare_benchmark_configuration,
            "container configuration differs",
        ),
    ):
        check(
            name,
            getattr(a_benchmark, attribute, None),
            getattr(b_benchmark, attribute, None),
            reason,
            required=required,
        )
    check(
        "host_platform",
        a_benchmark.info.platform if a_benchmark and a_benchmark.info else None,
        b_benchmark.info.platform if b_benchmark and b_benchmark.info else None,
        "Benchmark host platform differs",
        required=policy.compare_host_platform,
    )
    check(
        "native_score",
        left.score,
        right.score,
        "one case has no native score",
        presence_only=True,
    )
    return result()
