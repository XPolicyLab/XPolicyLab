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

"""Connect registry locks, independent runtimes and one-case result publication."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from threading import Event
from typing import Protocol

from roborsi.benchmarks.registry import BenchmarkLock, BenchmarkRegistry
from roborsi.evaluation.case_status import CasePublicationError
from roborsi.evaluation.case_trace import CaseTrace
from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline, CaseTimeout
from roborsi.evaluation.docker_common import DockerCommandError
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    EvaluationResult,
    Provenance,
    RuntimeConditions,
    RuntimeVerification,
    ServiceError,
)
from roborsi.evaluation.runner import CaseRunner
from roborsi.evaluation.services import AdapterClient, AgentClient
from roborsi.evaluation.store import ResultStore
from roborsi.utils.rpc.http_abort import interrupt_http_calls


class AgentService(Protocol):
    """Case-scoped Agent client with separately owned process lifetime."""

    client: AgentClient


class AdapterService(Protocol):
    """Case-scoped Adapter client with separately owned process lifetime."""

    client: AdapterClient


class AgentLauncher(Protocol):
    """Start an isolated Agent through its public service boundary."""

    def start(
        self,
        case_dir: Path,
        *,
        deadline: CaseDeadline | None = None,
        media_dir: Path | None = None,
    ) -> AbstractContextManager[AgentService]: ...


class AdapterExecutor(Protocol):
    """Start a locked Benchmark through its public service boundary."""

    def start_adapter(
        self,
        lock: BenchmarkLock,
        case_dir: Path,
        *,
        deadline: CaseDeadline | None = None,
    ) -> AbstractContextManager[AdapterService]: ...


class CaseOrchestrator:
    """Own one case from registry lookup through runtime cleanup and publication."""

    def __init__(
        self,
        registry: BenchmarkRegistry,
        store: ResultStore,
        agent_launcher: AgentLauncher,
        docker_executor: AdapterExecutor,
    ) -> None:
        self.registry = registry
        self.store = store
        self.agent_launcher = agent_launcher
        self.docker_executor = docker_executor

    def run(
        self,
        run_id: str,
        case_id: str,
        episode_id: str,
        benchmark: str,
        task_id: str,
        seed: int,
        agent_version: str,
        *,
        timeout_s: float = 30.0,
        case_timeout_s: float = 3600.0,
        cancel_event: Event | None = None,
    ) -> EvaluationResult:
        """Reserve one declared task before starting its Agent and Adapter."""
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = CaseDeadline(case_timeout_s, cancel_event=cancel_event)
        deadline.check()
        verify_version = getattr(self.agent_launcher, "verify_version", None)
        if verify_version is not None:
            verify_version(agent_version)
            deadline.check()
        with interrupt_http_calls(deadline.check):
            lock = self.registry.lock(benchmark)
        deadline.check()
        task = lock.task_message(task_id, seed)
        provenance = Provenance.from_dict(lock.provenance(agent_version).as_dict())
        deadline.check()
        directory = self.store.claim(
            run_id, case_id, task, provenance, episode_id=episode_id
        )
        with self.store.execution(run_id, case_id, episode_id):
            stage = "trace.start"
            trace: CaseTrace | None = None
            try:
                trace = self.store.start_trace(run_id, case_id, episode_id)
                stage = "media.setup"
                media_channel = CaseMediaChannel(directory / "media")
                stage = "adapter.startup"
                with ExitStack() as stack:
                    adapter_service = stack.enter_context(
                        self.docker_executor.start_adapter(
                            lock, directory, deadline=deadline
                        )
                    )
                    deadline.check()
                    stage = "agent.startup"
                    agent_directory = directory / "agent"
                    agent_directory.mkdir(exist_ok=True)
                    agent_service = stack.enter_context(
                        self.agent_launcher.start(
                            agent_directory,
                            deadline=deadline,
                            media_dir=media_channel.media_dir,
                        )
                    )
                    deadline.check()

                    def runtime_verification() -> RuntimeVerification:
                        agent_record = getattr(agent_service, "runtime_record", None)
                        benchmark_record = getattr(
                            adapter_service, "runtime_record", None
                        )
                        return RuntimeVerification(
                            agent=agent_record,
                            benchmark=benchmark_record,
                            observed=RuntimeVerification.observed_versions(
                                agent_record, benchmark_record
                            ),
                            conditions=RuntimeConditions(
                                timeout_s=timeout_s,
                                case_timeout_s=case_timeout_s,
                                evaluation=lock.as_dict()["evaluation"],
                            ),
                        )

                    stage = "case.run"
                    return CaseRunner(self.store).run(
                        run_id,
                        case_id,
                        episode_id,
                        task,
                        provenance,
                        agent_service.client,
                        adapter_service.client,
                        timeout_s=timeout_s,
                        preclaimed=True,
                        manage_status=False,
                        deadline=deadline,
                        media_channel=media_channel,
                        trace=trace,
                        runtime_cleanup=stack.close,
                        runtime_verification=runtime_verification,
                    )
            except Exception as exc:
                error = exc
                try:
                    deadline.check()
                except (CaseCancelled, CaseTimeout) as terminal:
                    error = terminal
                if self.store.status(run_id, case_id).result_published:
                    # A Store may commit a result outside the case directory.
                    raise
                agent_failure = stage == "agent.startup" and not isinstance(
                    error, (CaseCancelled, CaseTimeout, DockerCommandError)
                )
                status = (
                    "cancelled"
                    if isinstance(error, CaseCancelled)
                    else "agent_error"
                    if agent_failure
                    else "infrastructure_error"
                )
                code = (
                    "cancelled"
                    if isinstance(error, CaseCancelled)
                    else "timeout"
                    if isinstance(error, CaseTimeout)
                    else "agent_error"
                    if agent_failure
                    else "infrastructure_error"
                )
                result = EvaluationResult(
                    task=task,
                    episode_id=episode_id,
                    status=status,
                    success=None,
                    score=None,
                    provenance=provenance,
                    error=ServiceError(
                        code=code,
                        message=str(error) or type(error).__name__,
                        retryable=code in {"timeout", "infrastructure_error"},
                        details={"stage": stage, "exception": type(error).__name__},
                    ),
                )
                if trace is not None:
                    try:
                        reference = trace.finish(result.status, result.error)
                        result = result.model_copy(update={"trace_manifest": reference})
                    except (ValueError, OSError) as trace_error:
                        result = result.model_copy(
                            update={
                                "error": result.error.model_copy(
                                    update={
                                        "details": {
                                            **result.error.details,
                                            "trace_error": str(trace_error),
                                        }
                                    }
                                )
                            }
                        )
                try:
                    self.store.publish(run_id, case_id, result)
                except CasePublicationError as publication_error:
                    return publication_error.result
                return result
