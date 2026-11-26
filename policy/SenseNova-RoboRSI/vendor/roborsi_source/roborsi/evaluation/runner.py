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

"""Drive one RoboRSI case through public Agent, Adapter, and Store interfaces."""

from __future__ import annotations

import math
from collections.abc import Callable
from contextlib import ExitStack

from roborsi.evaluation.adapter_service import _validate_action_step
from roborsi.evaluation.case_status import CasePublicationError
from roborsi.evaluation.case_trace import CaseTrace
from roborsi.evaluation.deadline import CaseCancelled, CaseDeadline, CaseTimeout
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import (
    BenchmarkQuery,
    EpisodeStop,
    EvaluationResult,
    NativeResult,
    Observation,
    Provenance,
    RuntimeVerification,
    ServiceError,
    Task,
)
from roborsi.evaluation.services import AdapterClient, AgentClient
from roborsi.evaluation.store import ResultStore
from roborsi.utils.rpc import RpcError, RpcTransportError
from roborsi.utils.rpc.http_abort import interrupt_http_calls


class CaseRunner:
    """Synchronous single-case coordinator; process startup remains external."""

    def __init__(self, store: ResultStore) -> None:
        self.store = store

    def run(
        self,
        run_id: str,
        case_id: str,
        episode_id: str,
        task: Task,
        provenance: Provenance,
        agent: AgentClient,
        adapter: AdapterClient,
        *,
        timeout_s: float = 30.0,
        agent_timeout_s: float | None = None,
        trace_manifest: str | None = None,
        preclaimed: bool = False,
        manage_status: bool = True,
        deadline: CaseDeadline | None = None,
        media_channel: CaseMediaChannel | None = None,
        trace: CaseTrace | None = None,
        runtime_cleanup: Callable[[], None] | None = None,
        failure_cleanup: Callable[[], None] | None = None,
        runtime_verification: Callable[[], RuntimeVerification] | None = None,
    ) -> EvaluationResult:
        """Execute a claimed case and publish a native score or classified error.

        The caller supplies two running services and owns their processes.
        This method closes both clients and invokes optional runtime cleanup
        before finalizing the trace and result. An explicit ``trace_manifest``
        retains the caller's recording instead of creating a Runner trace.
        Set ``manage_status=False`` only when the caller already holds the
        store's execution context for this case.

        ``agent_timeout_s=None`` inherits ``timeout_s``. Zero disables the
        connected-read timeout for Agent act/resume/observe_end only, requires
        ``controller_owned_wait`` capability, and still honors ``deadline``.
        Other stages retain ``timeout_s``. On failure, ``failure_cleanup`` runs
        before client close to stop directly owned work blocked inside an RPC.
        Cleanup exceptions are recorded without replacing the primary failure.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if agent_timeout_s is not None:
            if (
                isinstance(agent_timeout_s, bool)
                or not isinstance(agent_timeout_s, (int, float))
                or not math.isfinite(agent_timeout_s)
                or agent_timeout_s < 0
            ):
                raise ValueError("agent_timeout_s must be finite and nonnegative")
            if (
                agent_timeout_s == 0
                and task.capabilities.get("controller_owned_wait") is not True
            ):
                raise ValueError("agent_timeout_s=0 requires controller_owned_wait")
        if trace is not None and trace_manifest is not None:
            raise ValueError("supply a trace recorder or a trace manifest, not both")
        with ExitStack() as lifecycle:
            try:
                if preclaimed:
                    directory = self.store.verify_claim(
                        run_id, case_id, task, provenance
                    )
                else:
                    directory = self.store.claim(
                        run_id, case_id, task, provenance, episode_id=episode_id
                    )
                if manage_status:
                    lifecycle.enter_context(
                        self.store.execution(run_id, case_id, episode_id)
                    )
            except Exception:
                for client in (agent, adapter):
                    try:
                        client.close()
                    except Exception:
                        pass
                raise
            stage = "adapter.start"
            native: NativeResult | None = None
            failure: ServiceError | None = None
            status = "infrastructure_error"

            def call_timeout() -> float:
                limit = timeout_s
                if agent_timeout_s is not None and stage in {
                    "agent.act",
                    "agent.resume",
                    "agent.observe_end",
                }:
                    limit = agent_timeout_s
                return (
                    deadline.remaining(None if limit == 0 else limit)
                    if deadline is not None
                    else limit
                )

            def call(method, *args):
                with interrupt_http_calls(
                    deadline.check if deadline is not None else None
                ):
                    return method(*args, timeout_s=call_timeout())

            def check_deadline() -> None:
                if deadline is not None:
                    deadline.check()

            def verify_media(observation: Observation) -> None:
                if observation.media:
                    if media_channel is None:
                        raise ValueError("observation media needs a case media channel")
                    collect = getattr(adapter, "collect_observation", None)
                    if collect is None:
                        raise ValueError(
                            "Benchmark images require HTTP artifact downloads"
                        )
                    call(collect, observation, media_channel)
                    media_channel.verify_observation(observation)
                    check_deadline()

            try:
                stage = "trace.start"
                if trace is None and trace_manifest is None:
                    trace = self.store.start_trace(run_id, case_id, episode_id)
                stage = "adapter.start"
                start = call(adapter.start, task, episode_id)
                check_deadline()
                verify_media(start.observation)
                if start.task != task:
                    raise ValueError("Adapter actual task differs from requested task")
                stage = "trace.record"
                if trace is not None:
                    trace.record(start, stage="adapter.start")
                stage = "agent.reset"
                call(agent.reset, start.task, episode_id)
                check_deadline()
                observation = start.observation
                accepted_steps = 0
                while accepted_steps < task.max_steps:
                    stage = "trace.record"
                    if trace is not None:
                        trace.begin_step(observation)
                    stage = "agent.act"
                    action = call(agent.act, observation)
                    check_deadline()
                    decision_stage = "agent.act"
                    while True:
                        if (
                            action.benchmark != observation.benchmark
                            or action.episode_id != observation.episode_id
                            or action.step_index != observation.step_index
                        ):
                            raise ValueError(
                                "Agent action differs from current observation"
                            )
                        stage = "trace.record"
                        if trace is not None:
                            trace.record(action, stage=decision_stage)
                        if not isinstance(action, BenchmarkQuery):
                            break
                        stage = "adapter.query"
                        reply = call(adapter.query, action)
                        check_deadline()
                        if (
                            reply.benchmark,
                            reply.episode_id,
                            reply.step_index,
                            reply.query_id,
                        ) != (
                            action.benchmark,
                            action.episode_id,
                            action.step_index,
                            action.query_id,
                        ):
                            raise ValueError("Adapter query reply differs from request")
                        stage = "trace.record"
                        if trace is not None:
                            trace.record(reply, stage="adapter.query")
                        stage = "agent.resume"
                        action = call(agent.resume, reply)
                        check_deadline()
                        decision_stage = "agent.resume"
                    operation = (
                        "adapter.stop"
                        if isinstance(action, EpisodeStop)
                        else "adapter.step"
                    )
                    stage = operation
                    step = call(
                        adapter.stop
                        if isinstance(action, EpisodeStop)
                        else adapter.step,
                        action,
                    )
                    check_deadline()
                    if not isinstance(action, EpisodeStop):
                        rejected = _validate_action_step(
                            step,
                            action,
                            recoverable_action_rejections=task.capabilities.get(
                                "recoverable_action_rejections"
                            )
                            is True,
                        )
                        if not rejected:
                            accepted_steps += 1
                    verify_media(step.observation)
                    observation = step.observation
                    stage = "trace.record"
                    if trace is not None:
                        trace.record(step, stage=operation)
                        trace.end_step()
                    if step.terminated or step.truncated:
                        observe_end = getattr(agent, "observe_end", None)
                        if observe_end is not None:
                            stage = "agent.observe_end"
                            call(observe_end, step)
                            check_deadline()
                        stage = "adapter.finish"
                        native = call(adapter.finish)
                        check_deadline()
                        if native.task != task or native.episode_id != episode_id:
                            raise ValueError("Adapter score identity differs from task")
                        stage = "adapter.artifacts"
                        if native.source is not None:
                            collect = getattr(adapter, "collect_native", None)
                            if collect is None:
                                raise ValueError(
                                    "result evidence invalid: Benchmark outputs require HTTP downloads"
                                )
                            call(collect, native, directory)
                        collect_artifacts = getattr(adapter, "collect_artifacts", None)
                        if collect_artifacts is not None:
                            call(collect_artifacts, directory)
                        check_deadline()
                        stage = "trace.record"
                        if trace is not None:
                            trace.record(native, stage="adapter.finish")
                        status = "success" if native.success else "task_failure"
                        break
                else:
                    stage = "adapter.step"
                    raise RuntimeError("Adapter did not terminate within max_steps")
            except Exception as exc:
                error = exc
                if deadline is not None:
                    try:
                        deadline.check()
                    except (CaseCancelled, CaseTimeout) as terminal:
                        error = terminal
                remote_code = error.code if isinstance(error, RpcError) else None
                if (
                    remote_code
                    not in {
                        "invalid_request",
                        "unsupported_version",
                        "infrastructure_error",
                    }
                    and not (
                        remote_code == "agent_error" and stage.startswith("agent.")
                    )
                    and not (
                        remote_code == "adapter_error" and stage.startswith("adapter.")
                    )
                ):
                    remote_code = None
                status = (
                    "cancelled"
                    if isinstance(error, CaseCancelled)
                    else "infrastructure_error"
                    if isinstance(error, CaseTimeout)
                    else "infrastructure_error"
                    if isinstance(error, RpcTransportError)
                    else "infrastructure_error"
                    if remote_code
                    in {
                        "invalid_request",
                        "unsupported_version",
                        "infrastructure_error",
                    }
                    else "agent_error"
                    if stage.startswith("agent.")
                    else "infrastructure_error"
                )
                code = (
                    "cancelled"
                    if isinstance(error, CaseCancelled)
                    else "timeout"
                    if isinstance(error, TimeoutError)
                    else "infrastructure_error"
                    if isinstance(error, RpcTransportError)
                    else remote_code
                    if remote_code is not None
                    else "agent_error"
                    if status == "agent_error"
                    else "adapter_error"
                    if stage.startswith("adapter.")
                    else "infrastructure_error"
                )
                failure = ServiceError(
                    code=code,
                    message=str(error) or type(error).__name__,
                    retryable=code
                    in {"timeout", "adapter_error", "infrastructure_error"},
                    details={"stage": stage, "exception": type(error).__name__},
                )
            cleanup_errors: list[str] = []
            if failure is not None and failure_cleanup is not None:
                try:
                    # An owning runtime can interrupt a controller whose RPC
                    # act still holds the service lock before close_episode.
                    failure_cleanup()
                except Exception as exc:
                    cleanup_errors.append(
                        f"failure runtime: {type(exc).__name__}: {exc}"
                    )
            for name, client in (("agent", agent), ("adapter", adapter)):
                if name == "adapter" and failure is not None:
                    collect_failure = getattr(client, "collect_failure_artifacts", None)
                    if callable(collect_failure):
                        try:
                            # The policy has closed. This is bounded evidence
                            # cleanup, not additional episode execution time.
                            collect_failure(directory, timeout_s=timeout_s)
                        except Exception as exc:
                            cleanup_errors.append(
                                f"adapter artifacts: {type(exc).__name__}: {exc}"
                            )
                try:
                    client.close()
                except Exception as exc:
                    cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
            if runtime_cleanup is not None:
                try:
                    runtime_cleanup()
                except Exception as exc:
                    cleanup_errors.append(f"runtime: {type(exc).__name__}: {exc}")
            if cleanup_errors:
                if failure is None:
                    status = "infrastructure_error"
                    failure = ServiceError(
                        code="infrastructure_error",
                        message="service cleanup failed",
                        details={
                            "stage": "service.cleanup",
                            "cleanup_errors": cleanup_errors,
                        },
                    )
                else:
                    failure = failure.model_copy(
                        update={
                            "details": {
                                **failure.details,
                                "cleanup_errors": cleanup_errors,
                            }
                        }
                    )
            if failure is None and deadline is not None:
                try:
                    deadline.check()
                except (CaseCancelled, CaseTimeout) as exc:
                    status = (
                        "cancelled"
                        if isinstance(exc, CaseCancelled)
                        else "infrastructure_error"
                    )
                    failure = ServiceError(
                        code="cancelled"
                        if isinstance(exc, CaseCancelled)
                        else "timeout",
                        message=str(exc),
                        retryable=isinstance(exc, CaseTimeout),
                        details={"stage": "service.cleanup"},
                    )
            runtime = None
            if runtime_verification is not None:
                try:
                    runtime = runtime_verification()
                    runtime.verify_requested(task, provenance)
                except Exception as exc:
                    if failure is None:
                        status = "infrastructure_error"
                        failure = ServiceError(
                            code="infrastructure_error",
                            message=str(exc),
                            details={"stage": "runtime.verify"},
                        )
                    else:
                        failure = failure.model_copy(
                            update={
                                "details": {
                                    **failure.details,
                                    "runtime_error": str(exc),
                                }
                            }
                        )
            if failure is None:
                assert native is not None
                result = EvaluationResult(
                    task=task,
                    episode_id=episode_id,
                    status=status,
                    success=native.success,
                    score=native.score,
                    native_score_source=native.source,
                    native_score_sha256=native.source_sha256,
                    trace_manifest=trace_manifest,
                    provenance=provenance,
                    runtime=runtime,
                    benchmark_artifacts=getattr(adapter, "artifact_records", ()),
                )
            else:
                result = EvaluationResult(
                    task=task,
                    episode_id=episode_id,
                    status=status,
                    success=None,
                    score=None,
                    trace_manifest=trace_manifest,
                    provenance=provenance,
                    runtime=runtime,
                    benchmark_artifacts=getattr(adapter, "artifact_records", ()),
                    error=failure,
                )

            def evidence_failure(
                exc: Exception, previous: EvaluationResult
            ) -> EvaluationResult:
                return EvaluationResult(
                    task=task,
                    episode_id=episode_id,
                    status="infrastructure_error",
                    success=None,
                    score=None,
                    provenance=provenance,
                    error=ServiceError(
                        code="infrastructure_error",
                        message=f"result evidence invalid: {exc}",
                        details={
                            "stage": "store.publish",
                            "previous_error": previous.error.as_dict()
                            if previous.error is not None
                            else None,
                        },
                    ),
                )

            try:
                self.store.verify_artifacts(run_id, case_id, result)
            except (ValueError, OSError) as exc:
                result = evidence_failure(exc, result)
            if trace is not None:
                try:
                    reference = trace.finish(result.status, result.error)
                    result = result.model_copy(update={"trace_manifest": reference})
                except (ValueError, OSError) as exc:
                    result = EvaluationResult(
                        task=task,
                        episode_id=episode_id,
                        status="infrastructure_error",
                        success=None,
                        score=None,
                        provenance=provenance,
                        error=ServiceError(
                            code="infrastructure_error",
                            message=f"trace finalization failed: {exc}",
                            details={
                                "stage": "trace.finish",
                                "previous_error": result.error.as_dict()
                                if result.error
                                else None,
                            },
                        ),
                    )
            try:
                self.store.publish(run_id, case_id, result)
            except CasePublicationError as exc:
                return exc.result
            except (ValueError, FileNotFoundError) as exc:
                result = evidence_failure(exc, result)
                try:
                    self.store.publish(run_id, case_id, result)
                except CasePublicationError as publication_error:
                    return publication_error.result
            return result
