# Project RoboOrchard
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
import copy
import queue
import weakref
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import torch
from robo_orchard_core.utils.logging import LoggerManager

from robo_orchard_lab.policy.base import PolicyConfig, PolicyMixin
from robo_orchard_lab.policy.evaluator.base import (
    PolicyEvaluator,
    PolicyEvaluatorConfig,
)
from robo_orchard_lab.policy.evaluator.benchmark.core import (
    BenchmarkAttemptRequest,
    BenchmarkDriver,
    BenchmarkEvaluateFailedEvent,
    BenchmarkEvaluateSucceededEvent,
    BenchmarkPrepareFailedEvent,
    BenchmarkPrepareSucceededEvent,
    BenchmarkResult,
    BenchmarkTerminalEvent,
)
from robo_orchard_lab.policy.evaluator.contracts import PreparedEnvStart
from robo_orchard_lab.policy.evaluator.metric_contracts import (
    EvaluatorMetrics,
)
from robo_orchard_lab.policy.evaluator.remote import (
    PolicyEvaluatorRemote,
    PolicyEvaluatorRemoteConfig,
)

__all__ = [
    "LocalBenchmarkBackend",
    "LocalBenchmarkBackendConfig",
    "RemoteBenchmarkBackend",
    "RemoteBenchmarkBackendConfig",
]

logger = LoggerManager().get_child(__name__)

_WorkerState: TypeAlias = Literal["idle", "preparing", "evaluating"]
_WORKER_EVENT_WAIT_TIMEOUT_S = 0.2


def _format_exception_message(exc: BaseException) -> str:
    """Return a serializable message that keeps the exception cause chain."""

    message = str(exc) or type(exc).__name__
    if (
        getattr(exc, "cause_type", None) is not None
        or getattr(exc, "cause_message", None) is not None
    ):
        return message
    cause_parts: list[str] = []
    seen = {id(exc)}
    cause = exc.__cause__
    if cause is None and not exc.__suppress_context__:
        cause = exc.__context__
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        cause_message = str(cause)
        if cause_message:
            cause_parts.append(f"{type(cause).__name__}: {cause_message}")
        else:
            cause_parts.append(type(cause).__name__)
        next_cause = cause.__cause__
        if next_cause is None and not cause.__suppress_context__:
            next_cause = cause.__context__
        cause = next_cause
    if cause_parts:
        return f"{message} Cause chain: {' <- '.join(cause_parts)}"
    return message


@dataclass(slots=True)
class LocalBenchmarkBackendConfig:
    """Configure the current-process single-worker benchmark backend.

    The local backend owns one :class:`PolicyEvaluator` at a time and runs
    prepare/evaluate serially in the caller process. It is intended for local
    debugging and lightweight smoke tests, not for timeout isolation. The
    caller supplies the evaluator config so domain evaluators own policy such
    as whether repeated same-config prepares may reuse the current env.
    """

    evaluator_cfg: PolicyEvaluatorConfig
    """Config used to create the local evaluator instance."""

    policy_or_cfg: PolicyConfig | PolicyMixin
    """Policy instance or config passed to first-time local evaluator setup."""

    worker_metrics_factory: Callable[[], EvaluatorMetrics]
    """Factory for fresh worker-local metric objects during first setup."""

    device: str | torch.device | None = None
    """Optional policy device passed through to local evaluator setup."""

    def __post_init__(self) -> None:
        if not callable(self.evaluator_cfg):
            raise TypeError("evaluator_cfg must be callable.")
        if not callable(self.worker_metrics_factory):
            raise TypeError("worker_metrics_factory must be callable.")


@dataclass(slots=True)
class _LocalPreparedWork:
    """Carry local prepare output into the serial evaluate step."""

    request: BenchmarkAttemptRequest
    """Concrete attempt prepared by the local evaluator."""

    prepared_start: PreparedEnvStart
    """Prepared observation/reset-info payload used to skip a second reset."""


class LocalBenchmarkBackend:
    """Run benchmark attempts serially in the current process.

    Use this backend when a benchmark needs the generic prepare/evaluate
    pipeline but should avoid Ray startup for local debugging or fast smoke
    tests. Domain readiness, retry, seed or offset advancement, artifact
    paths, and aggregate metrics remain in the supplied driver.
    """

    def __init__(self, cfg: LocalBenchmarkBackendConfig) -> None:
        self.cfg = cfg
        self._closed = False
        self._evaluator: PolicyEvaluator | None = None
        self._evaluator_configured = False

    def run(self, driver: BenchmarkDriver) -> BenchmarkResult:
        """Run the serial local scheduler until the driver completes.

        Each loop asks the driver for at most one ready job, materializes an
        attempt request, runs prepare and evaluate synchronously, then
        dispatches prepared and terminal callbacks before asking for more
        work. Retry, readiness, and aggregate result policy remain in the
        driver.

        Stage meanings:

        - Prepare initializes or reconfigures the evaluator, including env,
          policy, metrics, and device on first use. It resets per-attempt
          metrics and starts the env from the request's env reset input,
          producing a :class:`PreparedEnvStart` for rollout.
        - Evaluate runs exactly one episode from that prepared start. In this
          backend flow it does not reset the env again, but the evaluator
          still owns policy reset, rollout, terminal metric computation, and
          env episode finalization.

        The backend closes its evaluator before returning or re-raising.
        """

        self._ensure_open()
        try:
            while driver.has_unfinished_work():
                jobs = list(driver.get_ready_jobs(max_jobs=1))
                if len(jobs) > 1:
                    raise RuntimeError(
                        "Benchmark driver returned too many ready jobs: "
                        f"{len(jobs)} > 1."
                    )
                if not jobs:
                    raise RuntimeError(
                        "Benchmark driver made no progress: it reported "
                        "unfinished work but returned no ready jobs."
                    )

                request = driver.make_attempt_request(jobs[0])
                prepared = self._prepare(request)
                if isinstance(prepared, BenchmarkPrepareFailedEvent):
                    driver.on_terminal_event(prepared)
                    continue

                driver.on_attempt_prepared(
                    BenchmarkPrepareSucceededEvent(
                        request=request,
                        reset_info=dict(prepared.prepared_start.info),
                        worker_id=0,
                    )
                )
                driver.on_terminal_event(self._evaluate(prepared))

            return driver.result()
        finally:
            self.close()

    def close(self) -> None:
        """Close the local evaluator and make cleanup repeatable."""

        if self._closed:
            return
        self._closed = True
        self._close_local_evaluator()

    def __enter__(self) -> LocalBenchmarkBackend:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("LocalBenchmarkBackend is closed.")

    def _ensure_evaluator(self) -> PolicyEvaluator:
        if self._evaluator is None:
            self._evaluator = self.cfg.evaluator_cfg()
            self._evaluator_configured = False
        return self._evaluator

    def _prepare(
        self,
        request: BenchmarkAttemptRequest,
    ) -> _LocalPreparedWork | BenchmarkPrepareFailedEvent:
        evaluator = self._ensure_evaluator()
        try:
            if self._evaluator_configured:
                evaluator.reconfigure_env(request.env_cfg)
            else:
                logger.info(
                    "Initializing local benchmark evaluator: "
                    "worker=0 task=%s episode=%s",
                    request.episode.group_key,
                    request.episode.episode_id,
                )
                evaluator.setup(
                    env_cfg=request.env_cfg,
                    policy_or_cfg=self.cfg.policy_or_cfg,
                    metrics=self.cfg.worker_metrics_factory(),
                    device=self.cfg.device,
                )
                logger.info(
                    "Initialized local benchmark evaluator: "
                    "worker=0 task=%s episode=%s",
                    request.episode.group_key,
                    request.episode.episode_id,
                )
                self._evaluator_configured = True
            evaluator.reset_metrics(**request.metric_reset_input)
            reset_ret = evaluator.reset_env(
                env_reset_input=request.env_reset_input,
            )
            return _LocalPreparedWork(
                request=request,
                prepared_start=PreparedEnvStart.from_reset_return(reset_ret),
            )
        except Exception as exc:
            logger.exception(
                "Local benchmark prepare failed: task=%s episode=%s",
                request.episode.group_key,
                request.episode.episode_id,
            )
            self._close_local_evaluator()
            return BenchmarkPrepareFailedEvent(
                request=request,
                reset_info=dict(getattr(exc, "reset_info", {}))
                if isinstance(getattr(exc, "reset_info", None), dict)
                else {},
                error_type=type(exc).__name__,
                error_message=_format_exception_message(exc),
                worker_id=0,
            )

    def _evaluate(
        self,
        work: _LocalPreparedWork,
    ) -> BenchmarkTerminalEvent:
        evaluator = self._require_evaluator()
        try:
            episode_metrics = evaluator.evaluate_episode(
                max_steps=work.request.max_steps,
                env_reset_input=work.prepared_start,
                policy_reset_input=work.request.policy_reset_input,
            )
            worker_metrics = evaluator.get_metrics()
            if not isinstance(worker_metrics, EvaluatorMetrics):
                raise TypeError(
                    "Local evaluator get_metrics() must return "
                    "EvaluatorMetrics after setup. Got "
                    f"{type(worker_metrics).__name__}."
                )
            return BenchmarkEvaluateSucceededEvent(
                request=work.request,
                episode_metrics=dict(episode_metrics),
                worker_metrics=copy.deepcopy(worker_metrics),
                reset_info=dict(work.prepared_start.info),
                worker_id=0,
            )
        except Exception as exc:
            logger.exception(
                "Local benchmark evaluate failed: task=%s episode=%s",
                work.request.episode.group_key,
                work.request.episode.episode_id,
            )
            self._close_local_evaluator()
            return BenchmarkEvaluateFailedEvent(
                request=work.request,
                reset_info=dict(work.prepared_start.info),
                error_type=type(exc).__name__,
                error_message=_format_exception_message(exc),
                worker_id=0,
            )

    def _require_evaluator(self) -> PolicyEvaluator:
        if self._evaluator is None:
            raise RuntimeError("Local evaluator is not configured.")
        return self._evaluator

    def _close_local_evaluator(self) -> None:
        evaluator = self._evaluator
        self._evaluator = None
        self._evaluator_configured = False
        if evaluator is not None:
            evaluator.close()


@dataclass(slots=True)
class RemoteBenchmarkBackendConfig:
    """Configure the generic remote benchmark worker pool.

    The backend owns remote evaluator instances, prepare/evaluate scheduling,
    and cleanup. The driver still owns domain readiness, retry, logical
    records, and metric aggregation. Env reuse for configured workers is
    controlled by the ``PolicyEvaluatorConfig`` inside ``remote_cfg``. Remote
    call timeout defaults also live on ``remote_cfg``; this backend does not
    duplicate those timeout fields.
    """

    remote_cfg: PolicyEvaluatorRemoteConfig
    """Factory/config object used to create one remote evaluator per worker."""

    num_workers: int
    """Maximum number of workers preparing or evaluating in flight."""

    policy_or_cfg: PolicyConfig | PolicyMixin
    """Policy instance or config passed to first-time worker setup."""

    worker_metrics_factory: Callable[[], EvaluatorMetrics]
    """Factory for fresh worker-local metric objects during first setup."""

    device: str | torch.device | None = None
    """Optional policy device passed through to worker setup."""

    def __post_init__(self) -> None:
        if self.num_workers <= 0:
            raise ValueError("Remote benchmark backend requires workers.")
        if not callable(self.worker_metrics_factory):
            raise TypeError("worker_metrics_factory must be callable.")


@dataclass(slots=True)
class _BenchmarkWorkerLease:
    """Capture a worker identity and evaluator for one submitted future."""

    worker_id: int
    """Stable slot id used to route completion back to the scheduler."""

    generation: int
    """Generation used to ignore stale completions after worker replace."""

    evaluator: PolicyEvaluatorRemote
    """Remote evaluator actor owned by the slot at submission time."""


@dataclass(slots=True)
class _BenchmarkPreparedWork:
    """Carry successful prepare output into the evaluate stage."""

    lease: _BenchmarkWorkerLease
    """Worker lease that prepared the env and must run the rollout."""

    request: BenchmarkAttemptRequest
    """Concrete attempt prepared on the leased worker."""

    prepared_start: PreparedEnvStart
    """Prepared observation/reset-info payload used to skip a second reset."""


_BenchmarkWorkerPayload: TypeAlias = (
    _BenchmarkPreparedWork | BenchmarkTerminalEvent | BaseException
)


@dataclass(slots=True)
class _BenchmarkWorkerEvent:
    """Queue message emitted by worker futures for scheduler consumption."""

    worker_id: int
    """Slot id captured when the future was submitted."""

    generation: int
    """Slot generation captured when the future was submitted."""

    payload: _BenchmarkWorkerPayload
    """Prepared work, terminal event, or unexpected future exception."""


@dataclass(slots=True)
class _BenchmarkWorkerSlot:
    """Mutable scheduler-owned state for one remote worker slot."""

    worker_id: int
    """Stable slot id in the backend worker list."""

    generation: int
    """Incremented whenever the remote evaluator is replaced after failure."""

    evaluator: PolicyEvaluatorRemote
    """Current remote evaluator actor for this slot."""

    state: _WorkerState = "idle"
    """Scheduler-visible lifecycle state for capacity and shutdown."""

    configured: bool = False
    """Whether this evaluator can skip setup on the next prepare.

    ``False`` means the prepare worker must call ``setup(...)`` to initialize
    env, policy, metrics, and device. ``True`` means the evaluator already
    has configured policy/metric runtime, so prepare calls ``reconfigure_env``
    for the request env config before resetting metrics and env. Env reuse is
    governed by the remote evaluator's ``PolicyEvaluatorConfig``. The
    scheduler sets this to ``True`` only after prepare succeeds, and resets it
    to ``False`` when replacing a failed remote evaluator.
    """

    future: Future[_BenchmarkWorkerPayload] | None = None
    """In-flight prepare/evaluate future, if any."""

    request: BenchmarkAttemptRequest | None = None
    """Current concrete attempt for diagnostics and cleanup visibility."""


class RemoteBenchmarkBackend:
    """Run benchmark attempts on a pool of remote policy evaluators.

    Use this backend when a domain benchmark needs a prepare/reset stage
    before rollout but wants the generic worker pool, scheduling, stale-event
    handling, and cleanup owned in one place. The backend does not decide
    retry policy, domain readiness, offsets, artifact paths, or aggregate
    metric semantics; those remain in the supplied :class:`BenchmarkDriver`.
    """

    def __init__(
        self,
        cfg: RemoteBenchmarkBackendConfig,
    ) -> None:
        self.cfg = cfg
        self._worker_event_queue: queue.Queue[_BenchmarkWorkerEvent]
        self._worker_event_queue = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=cfg.num_workers)
        self._closed = False
        logger.info(
            "Starting remote benchmark evaluator workers: workers=%s",
            cfg.num_workers,
        )
        self._workers = [
            _BenchmarkWorkerSlot(
                worker_id=worker_id,
                generation=0,
                evaluator=cast(PolicyEvaluatorRemote, cfg.remote_cfg()),
            )
            for worker_id in range(cfg.num_workers)
        ]
        logger.info(
            "Started remote benchmark evaluator workers: workers=%s",
            cfg.num_workers,
        )
        self._finalizer = weakref.finalize(
            self,
            self._finalize_resources,
            self._workers,
            self._executor,
        )

    def run(self, driver: BenchmarkDriver) -> BenchmarkResult:
        """Run the remote event-driven scheduler until the driver completes.

        Scheduler state is owned by this calling thread. Worker futures only
        communicate completion through an internal thread-safe queue, and all
        driver callbacks are invoked serially from this method.

        Stage meanings:

        - ``idle`` means the scheduler may lease the worker for a new
          attempt.
        - ``preparing`` runs in a worker future. It initializes or
          reconfigures the remote evaluator, including env, policy, metrics,
          and device on first use; resets per-attempt metrics; and starts the
          env from the request's env reset input.
        - ``evaluating`` runs one episode rollout from the prepared start and
          captures the worker's metric snapshot. In this backend flow it does
          not reset the env again, but the remote evaluator still owns policy
          reset, rollout, terminal metric computation, and env episode
          finalization.

        Event flow::

            driver.get_ready_jobs()
                    |
                    v
            _schedule_ready_work() --> prepare future
                    ^                       |
                    |                       v
            _drain_worker_events() <-- _worker_event_queue
                    |
                    v
            _handle_worker_event()
                |-- stale generation: drop
                |-- prepare succeeded: driver prepared callback
                |                      then start evaluate future
                |-- terminal event: replace failed workers
                                  then driver terminal callback

        When work is in flight and no queued events are ready, the scheduler
        waits boundedly for one worker event. Retry and aggregate result
        policy remain in the driver; worker replacement remains in scheduler
        event handling.
        """

        self._ensure_open()
        try:
            while driver.has_unfinished_work() or self._has_pending_work():
                self._drain_worker_events(driver)
                self._schedule_ready_work(driver)

                if self._has_pending_work():
                    try:
                        event = self._worker_event_queue.get(
                            timeout=_WORKER_EVENT_WAIT_TIMEOUT_S,
                        )
                    except queue.Empty:
                        continue
                    self._handle_worker_event(driver, event)
                    continue

                if driver.has_unfinished_work():
                    raise RuntimeError(
                        "Benchmark driver made no progress: it reported "
                        "unfinished work but returned no ready jobs and the "
                        "backend has no in-flight worker work."
                    )
                break

            return driver.result()
        finally:
            self.close()

    def close(self) -> None:
        """Best-effort cleanup for queued work, futures, and remote workers."""

        if self._closed:
            return
        self._closed = True
        for worker in self._workers:
            if worker.future is not None:
                worker.future.cancel()
            self._close_evaluator(worker.evaluator)
            worker.state = "idle"
            worker.request = None
            worker.future = None
            worker.configured = False
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._finalizer.detach()

    def __enter__(self) -> RemoteBenchmarkBackend:
        self._ensure_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RemoteBenchmarkBackend is closed.")

    def _has_pending_work(self) -> bool:
        return any(worker.state != "idle" for worker in self._workers)

    def _schedule_ready_work(self, driver: BenchmarkDriver) -> None:
        """Pull driver-ready jobs and submit any that fit idle workers.

        This method is scheduler-owned state only: it is the only place that
        asks the driver for new jobs, and it converts jobs into concrete
        attempts only after an idle worker can be reserved.
        """

        idle_workers = [
            worker for worker in self._workers if worker.state == "idle"
        ]
        capacity = len(idle_workers)
        if capacity <= 0:
            return

        jobs = list(driver.get_ready_jobs(max_jobs=capacity))
        if len(jobs) > capacity:
            raise RuntimeError(
                "Benchmark driver returned too many ready jobs: "
                f"{len(jobs)} > {capacity}."
            )

        for worker, job in zip(idle_workers, jobs, strict=False):
            request = driver.make_attempt_request(job)
            self._start_prepare_future(worker, request)

    def _start_prepare_future(
        self,
        worker: _BenchmarkWorkerSlot,
        request: BenchmarkAttemptRequest,
    ) -> None:
        """Submit blocking prepare/reset work for one reserved worker.

        The done callback only enqueues an event with the captured generation;
        it never mutates scheduler-owned worker state directly.
        """

        worker.state = "preparing"
        worker.request = request
        future = self._executor.submit(
            self._run_prepare_worker,
            _BenchmarkWorkerLease(
                worker_id=worker.worker_id,
                generation=worker.generation,
                evaluator=worker.evaluator,
            ),
            request,
            worker.configured,
        )
        worker.future = future
        worker_id = worker.worker_id
        generation = worker.generation
        future.add_done_callback(
            lambda done: (
                self._on_worker_future_done(worker_id, generation, done)
            )
        )

    def _start_evaluate_future(
        self,
        worker: _BenchmarkWorkerSlot,
        work: _BenchmarkPreparedWork,
    ) -> None:
        """Submit rollout/metric-capture work on the prepared worker."""

        worker.state = "evaluating"
        worker.request = work.request
        future = self._executor.submit(self._run_evaluate_worker, work)
        worker.future = future
        worker_id = worker.worker_id
        generation = worker.generation
        future.add_done_callback(
            lambda done: (
                self._on_worker_future_done(worker_id, generation, done)
            )
        )

    def _run_prepare_worker(
        self,
        lease: _BenchmarkWorkerLease,
        request: BenchmarkAttemptRequest,
        evaluator_configured: bool,
    ) -> _BenchmarkWorkerPayload:
        """Run prepare in an executor thread and return a queue payload.

        Prepare owns setup/reconfigure, per-attempt metric reset, and env
        reset. It returns prepared observations to the scheduler; driver
        callbacks still run only on the scheduler thread.

        ``evaluator_configured`` is the scheduler-owned slot state captured at
        submission time. When it is ``False``, prepare performs first-time
        ``setup(...)`` with env, policy, metrics, and device. When it is
        ``True``, those policy/metric resources are reused and prepare only
        calls ``reconfigure_env(...)`` for the request env config before
        resetting metrics and env. Env reuse is governed by the remote
        evaluator's ``PolicyEvaluatorConfig``.
        """

        reset_info: dict[str, Any] = {}
        try:
            if evaluator_configured:
                lease.evaluator.reconfigure_env(
                    request.env_cfg,
                )
            else:
                logger.debug(
                    "Initializing remote benchmark evaluator: "
                    "worker=%s task=%s episode=%s",
                    lease.worker_id,
                    request.episode.group_key,
                    request.episode.episode_id,
                )
                lease.evaluator.setup(
                    env_cfg=request.env_cfg,
                    policy_or_cfg=self.cfg.policy_or_cfg,
                    metrics=self.cfg.worker_metrics_factory(),
                    device=self.cfg.device,
                )
                logger.debug(
                    "Initialized remote benchmark evaluator: "
                    "worker=%s task=%s episode=%s",
                    lease.worker_id,
                    request.episode.group_key,
                    request.episode.episode_id,
                )
            lease.evaluator.reset_metrics(
                **request.metric_reset_input,
            )
            reset_ret = lease.evaluator.reset_env(
                env_reset_input=request.env_reset_input,
            )
            prepared_start = PreparedEnvStart.from_reset_return(reset_ret)
            reset_info = dict(prepared_start.info)
            return _BenchmarkPreparedWork(
                lease=lease,
                request=request,
                prepared_start=prepared_start,
            )
        except Exception as exc:
            logger.exception(
                "Remote benchmark prepare failed: worker=%s task=%s "
                "episode=%s",
                lease.worker_id,
                request.episode.group_key,
                request.episode.episode_id,
            )
            return BenchmarkPrepareFailedEvent(
                request=request,
                reset_info=(
                    reset_info
                    or (
                        dict(getattr(exc, "reset_info", {}))
                        if isinstance(getattr(exc, "reset_info", None), dict)
                        else {}
                    )
                ),
                error_type=type(exc).__name__,
                error_message=_format_exception_message(exc),
                worker_id=lease.worker_id,
            )

    def _run_evaluate_worker(
        self,
        work: _BenchmarkPreparedWork,
    ) -> _BenchmarkWorkerPayload:
        """Run rollout and metric snapshot capture in an executor thread."""

        try:
            episode_metrics = work.lease.evaluator.evaluate_episode(
                max_steps=work.request.max_steps,
                env_reset_input=work.prepared_start,
                policy_reset_input=work.request.policy_reset_input,
            )
            worker_metrics = work.lease.evaluator.get_metrics()
            if not isinstance(worker_metrics, EvaluatorMetrics):
                raise TypeError(
                    "Remote evaluator get_metrics() must return "
                    "EvaluatorMetrics after setup. Got "
                    f"{type(worker_metrics).__name__}."
                )
            return BenchmarkEvaluateSucceededEvent(
                request=work.request,
                episode_metrics=dict(episode_metrics),
                worker_metrics=worker_metrics,
                reset_info=dict(getattr(work.prepared_start, "info", {})),
                worker_id=work.lease.worker_id,
            )
        except Exception as exc:
            logger.exception(
                "Remote benchmark evaluate failed: worker=%s task=%s "
                "episode=%s",
                work.lease.worker_id,
                work.request.episode.group_key,
                work.request.episode.episode_id,
            )
            return BenchmarkEvaluateFailedEvent(
                request=work.request,
                reset_info=dict(getattr(work.prepared_start, "info", {})),
                error_type=type(exc).__name__,
                error_message=_format_exception_message(exc),
                worker_id=work.lease.worker_id,
            )

    def _on_worker_future_done(
        self,
        worker_id: int,
        generation: int,
        future: Future[_BenchmarkWorkerPayload],
    ) -> None:
        """Move future completion into the scheduler-owned event queue."""

        try:
            payload = future.result()
        except BaseException as exc:
            payload = exc
        self._worker_event_queue.put(
            _BenchmarkWorkerEvent(
                worker_id=worker_id,
                generation=generation,
                payload=payload,
            )
        )

    def _drain_worker_events(
        self,
        driver: BenchmarkDriver,
    ) -> None:
        """Handle every queued worker event before blocking for more work."""

        while True:
            try:
                event = self._worker_event_queue.get_nowait()
            except queue.Empty:
                return
            self._handle_worker_event(driver, event)

    def _handle_worker_event(
        self,
        driver: BenchmarkDriver,
        event: _BenchmarkWorkerEvent,
    ) -> None:
        """Apply one worker completion on the scheduler thread.

        Stale generation events and callbacks arriving after close are ignored.
        All driver callbacks and worker state transitions happen here, never
        in executor callback threads.
        """

        worker = self._workers[event.worker_id]
        if self._closed:
            logger.debug(
                "Ignoring benchmark worker event after backend close: "
                "worker=%s generation=%s",
                event.worker_id,
                event.generation,
            )
            return
        if event.generation != worker.generation:
            logger.debug(
                "Ignoring stale benchmark worker event: worker=%s "
                "event_generation=%s current_generation=%s",
                event.worker_id,
                event.generation,
                worker.generation,
            )
            return
        payload = event.payload
        if isinstance(payload, BaseException):
            raise payload

        if isinstance(payload, _BenchmarkPreparedWork):
            if payload.lease.evaluator is not worker.evaluator:
                raise RuntimeError(
                    "Prepared work evaluator does not match worker slot."
                )
            worker.configured = True
            prepared_event = BenchmarkPrepareSucceededEvent(
                request=payload.request,
                reset_info=dict(getattr(payload.prepared_start, "info", {})),
                worker_id=worker.worker_id,
            )
            driver.on_attempt_prepared(prepared_event)
            self._start_evaluate_future(worker, payload)
            return

        if isinstance(payload, BenchmarkPrepareFailedEvent):
            self._finish_worker(worker, replace=True)
            driver.on_terminal_event(payload)
            return

        if isinstance(
            payload,
            (BenchmarkEvaluateSucceededEvent, BenchmarkEvaluateFailedEvent),
        ):
            self._finish_worker(
                worker,
                replace=isinstance(payload, BenchmarkEvaluateFailedEvent),
            )
            driver.on_terminal_event(payload)
            return

        raise TypeError(f"Unsupported benchmark worker payload: {payload!r}")

    def _finish_worker(
        self,
        worker: _BenchmarkWorkerSlot,
        *,
        replace: bool,
    ) -> None:
        """Return a worker to idle, or replace its actor after failure."""

        old_evaluator = worker.evaluator if replace else None
        if replace:
            worker.generation += 1
            worker.evaluator = cast(
                PolicyEvaluatorRemote,
                self.cfg.remote_cfg(),
            )
            worker.configured = False
        worker.state = "idle"
        worker.future = None
        worker.request = None
        if old_evaluator is not None:
            self._close_evaluator(old_evaluator)

    @staticmethod
    def _close_evaluator(evaluator: PolicyEvaluatorRemote) -> None:
        """Close remote evaluators during normal and best-effort cleanup."""

        try:
            evaluator.close()
        except Exception:
            logger.exception("Failed to close benchmark worker evaluator.")

    @staticmethod
    def _finalize_resources(
        workers: list[_BenchmarkWorkerSlot],
        executor: ThreadPoolExecutor,
    ) -> None:
        """Fallback cleanup if the backend object is garbage-collected."""

        for worker in workers:
            future = worker.future
            if future is not None:
                future.cancel()
            RemoteBenchmarkBackend._close_evaluator(worker.evaluator)
        executor.shutdown(wait=False, cancel_futures=True)
