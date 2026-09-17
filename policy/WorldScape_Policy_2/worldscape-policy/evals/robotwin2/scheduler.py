from __future__ import annotations

import queue
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from typing import Any, Protocol


@dataclass(frozen=True)
class RunTask:
    """Ask a long-lived worker to evaluate one task."""

    sequence: int
    payload: Any


@dataclass(frozen=True)
class StopWorker:
    """Ask a worker to release its runtime and exit."""


WorkerCommand = RunTask | StopWorker


@dataclass(frozen=True)
class WorkerReady:
    worker_id: int


@dataclass(frozen=True)
class TaskFinished:
    worker_id: int
    sequence: int
    result: Any


@dataclass(frozen=True)
class WorkerFailed:
    worker_id: int
    sequence: int | None
    error_type: str
    error_message: str
    traceback: str


WorkerEvent = WorkerReady | TaskFinished | WorkerFailed


class Worker(Protocol):
    worker_id: int

    def start(self) -> None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...


class CommandQueue(Protocol):
    def put(self, command: WorkerCommand) -> None: ...


class EventSource(Protocol):
    def receive(self, timeout: float | None = None) -> WorkerEvent: ...


class DynamicTaskScheduler:
    """Dispatch from one global queue whenever any worker becomes free."""

    def __init__(
        self,
        workers: Iterable[Worker],
        commands: CommandQueue,
        events: EventSource,
        *,
        poll_interval_s: float = 1.0,
    ) -> None:
        self.workers = tuple(workers)
        self.commands = commands
        self.events = events
        self.poll_interval_s = float(poll_interval_s)
        if not self.workers:
            raise ValueError("At least one worker is required")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        ids = [worker.worker_id for worker in self.workers]
        if len(ids) != len(set(ids)):
            raise ValueError("worker_id values must be unique")

    def run(
        self,
        tasks: Iterable[Any],
        *,
        on_result: Callable[[int, Any], None] | None = None,
    ) -> list[Any]:
        pending = list(enumerate(tasks))
        results: dict[int, Any] = {}
        ready: set[int] = set()
        workers = {worker.worker_id: worker for worker in self.workers}
        started: list[Worker] = []
        clean_shutdown = False
        try:
            for worker in self.workers:
                worker.start()
                started.append(worker)

            while len(ready) < len(self.workers):
                event = self._receive()
                if isinstance(event, WorkerFailed):
                    raise _worker_error(event)
                if not isinstance(event, WorkerReady):
                    raise RuntimeError(
                        "Worker emitted a task result before becoming ready"
                    )
                if event.worker_id not in workers:
                    raise RuntimeError(f"Unknown worker id {event.worker_id}")
                if event.worker_id in ready:
                    raise RuntimeError(
                        f"Worker {event.worker_id} became ready more than once"
                    )
                ready.add(event.worker_id)

            for sequence, payload in pending:
                self.commands.put(RunTask(sequence, payload))

            while len(results) < len(pending):
                event = self._receive()
                if isinstance(event, WorkerFailed):
                    raise _worker_error(event)
                if isinstance(event, WorkerReady):
                    raise RuntimeError(
                        f"Worker {event.worker_id} became ready more than once"
                    )
                if event.worker_id not in workers:
                    raise RuntimeError(f"Unknown worker id {event.worker_id}")
                if event.sequence < 0 or event.sequence >= len(pending):
                    raise RuntimeError(
                        f"Worker {event.worker_id} returned unknown task "
                        f"{event.sequence}"
                    )
                if event.sequence in results:
                    raise RuntimeError(f"Task {event.sequence} completed more than once")
                results[event.sequence] = event.result
                if on_result is not None:
                    on_result(event.sequence, event.result)

            # Keep idle workers blocked on the command queue until every task
            # result has reached the manager. Sending stop commands earlier
            # lets fast workers exit while slower workers are still running;
            # _receive() then cannot distinguish that expected exit from a
            # crashed worker.
            for _ in self.workers:
                self.commands.put(StopWorker())

            for worker in self.workers:
                worker.join()
            clean_shutdown = True
        finally:
            if not clean_shutdown:
                for worker in started:
                    if worker.is_alive():
                        worker.terminate()
                for worker in started:
                    worker.join(timeout=5.0)

        return [results[index] for index in range(len(pending))]

    def _receive(self) -> WorkerEvent:
        dead_since: float | None = None
        exit_grace_s = max(5.0, 5 * self.poll_interval_s)
        while True:
            try:
                return self.events.receive(timeout=self.poll_interval_s)
            except queue.Empty:
                dead = [worker.worker_id for worker in self.workers if not worker.is_alive()]
                if dead:
                    dead_since = dead_since or time.monotonic()
                    if time.monotonic() - dead_since >= exit_grace_s:
                        raise RuntimeError(
                            "Evaluation worker(s) exited without an event after "
                            f"{exit_grace_s:.1f}s grace period: {dead}"
                        )
                else:
                    dead_since = None


class QueueEventSource:
    def __init__(self, events: Queue) -> None:
        self.events = events

    def receive(self, timeout: float | None = None) -> WorkerEvent:
        return self.events.get(timeout=timeout)


class ProcessWorker:
    """Spawn-backed worker consuming the shared command queue."""

    def __init__(
        self,
        worker_id: int,
        *,
        device_id: int,
        context: BaseContext,
        commands: Queue,
        events: Queue,
        runner_factory: Callable[[int], Callable[[Any], Any]],
    ) -> None:
        self.worker_id = worker_id
        self._process = context.Process(
            target=_worker_main,
            args=(
                worker_id,
                device_id,
                commands,
                events,
                runner_factory,
            ),
            name=f"robotwin2-gpu-{device_id}-worker-{worker_id}",
        )

    def start(self) -> None:
        self._process.start()

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._process.join(timeout)

    def terminate(self) -> None:
        self._process.terminate()


def _worker_main(
    worker_id: int,
    device_id: int,
    commands: Queue,
    events: Queue,
    runner_factory: Callable[[int], Callable[[Any], Any]],
) -> None:
    sequence: int | None = None
    try:
        # The factory owns model/runtime construction and runs exactly once.
        run_task = runner_factory(device_id)
        events.put(WorkerReady(worker_id))
        while True:
            command = commands.get()
            if isinstance(command, StopWorker):
                return
            if not isinstance(command, RunTask):
                raise TypeError(f"Unsupported worker command: {type(command).__name__}")
            sequence = command.sequence
            result = run_task(command.payload)
            events.put(TaskFinished(worker_id, sequence, result))
            sequence = None
    except BaseException as exc:  # noqa: BLE001 - cross-process failure transport
        events.put(
            WorkerFailed(
                worker_id=worker_id,
                sequence=sequence,
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )
        )


def _worker_error(event: WorkerFailed) -> RuntimeError:
    task = "during initialization" if event.sequence is None else f"on task {event.sequence}"
    return RuntimeError(
        f"Worker {event.worker_id} failed {task}: "
        f"{event.error_type}: {event.error_message}\n{event.traceback}"
    )


__all__ = [
    "CommandQueue",
    "DynamicTaskScheduler",
    "EventSource",
    "ProcessWorker",
    "QueueEventSource",
    "RunTask",
    "StopWorker",
    "TaskFinished",
    "Worker",
    "WorkerFailed",
    "WorkerReady",
]
