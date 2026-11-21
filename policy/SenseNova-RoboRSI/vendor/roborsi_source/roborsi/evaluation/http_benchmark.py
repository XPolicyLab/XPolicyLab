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

"""Discover and run externally deployed Benchmarks without local assets or Docker."""

from __future__ import annotations

import hashlib
import json
from contextlib import AbstractContextManager
from pathlib import Path

from roborsi.benchmarks.registry import RegisteredTask
from roborsi.evaluation.benchmark_client import BenchmarkClient
from roborsi.evaluation.deadline import CaseDeadline
from roborsi.evaluation.messages import (
    BenchmarkRuntimeInfo,
    BenchmarkRuntimeRecord,
    Task,
)
from roborsi.evaluation.protocol import RunProvenance, TaskIdentity
from roborsi.utils.rpc.http_abort import interrupt_http_calls


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class HttpBenchmarkLock:
    """Snapshot API metadata; this never claims Docker verification of a remote host."""

    def __init__(
        self, endpoint: str, *, token: str | None = None, timeout_s: float = 30.0
    ) -> None:
        self.endpoint, self.token, self.timeout_s = endpoint, token, timeout_s
        client = self.client()
        _, description = client.transport.request(
            "GET", "/v1/benchmark", timeout_s=timeout_s
        )
        info = description.get("runtime")
        if info is None:
            raise ValueError(
                "remote batch evaluation requires Benchmark runtime and scoring information"
            )
        self.info = BenchmarkRuntimeInfo.from_dict(info)
        if description["benchmark"] != self.info.benchmark:
            raise ValueError("remote Benchmark identity differs")
        self.benchmark = self.info.benchmark
        self.assets = description["assets"]
        self._tasks = client.tasks(timeout_s=timeout_s)
        if not self._tasks or any(t.benchmark != self.benchmark for t in self._tasks):
            raise ValueError("remote task catalog does not match Benchmark")
        identities = [(t.task_id, t.seed) for t in self._tasks]
        if len(set(identities)) != len(identities):
            raise ValueError("remote catalog contains duplicate task/seed pairs")
        self.manifest_sha256 = _digest(
            {"description": description, "tasks": [t.as_dict() for t in self._tasks]}
        )

    def client(self) -> BenchmarkClient:
        return BenchmarkClient(self.endpoint, token=self.token)

    def as_dict(self) -> dict:
        return {"benchmark": self.benchmark, "evaluation": dict(self.info.evaluation)}

    def list_tasks(self) -> tuple[TaskIdentity, ...]:
        return tuple(TaskIdentity(t.benchmark, t.task_id, t.seed) for t in self._tasks)

    def task_message(self, task_id: str, seed: int) -> Task:
        for task in self._tasks:
            if (task.task_id, task.seed) == (task_id, seed):
                return Task.from_dict(task.as_dict())
        raise KeyError(f"undeclared remote task/seed: {task_id}/{seed}")

    def task(self, task_id: str, seed: int) -> RegisteredTask:
        task = self.task_message(task_id, seed)
        return RegisteredTask(
            identity=TaskIdentity(task.benchmark, task_id, seed),
            instruction=task.instruction,
            max_steps=task.max_steps,
            native=task.native,
            capabilities=task.capabilities,
        )

    def provenance(self, agent_version: str | None = None) -> RunProvenance:
        return RunProvenance(
            benchmark_version=self.info.benchmark_version,
            benchmark_commit=self.info.benchmark_commit,
            adapter_version=self.info.adapter_version,
            assets_version=_digest(self.assets),
            agent_version=agent_version,
        )

    def verify(self) -> None:
        current = HttpBenchmarkLock(
            self.endpoint, token=self.token, timeout_s=self.timeout_s
        )
        if current.manifest_sha256 != self.manifest_sha256:
            raise ValueError("remote Benchmark metadata or task catalog changed")


class HttpBenchmarkRegistry:
    """Query one or more remote catalogs through the existing Runner interface."""

    def __init__(
        self,
        endpoints: dict[str, str],
        *,
        token: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._locks = {}
        for name, endpoint in endpoints.items():
            lock = HttpBenchmarkLock(endpoint, token=token, timeout_s=timeout_s)
            if lock.benchmark != name:
                raise ValueError("remote Benchmark name differs from requested name")
            self._locks[name] = lock

    def lock(self, benchmark: str) -> HttpBenchmarkLock:
        lock = self._locks[benchmark]
        lock.verify()
        return lock


class _HttpBenchmarkSession(AbstractContextManager):
    def __init__(self, lock: HttpBenchmarkLock, deadline: CaseDeadline | None) -> None:
        self.lock, self.deadline = lock, deadline
        self.client = lock.client()
        self.runtime_record = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.client.close()
        if self.deadline is not None:
            self.deadline.check()
        with interrupt_http_calls(self.deadline.check if self.deadline else None):
            self.lock.verify()
        self.runtime_record = BenchmarkRuntimeRecord(
            info=self.lock.info,
            assets_version=_digest(self.lock.assets),
            assets=tuple(self.lock.assets),
            configuration={"transport": "http", "verification": "service_report"},
            verified_after_close=True,
        )
        if self.client.downloads_complete and self.client.transport.episode_id:
            self.client.delete(self.client.transport.episode_id, timeout_s=5.0)


class HttpBenchmarkExecutor:
    """Own an HTTP episode connection, never the remote process or filesystem."""

    def start_adapter(
        self,
        lock: HttpBenchmarkLock,
        case_dir: Path,
        *,
        deadline: CaseDeadline | None = None,
    ) -> _HttpBenchmarkSession:
        if deadline is not None:
            deadline.check()
        if not isinstance(lock, HttpBenchmarkLock):
            raise TypeError("HTTP Benchmark executor requires an HTTP catalog")
        with interrupt_http_calls(deadline.check if deadline else None):
            lock.verify()
        return _HttpBenchmarkSession(lock, deadline)
