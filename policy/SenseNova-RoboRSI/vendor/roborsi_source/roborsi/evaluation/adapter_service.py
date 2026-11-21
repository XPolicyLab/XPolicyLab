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

"""Benchmark-owned native Adapter contract and RPC facade."""

from __future__ import annotations

import json
from typing import Protocol

from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    BenchmarkRuntimeInfo,
    EpisodeStop,
    NativeResult,
    QueryResult,
    StartResult,
    StepResult,
    Task,
)
from roborsi.evaluation.service_protocol import (
    RoboRSIFacade,
    RoboRSIRequestError,
    _decode,
    _encode,
    _parse_wire,
    _raise_backend_error,
    _RequestReplay,
)


class BenchmarkAdapter(Protocol):
    """Benchmark-owned environment and native evaluator.

    An optional ``stop(EpisodeStop) -> StepResult`` ends the native episode
    without executing an action. Its terminal observation retains step_index.
    An optional ``query(BenchmarkQuery) -> QueryResult`` provides native data or
    planning without advancing the scene, fetching observations or scoring.
    """

    def start(self, task: Task, episode_id: str) -> StartResult: ...

    def step(self, action: Action) -> StepResult: ...

    def finish(self) -> NativeResult: ...

    def close(self) -> None: ...


def _validate_action_step(
    result: StepResult, action: Action, *, recoverable_action_rejections: bool
) -> bool:
    """Validate native progress and return whether this action was rejected.

    Only an explicitly enabled task can report nonterminal, zero-movement
    validation feedback. Both markers are required so ordinary observations
    cannot silently weaken the one-action/one-step contract.
    """
    native_marker = "action_rejected" in result.native
    observation_marker = "action_rejected" in result.observation.data
    rejected = native_marker or observation_marker
    if rejected:
        if (
            not recoverable_action_rejections
            or result.native.get("action_rejected") is not True
            or result.observation.data.get("action_rejected") is not True
            or result.terminated
            or result.truncated
            or result.termination_reason is not None
            or result.observation.step_index != action.step_index
        ):
            raise ValueError("Invalid recoverable action rejection feedback")
    elif result.observation.step_index != action.step_index + 1:
        raise ValueError("Adapter observation step is out of order")
    return rejected


class AdapterFacade(RoboRSIFacade):
    """Expose one native Benchmark episode through versioned JSON envelopes."""

    def __init__(self, adapter: BenchmarkAdapter) -> None:
        super().__init__()
        self._adapter = adapter
        self._replay = _RequestReplay()
        self._identity: tuple[str, str] | None = None
        self._task: tuple[str, int] | None = None
        self._next_step = 0
        self._max_steps = 0
        self._recoverable_action_rejections = False
        self._ended = False
        self._started = False
        self._closed = False
        self._rpc.update(
            {
                "start": self.start,
                "step": self.step,
                "finish": self.finish,
                "close_episode": self.close_episode,
            }
        )
        if callable(getattr(adapter, "runtime_info", None)):
            self._rpc["runtime_info"] = self.runtime_info
        if callable(getattr(adapter, "stop", None)):
            self._rpc["stop"] = self.stop
        if callable(getattr(adapter, "query", None)):
            self._rpc["query"] = self.query

    def runtime_info(self) -> str:
        """Read optional Benchmark metadata even after the episode has closed."""
        info = self._adapter.runtime_info()
        if info is None:
            return "null"
        if not isinstance(info, BenchmarkRuntimeInfo):
            raise TypeError("Adapter.runtime_info must return BenchmarkRuntimeInfo")
        return json.dumps(info.as_dict(), allow_nan=False)

    def start(self, wire_task: str) -> str:
        envelope = _parse_wire(wire_task)
        cached = self._replay.cached(envelope.request_id, wire_task)
        if cached is not None:
            return cached
        if self._started or self._closed:
            raise RoboRSIRequestError("Adapter service owns only one episode")
        task = _decode(wire_task, Task)
        try:
            result = self._adapter.start(task, envelope.episode_id)
            if not isinstance(result, StartResult):
                raise TypeError("Adapter.start must return StartResult")
            encoded = _encode(
                result, envelope.benchmark, envelope.episode_id, envelope.request_id
            )
            if (result.task.task_id, result.task.seed) != (task.task_id, task.seed):
                raise ValueError("Adapter started a different task or seed")
        except Exception as exc:
            _raise_backend_error(
                exc,
                "adapter_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_task,
            )
        self._identity = (envelope.benchmark, envelope.episode_id)
        self._task = (result.task.task_id, result.task.seed)
        self._next_step = 0
        self._max_steps = result.task.max_steps
        self._recoverable_action_rejections = (
            task.capabilities.get("recoverable_action_rejections") is True
            and result.task.capabilities.get("recoverable_action_rejections") is True
        )
        self._ended = False
        self._started = True
        self._replay.remember(envelope.request_id, wire_task, encoded)
        return encoded

    def step(self, wire_action: str) -> str:
        envelope = _parse_wire(wire_action)
        cached = self._replay.cached(envelope.request_id, wire_action)
        if cached is not None:
            return cached
        if self._identity != (envelope.benchmark, envelope.episode_id):
            raise RoboRSIRequestError("Adapter episode identity mismatch")
        if self._ended:
            raise RoboRSIRequestError("Adapter episode has ended")
        action = _decode(wire_action, Action)
        if action.step_index != self._next_step:
            raise RoboRSIRequestError("Adapter action step is out of order")
        if self._next_step >= self._max_steps:
            raise RoboRSIRequestError("Adapter execution limit reached")
        try:
            result = self._adapter.step(action)
            if not isinstance(result, StepResult):
                raise TypeError("Adapter.step must return StepResult")
            rejected = _validate_action_step(
                result,
                action,
                recoverable_action_rejections=self._recoverable_action_rejections,
            )
            encoded = _encode(result, *self._identity, envelope.request_id)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "adapter_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_action,
            )
        if not rejected:
            self._next_step += 1
        self._ended = result.terminated or result.truncated
        self._replay.remember(envelope.request_id, wire_action, encoded)
        return encoded

    def query(self, wire_query: str) -> str:
        """Run a benchmark query at the current step without submitting an action."""
        envelope = _parse_wire(wire_query)
        cached = self._replay.cached(envelope.request_id, wire_query)
        if cached is not None:
            return cached
        request = _decode(wire_query, BenchmarkQuery, self._identity)
        if (
            self._identity is None
            or self._ended
            or request.step_index != self._next_step
        ):
            raise RoboRSIRequestError("Adapter query needs the active observation step")
        query = getattr(self._adapter, "query", None)
        if not callable(query):
            raise RoboRSIRequestError("Adapter does not support benchmark queries")
        try:
            result = query(request)
            if not isinstance(result, QueryResult):
                raise TypeError("Adapter.query must return QueryResult")
            if (result.step_index, result.query_id) != (
                request.step_index,
                request.query_id,
            ):
                raise ValueError("Adapter query reply has a different step or query ID")
            encoded = _encode(result, *self._identity, envelope.request_id)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "adapter_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_query,
            )
        self._replay.remember(envelope.request_id, wire_query, encoded)
        return encoded

    def stop(self, wire_stop: str) -> str:
        """End an episode without incrementing its physical step counter."""
        envelope = _parse_wire(wire_stop)
        cached = self._replay.cached(envelope.request_id, wire_stop)
        if cached is not None:
            return cached
        if self._identity != (envelope.benchmark, envelope.episode_id):
            raise RoboRSIRequestError("Adapter episode identity mismatch")
        if self._ended:
            raise RoboRSIRequestError("Adapter episode has ended")
        request = _decode(wire_stop, EpisodeStop)
        if request.step_index != self._next_step:
            raise RoboRSIRequestError("Adapter stop step is out of order")
        stop = getattr(self._adapter, "stop", None)
        if not callable(stop):
            raise RoboRSIRequestError("Adapter does not support episode stop")
        try:
            result = stop(request)
            if not isinstance(result, StepResult):
                raise TypeError("Adapter.stop must return StepResult")
            if result.observation.step_index != request.step_index:
                raise ValueError("Adapter.stop must not advance the environment")
            if not (result.terminated or result.truncated):
                raise ValueError("Adapter.stop must return a terminal result")
            encoded = _encode(result, *self._identity, envelope.request_id)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "adapter_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_stop,
            )
        self._ended = True
        self._replay.remember(envelope.request_id, wire_stop, encoded)
        return encoded

    def finish(self, request_id: str | None = None) -> str:
        if request_id is not None and (
            not isinstance(request_id, str) or not request_id.strip()
        ):
            raise RoboRSIRequestError("request_id must be non-empty text")
        cached = self._replay.cached(request_id, "finish")
        if cached is not None:
            return cached
        if self._identity is None or not self._ended:
            raise RoboRSIRequestError("Adapter episode must end before finish")
        try:
            result = self._adapter.finish()
            if not isinstance(result, NativeResult):
                raise TypeError("Adapter.finish must return NativeResult")
            if (result.task.task_id, result.task.seed) != self._task:
                raise ValueError("Adapter score task differs from started task")
            encoded = _encode(result, *self._identity, request_id)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "adapter_error",
                self.close_episode,
                self._replay,
                request_id,
                "finish",
            )
        self.close_episode()
        self._replay.remember(request_id, "finish", encoded)
        return encoded

    def close_episode(self) -> dict[str, bool]:
        if not self._closed:
            self._closed = True
            try:
                self._adapter.close()
            finally:
                self._identity = None
                self._task = None
                self._ended = False
        return {"ok": True}

    def close(self) -> None:
        self.close_episode()
