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

"""RoboRSI Agent services and clients; Adapter exports remain compatible."""

from __future__ import annotations

import json
import time
from typing import Any, Protocol

from roborsi.evaluation.adapter_service import (
    AdapterFacade as AdapterFacade,
)
from roborsi.evaluation.adapter_service import (
    BenchmarkAdapter as BenchmarkAdapter,
)
from roborsi.evaluation.adapter_service import _validate_action_step
from roborsi.evaluation.messages import (
    Action,
    BenchmarkQuery,
    BenchmarkRuntimeInfo,
    EpisodeStop,
    NativeResult,
    Observation,
    QueryResult,
    StartResult,
    StepResult,
    Task,
)
from roborsi.evaluation.service_protocol import (
    RoboRSIBackendError as RoboRSIBackendError,
)
from roborsi.evaluation.service_protocol import (
    RoboRSIFacade as RoboRSIFacade,
)
from roborsi.evaluation.service_protocol import (
    RoboRSIRequestError as RoboRSIRequestError,
)
from roborsi.evaluation.service_protocol import (
    _call_replayable as _call_replayable,
)
from roborsi.evaluation.service_protocol import _close_timeout
from roborsi.evaluation.service_protocol import (
    _decode as _decode,
)
from roborsi.evaluation.service_protocol import (
    _encode as _encode,
)
from roborsi.evaluation.service_protocol import (
    _negotiate as _negotiate,
)
from roborsi.evaluation.service_protocol import (
    _new_request_id as _new_request_id,
)
from roborsi.evaluation.service_protocol import (
    _parse_wire as _parse_wire,
)
from roborsi.evaluation.service_protocol import (
    _raise_backend_error as _raise_backend_error,
)
from roborsi.evaluation.service_protocol import (
    _reject_constant as _reject_constant,
)
from roborsi.evaluation.service_protocol import (
    _RequestReplay as _RequestReplay,
)
from roborsi.utils.rpc import RpcClient


class Agent(Protocol):
    """User-owned policy, independent of Benchmark imports and state.

    An optional ``resume(QueryResult)`` continues a pending BenchmarkQuery and
    returns the next decision at the same observation step.
    """

    def reset(self, task: Task) -> None: ...

    def act(
        self, observation: Observation
    ) -> Action | EpisodeStop | BenchmarkQuery: ...

    def close(self) -> None: ...


class AgentFacade(RoboRSIFacade):
    """Expose one Agent episode through versioned JSON envelopes."""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self._agent = agent
        self._replay = _RequestReplay()
        self._identity: tuple[str, str] | None = None
        self._next_step = 0
        self._started = False
        self._closed = False
        self._ended = False
        self._stop_requested = False
        self._pending_query: BenchmarkQuery | None = None
        self._pending_action_step: int | None = None
        self._recoverable_action_rejections = False
        self._rpc.update(
            {"reset": self.reset, "act": self.act, "close_episode": self.close_episode}
        )
        if callable(getattr(agent, "observe_end", None)):
            self._rpc["observe_end"] = self.observe_end
        if callable(getattr(agent, "resume", None)):
            self._rpc["resume"] = self.resume

    def reset(self, wire_task: str) -> dict[str, Any]:
        envelope = _parse_wire(wire_task)
        cached = self._replay.cached(envelope.request_id, wire_task)
        if cached is not None:
            return cached
        if self._started or self._closed:
            raise RoboRSIRequestError("Agent service owns only one episode")
        task = _decode(wire_task, Task)
        try:
            self._agent.reset(task)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "agent_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_task,
            )
        self._identity = (envelope.benchmark, envelope.episode_id)
        self._next_step = 0
        self._recoverable_action_rejections = (
            task.capabilities.get("recoverable_action_rejections") is True
        )
        self._started = True
        reply = {"ok": True, "request_id": envelope.request_id}
        self._replay.remember(envelope.request_id, wire_task, reply)
        return reply

    def act(self, wire_observation: str) -> str:
        envelope = _parse_wire(wire_observation)
        cached = self._replay.cached(envelope.request_id, wire_observation)
        if cached is not None:
            return cached
        if self._ended or self._closed or self._stop_requested:
            raise RoboRSIRequestError("Agent episode has ended")
        if self._pending_query is not None:
            raise RoboRSIRequestError(
                "Agent needs the pending query result before another act"
            )
        if self._identity != (envelope.benchmark, envelope.episode_id):
            raise RoboRSIRequestError("Agent episode identity mismatch")
        observation = _decode(wire_observation, Observation)
        if "action_rejected" in observation.data:
            if (
                not self._recoverable_action_rejections
                or observation.data["action_rejected"] is not True
                or self._pending_action_step is None
                or observation.step_index != self._pending_action_step
                or self._next_step != self._pending_action_step + 1
            ):
                raise RoboRSIRequestError(
                    "Invalid recoverable action rejection observation"
                )
            self._next_step = self._pending_action_step
        if observation.step_index != self._next_step:
            raise RoboRSIRequestError("Agent observation step is out of order")
        # Consume preceding action feedback before policy/query execution. It
        # cannot be reused to roll back an unrelated resumed decision.
        self._pending_action_step = None
        try:
            action = self._agent.act(observation)
            encoded = self._encode_decision(action, envelope.request_id)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "agent_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_observation,
            )
        self._replay.remember(envelope.request_id, wire_observation, encoded)
        return encoded

    def _encode_decision(
        self, action: Action | EpisodeStop | BenchmarkQuery, request_id: str | None
    ) -> str:
        if not isinstance(action, (Action, EpisodeStop, BenchmarkQuery)):
            raise TypeError("Agent must return Action, EpisodeStop or BenchmarkQuery")
        if action.step_index != self._next_step:
            raise ValueError("Agent action step does not match observation")
        if isinstance(action, BenchmarkQuery) and "resume" not in self._rpc:
            raise ValueError("Agent benchmark queries require resume(QueryResult)")
        encoded = _encode(action, *self._identity, request_id)
        self._pending_query = action if isinstance(action, BenchmarkQuery) else None
        if isinstance(action, EpisodeStop):
            self._stop_requested = True
        elif isinstance(action, Action):
            self._pending_action_step = action.step_index
            self._next_step += 1
        return encoded

    def resume(self, wire_reply: str) -> str:
        """Continue the same decision after its pending benchmark query."""
        envelope = _parse_wire(wire_reply)
        cached = self._replay.cached(envelope.request_id, wire_reply)
        if cached is not None:
            return cached
        pending = self._pending_query
        if pending is None or self._closed or self._ended:
            raise RoboRSIRequestError("Agent has no pending benchmark query")
        reply = _decode(wire_reply, QueryResult, self._identity)
        if (reply.step_index, reply.query_id) != (pending.step_index, pending.query_id):
            raise RoboRSIRequestError(
                "Agent query reply has a different step or query ID"
            )
        try:
            decision = self._agent.resume(reply)
            encoded = self._encode_decision(decision, envelope.request_id)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "agent_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_reply,
            )
        self._replay.remember(envelope.request_id, wire_reply, encoded)
        return encoded

    def observe_end(self, wire_step: str) -> dict[str, Any]:
        """Accept one replayable terminal tool response for capable Agents."""
        envelope = _parse_wire(wire_step)
        cached = self._replay.cached(envelope.request_id, wire_step)
        if cached is not None:
            return cached
        if not self._started or self._closed or self._ended:
            raise RoboRSIRequestError("Agent has no active episode")
        if self._pending_query is not None:
            raise RoboRSIRequestError("Agent still has a pending benchmark query")
        step = _decode(wire_step, StepResult, self._identity)
        if not (step.terminated or step.truncated):
            raise RoboRSIRequestError("Agent final feedback needs a terminal step")
        if (
            self._next_step == 0 and not self._stop_requested
        ) or step.observation.step_index != self._next_step:
            raise RoboRSIRequestError("Agent final feedback step is out of order")
        try:
            self._agent.observe_end(step)
        except Exception as exc:
            _raise_backend_error(
                exc,
                "agent_error",
                self.close_episode,
                self._replay,
                envelope.request_id,
                wire_step,
            )
        self._ended = True
        reply = {"ok": True, "request_id": envelope.request_id}
        self._replay.remember(envelope.request_id, wire_step, reply)
        return reply

    def close_episode(self) -> dict[str, bool]:
        if not self._closed:
            self._closed = True
            try:
                self._agent.close()
            finally:
                self._identity = None
        return {"ok": True}

    def close(self) -> None:
        self.close_episode()


class AgentClient:
    """Typed caller for an independently hosted AgentFacade."""

    def __init__(self, rpc: RpcClient, *, close_timeout_s: float = 5.0) -> None:
        self._rpc = rpc
        self._close_timeout_s = _close_timeout(close_timeout_s)
        self._identity: tuple[str, str] | None = None
        self._capabilities: dict[str, Any] = {}

    def reset(self, task: Task, episode_id: str, *, timeout_s: float) -> None:
        self._capabilities.clear()
        timeout_s = _negotiate(self._rpc, timeout_s, self._capabilities)
        request_id = _new_request_id()
        reply = _call_replayable(
            self._rpc,
            "reset",
            (_encode(task, task.benchmark, episode_id, request_id),),
            timeout_s,
        )
        if not isinstance(reply, dict) or reply.get("request_id") != request_id:
            raise ValueError("RoboRSI reset response request_id mismatch")
        self._identity = (task.benchmark, episode_id)

    def act(
        self, observation: Observation, *, timeout_s: float
    ) -> Action | EpisodeStop | BenchmarkQuery:
        request_id = _new_request_id()
        wire = _call_replayable(
            self._rpc,
            "act",
            (
                _encode(
                    observation,
                    observation.benchmark,
                    observation.episode_id,
                    request_id,
                ),
            ),
            timeout_s,
        )
        action = self._decision(wire, request_id)
        if action.step_index != observation.step_index:
            raise ValueError("Agent action step does not match observation")
        return action

    def _decision(
        self, wire: str, request_id: str
    ) -> Action | EpisodeStop | BenchmarkQuery:
        kind = _parse_wire(wire).kind
        model = {"episode_stop": EpisodeStop, "benchmark_query": BenchmarkQuery}.get(
            kind, Action
        )
        return _decode(wire, model, self._identity, request_id)

    def resume(
        self, reply: QueryResult, *, timeout_s: float
    ) -> Action | EpisodeStop | BenchmarkQuery:
        """Deliver a query result without starting another observation turn."""
        versions = self._capabilities.get("benchmark_query_versions", [])
        if not isinstance(versions, list) or not any(
            type(v) is int and v == 1 for v in versions
        ):
            raise RoboRSIRequestError("Agent does not advertise query resumption")
        request_id = _new_request_id()
        wire = _call_replayable(
            self._rpc,
            "resume",
            (_encode(reply, reply.benchmark, reply.episode_id, request_id),),
            timeout_s,
        )
        decision = self._decision(wire, request_id)
        if decision.step_index != reply.step_index:
            raise ValueError("Agent resumed at a different observation step")
        return decision

    def observe_end(self, step: StepResult, *, timeout_s: float) -> None:
        """Send terminal feedback when advertised; preserve older Agent services."""
        versions = self._capabilities.get("agent_feedback_versions", [])
        if not isinstance(versions, list) or not any(
            type(version) is int and version == 1 for version in versions
        ):
            return
        if not (step.terminated or step.truncated):
            raise ValueError("Agent final feedback needs a terminal step")
        observation = step.observation
        request_id = _new_request_id()
        reply = _call_replayable(
            self._rpc,
            "observe_end",
            (_encode(step, observation.benchmark, observation.episode_id, request_id),),
            timeout_s,
        )
        if not isinstance(reply, dict) or reply.get("request_id") != request_id:
            raise ValueError("RoboRSI final feedback response request_id mismatch")

    def close(self) -> None:
        try:
            self._rpc.call("close_episode", timeout_s=self._close_timeout_s)
        finally:
            self._identity = None
            self._rpc.close()


class AdapterClient:
    """Typed caller for an independently hosted AdapterFacade."""

    def __init__(self, rpc: RpcClient, *, close_timeout_s: float = 5.0) -> None:
        self._rpc = rpc
        self._close_timeout_s = _close_timeout(close_timeout_s)
        self._identity: tuple[str, str] | None = None
        self._task: tuple[str, int] | None = None
        self._capabilities: dict[str, Any] = {}
        self._recoverable_action_rejections = False

    def runtime_info(self, *, timeout_s: float) -> BenchmarkRuntimeInfo | None:
        """Read advertised metadata; an older service remains explicitly unknown."""
        started = time.monotonic()
        health = self._rpc.call("healthz", timeout_s=timeout_s)
        if not isinstance(health, dict) or not health.get("runtime_info_versions"):
            return None
        versions = health["runtime_info_versions"]
        if (
            not isinstance(versions, list)
            or any(type(v) is not int for v in versions)
            or 1 not in versions
        ):
            raise ValueError("unsupported Benchmark runtime_info version")
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Benchmark runtime_info timed out")
        raw = self._rpc.call("runtime_info", timeout_s=remaining)
        value = json.loads(raw, parse_constant=_reject_constant)
        return None if value is None else BenchmarkRuntimeInfo.from_dict(value)

    def start(self, task: Task, episode_id: str, *, timeout_s: float) -> StartResult:
        self._capabilities.clear()
        timeout_s = _negotiate(self._rpc, timeout_s, self._capabilities)
        request_id = _new_request_id()
        wire = _call_replayable(
            self._rpc,
            "start",
            (_encode(task, task.benchmark, episode_id, request_id),),
            timeout_s,
        )
        result = _decode(wire, StartResult, (task.benchmark, episode_id), request_id)
        if (result.task.task_id, result.task.seed) != (task.task_id, task.seed):
            raise ValueError("Adapter started a different task or seed")
        self._identity = (task.benchmark, episode_id)
        self._task = (result.task.task_id, result.task.seed)
        self._recoverable_action_rejections = (
            task.capabilities.get("recoverable_action_rejections") is True
            and result.task.capabilities.get("recoverable_action_rejections") is True
        )
        return result

    def step(self, action: Action, *, timeout_s: float) -> StepResult:
        request_id = _new_request_id()
        wire = _call_replayable(
            self._rpc,
            "step",
            (_encode(action, action.benchmark, action.episode_id, request_id),),
            timeout_s,
        )
        result = _decode(wire, StepResult, self._identity, request_id)
        _validate_action_step(
            result,
            action,
            recoverable_action_rejections=self._recoverable_action_rejections,
        )
        return result

    def query(self, request: BenchmarkQuery, *, timeout_s: float) -> QueryResult:
        """Ask the native owner for data or planning without advancing a step."""
        versions = self._capabilities.get("benchmark_query_versions", [])
        if not isinstance(versions, list) or not any(
            type(v) is int and v == 1 for v in versions
        ):
            raise RoboRSIRequestError("Adapter does not advertise benchmark queries")
        request_id = _new_request_id()
        wire = _call_replayable(
            self._rpc,
            "query",
            (_encode(request, request.benchmark, request.episode_id, request_id),),
            timeout_s,
        )
        reply = _decode(wire, QueryResult, self._identity, request_id)
        if (reply.step_index, reply.query_id) != (request.step_index, request.query_id):
            raise ValueError("Adapter query reply has a different step or query ID")
        return reply

    def stop(self, request: EpisodeStop, *, timeout_s: float) -> StepResult:
        """Request a replayable native end without submitting another action."""
        versions = self._capabilities.get("episode_stop_versions", [])
        if not isinstance(versions, list) or not any(
            type(version) is int and version == 1 for version in versions
        ):
            raise RoboRSIRequestError("Adapter does not advertise episode stop support")
        request_id = _new_request_id()
        wire = _call_replayable(
            self._rpc,
            "stop",
            (_encode(request, request.benchmark, request.episode_id, request_id),),
            timeout_s,
        )
        result = _decode(wire, StepResult, self._identity, request_id)
        if result.observation.step_index != request.step_index:
            raise ValueError("Adapter.stop must not advance the environment")
        if not (result.terminated or result.truncated):
            raise ValueError("Adapter.stop must return a terminal result")
        return result

    def finish(self, *, timeout_s: float) -> NativeResult:
        request_id = _new_request_id()
        wire = _call_replayable(self._rpc, "finish", (request_id,), timeout_s)
        result = _decode(wire, NativeResult, self._identity, request_id)
        if (result.task.task_id, result.task.seed) != self._task:
            raise ValueError("Adapter score task differs from started task")
        return result

    def close(self) -> None:
        try:
            self._rpc.call("close_episode", timeout_s=self._close_timeout_s)
        finally:
            self._identity = None
            self._task = None
            self._rpc.close()
