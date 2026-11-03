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

"""Shared RoboRSI wire validation, replay and RPC capability negotiation."""

from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any, Callable, TypeVar

from roborsi.evaluation.messages import (
    InteractionMessage,
    from_envelope,
    to_envelope,
)
from roborsi.evaluation.protocol import (
    INTERACTION_SCHEMA_VERSION,
    InteractionEnvelope,
)
from roborsi.utils.rpc import RpcClient, RpcError, RpcFacade, RpcTransportError

MessageT = TypeVar("MessageT", bound=InteractionMessage)


class RoboRSIRequestError(ValueError):
    """A malformed or incompatible RoboRSI request, before native execution."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.rpc_code = code


class RoboRSIBackendError(ValueError):
    """A native Agent or Adapter failure after accepting a request."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.rpc_code = code


def _close_timeout(value: float) -> float:
    """Validate explicit cleanup waiting; zero leaves normal reads unbounded."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("close_timeout_s must be finite and nonnegative")
    return value


def _raise_backend_error(
    exc: Exception,
    code: str,
    close_episode: Callable[[], Any],
    replay: _RequestReplay,
    request_id: str | None,
    request: str,
) -> None:
    """Preserve the native failure even when resource cleanup also fails."""
    if isinstance(exc, RoboRSIBackendError) and exc.rpc_code == "infrastructure_error":
        code = exc.rpc_code
    message = str(exc) or type(exc).__name__
    try:
        close_episode()
    except Exception as cleanup:
        message += f"; cleanup failed: {type(cleanup).__name__}: {cleanup}"
    failure = RoboRSIBackendError(message, code=code)
    replay.remember(request_id, request, failure)
    raise failure from exc


class RoboRSIFacade(RpcFacade):
    """Advertise the interaction versions supported by a RoboRSI service."""

    def _builtin_dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        result = super()._builtin_dispatch(method, args, kwargs)
        if method == "healthz":
            result = {
                **result,
                "roborsi_protocol_versions": [INTERACTION_SCHEMA_VERSION],
            }
            if "runtime_info" in self._rpc:
                result["runtime_info_versions"] = [1]
            if "observe_end" in self._rpc:
                result["agent_feedback_versions"] = [1]
            if "stop" in self._rpc:
                result["episode_stop_versions"] = [1]
            if "query" in self._rpc or "resume" in self._rpc:
                result["benchmark_query_versions"] = [1]
        return result


def _parse_wire(value: str) -> InteractionEnvelope:
    if not isinstance(value, str):
        raise RoboRSIRequestError("RoboRSI RPC message must be JSON text")
    try:
        document = json.loads(
            value, parse_constant=lambda constant: _reject_constant(constant)
        )
    except ValueError as exc:
        raise RoboRSIRequestError(f"invalid RoboRSI JSON: {exc}") from exc
    if (
        isinstance(document, dict)
        and "schema_version" in document
        and (
            type(document["schema_version"]) is not int
            or document["schema_version"] != INTERACTION_SCHEMA_VERSION
        )
    ):
        raise RoboRSIRequestError(
            "unsupported interaction schema_version", code="unsupported_version"
        )
    try:
        return InteractionEnvelope.from_dict(document)
    except (TypeError, ValueError) as exc:
        raise RoboRSIRequestError(str(exc)) from exc


def _reject_constant(constant: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {constant}")


def _decode(
    value: str,
    expected: type[MessageT],
    identity: tuple[str, str] | None = None,
    request_id: str | None = None,
) -> MessageT:
    envelope = _parse_wire(value)
    if request_id is not None and envelope.request_id != request_id:
        raise RoboRSIRequestError("RoboRSI response request_id mismatch")
    if identity is not None and (envelope.benchmark, envelope.episode_id) != identity:
        raise RoboRSIRequestError("RoboRSI response episode identity mismatch")
    try:
        message = from_envelope(envelope)
    except (TypeError, ValueError) as exc:
        raise RoboRSIRequestError(str(exc)) from exc
    if type(message) is not expected:
        raise RoboRSIRequestError(f"expected {expected.__name__} message")
    return message


def _encode(
    message: InteractionMessage,
    benchmark: str,
    episode_id: str,
    request_id: str | None = None,
) -> str:
    document = to_envelope(
        message, benchmark=benchmark, episode_id=episode_id, request_id=request_id
    )
    return json.dumps(document.as_dict(), allow_nan=False)


def _new_request_id() -> str:
    return uuid.uuid4().hex


def _negotiate(
    rpc: RpcClient, timeout_s: float, capabilities: dict | None = None
) -> float:
    """Check version before mutation, sharing the initiating call's timeout."""
    started = time.monotonic()
    health = rpc.call("healthz", timeout_s=30.0 if timeout_s == 0 else timeout_s)
    versions = (
        health.get("roborsi_protocol_versions") if isinstance(health, dict) else None
    )
    if not isinstance(versions, list) or not any(
        type(version) is int and version == INTERACTION_SCHEMA_VERSION
        for version in versions
    ):
        raise RpcError(
            "healthz",
            f"RoboRSI protocol version {INTERACTION_SCHEMA_VERSION} is unsupported",
            code="unsupported_version",
        )
    remaining = timeout_s - (time.monotonic() - started)
    if timeout_s == 0:
        if capabilities is not None:
            capabilities.update(health)
        return 0
    if remaining <= 0:
        raise TimeoutError("RoboRSI protocol negotiation timed out")
    if capabilities is not None:
        capabilities.update(health)
    return remaining


def _call_replayable(rpc: RpcClient, method: str, args: tuple, timeout_s: float) -> Any:
    """Retry a lost response once with the exact request inside one time budget."""
    started = time.monotonic()
    try:
        return rpc.call(method, args=args, timeout_s=timeout_s / 2)
    except (RpcTransportError, TimeoutError):
        if timeout_s == 0:
            return rpc.call(method, args=args, timeout_s=0)
        remaining = timeout_s - (time.monotonic() - started)
        if remaining <= 0:
            raise
        return rpc.call(method, args=args, timeout_s=remaining)


class _RequestReplay:
    """Keep completed single-episode calls available for safe RPC retries."""

    def __init__(self) -> None:
        self._replies: dict[str, tuple[str, Any]] = {}

    def cached(self, request_id: str | None, request: str) -> Any | None:
        if request_id is None or request_id not in self._replies:
            return None
        previous, reply = self._replies[request_id]
        if previous != request:
            raise RoboRSIRequestError("request_id was reused with a different request")
        if isinstance(reply, RoboRSIBackendError):
            raise RoboRSIBackendError(str(reply), code=reply.rpc_code)
        return reply

    def remember(self, request_id: str | None, request: str, reply: Any) -> None:
        if request_id is not None:
            self._replies[request_id] = (request, reply)
