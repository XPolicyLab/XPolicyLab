"""Normalize execution errors for dispatch summaries and webhooks."""

from __future__ import annotations

from typing import Any

from station.env_client.runner import TrialRunnerError
from client_server.ws.protocol.exceptions import WsError


def normalize_execution_error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, WsError):
        error: dict[str, Any] = {"code": exc.code.value, "message": exc.message}
        if exc.details:
            error["details"] = exc.details
        return error
    if isinstance(exc, TrialRunnerError) and exc.error:
        return dict(exc.error)
    return {"code": "internal", "message": str(exc)}
