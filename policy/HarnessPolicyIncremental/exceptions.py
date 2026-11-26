"""Episode control flow and tool-result errors for the policy adapter."""

from __future__ import annotations

from typing import Any


class EpisodeFinished(BaseException):
    """Stop the program without being swallowed by ``except Exception``."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class MotionUnacknowledged(BaseException):
    """Stop rather than retry a motion with uncertain execution."""


class ToolError(RuntimeError):
    """A tool refused the call or failed to run."""

    def __init__(self, tool: str, message: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.message = message
        self.payload = payload or {}


def unwrap(result: Any, tool: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Split a tool payload from its observation envelope, if present."""
    payload = getattr(result, "result", result)
    if not isinstance(payload, dict):
        raise ToolError(tool, f"backend returned {type(payload).__name__}, not a dict")
    log = payload.get("log")
    if isinstance(log, dict) and "state" in payload:
        inner = log.get("result")
        if not isinstance(inner, dict):
            inner = {"value": inner}
        return inner, payload
    return payload, None


def raise_for_error(tool: str, payload: dict[str, Any]) -> None:
    """Raise when a tool payload carries an error."""
    error = payload.get("error")
    if error:
        raise ToolError(tool, str(error), payload)
