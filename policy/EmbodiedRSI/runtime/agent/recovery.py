"""Recover an interrupted model connection without restarting the simulator."""

from __future__ import annotations

import json
import re
from pathlib import Path

NETWORK_ERRORS = (
    "stream disconnected", "transport error", "network error", "connection reset",
    "no account slot", "error sending request", "timed out", "rate limit", "429",
)
RESUME_PROMPT = (
    "The model connection was interrupted. Continue this same session. "
    "The simulator, workspace, Python variables and execution budget are unchanged. "
    "First run scripts/env.py status and inspect the latest observation/feedback. "
    "Do not repeat a previously completed submission merely because its response was interrupted. "
    "Continue the original phase goal and call scripts/env.py finish when done."
)


def rollout_state(path: Path) -> tuple[str | None, str | None]:
    """Read identity and terminal transport errors without loading image history."""
    if not path.is_file():
        return None, None
    with path.open("rb") as stream:
        first = stream.readline()
        stream.seek(0, 2)
        size = stream.tell()
        start = max(0, size - 256 * 1024)
        stream.seek(start)
        lines = stream.read().splitlines()
    try:
        metadata = json.loads(first).get("payload", {})
        session_id = metadata.get("id") or metadata.get("session_id")
    except (ValueError, AttributeError):
        session_id = None
    error = None
    for line in lines[1:] if start else lines:
        try:
            payload = json.loads(line).get("payload", {})
        except (ValueError, AttributeError):
            continue
        if payload.get("type") == "task_started":
            error = None
        if payload.get("type") == "task_complete":
            terminal = payload.get("error")
            error = terminal.get("message") if isinstance(terminal, dict) else None
    return session_id, error


def recovery_reason(*, state, returncode, idle_seconds, idle_limit, error):
    """A running action, closed episode or normal agent exit is never restarted."""
    if state.get("closed") or state.get("valid") is not True or state.get("reason") != "active":
        return None
    if returncode == 0:
        return None
    if returncode is not None:
        transient_http = bool(error and re.search(
            r"\b(?:http(?: status)?|unexpected status)\s+5\d{2}\b", error, re.IGNORECASE
        ))
        if error and (transient_http or any(marker in error.lower() for marker in NETWORK_ERRORS)):
            return "agent_transport_error"
        return None
    if idle_seconds >= idle_limit:
        return "agent_response_stalled"
    return None


def resume_argv(argv, session_id):
    """Reuse the same container mounts and staged HOME, resuming only this thread."""
    result = list(argv)
    index = result.index("codex")
    if result[index + 1] != "exec":
        raise ValueError("Only Codex exec sessions support transport recovery")
    result.insert(index + 2, "resume")
    result[-1:] = [session_id, RESUME_PROMPT]
    return result
