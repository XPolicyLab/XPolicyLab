"""Separate native execution time from installation and service startup time.

The controller publishes this host-only receipt when execution starts. A case
supervisor may then enforce the same cumulative execution deadline without
charging cold startup against the original agent's episode budget.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

_FORMAT = "roborsi_execution_budget_v1"
_PHASES = {"replay", "agent"}


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return float(value)


def write_execution_budget(
    path: Path,
    remaining_seconds: float,
    *,
    phase: str = "agent",
    started_at: float | None = None,
) -> dict:
    """Atomically publish a measured execution start and its remaining budget.

    A resumed controller writes ``replay`` before native action replay, then
    writes ``agent`` with the replay time already deducted. No prompt, state,
    credentials or model content is included in this supervisor receipt.
    """
    if phase not in _PHASES:
        raise ValueError("Execution phase must be replay or agent")
    record = {
        "format": _FORMAT,
        "phase": phase,
        "started_at": _number(
            time.time() if started_at is None else started_at, "started_at"
        ),
        "remaining_seconds": _number(
            remaining_seconds, "remaining_seconds", positive=True
        ),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".execution-budget-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return record


class CaseWatchdog:
    """An independent startup limit followed by a non-extendable execution limit.

    ``remaining`` and the controller receipt use epoch seconds so a supervisor
    can read them across processes. All later execution receipts can shorten
    the deadline; they can never renew an earlier execution budget.
    """

    def __init__(
        self, startup_seconds: float = 1800.0, *, started_at: float | None = None
    ) -> None:
        start = _number(
            time.time() if started_at is None else started_at, "started_at"
        )
        self.startup_deadline = start + _number(
            startup_seconds, "startup_seconds", positive=True
        )
        self.execution_deadline: float | None = None
        self.phase = "startup"

    def remaining(self, path: Path, *, now: float | None = None) -> float:
        """Read a receipt if present, then return seconds until this case expires."""
        current = _number(time.time() if now is None else now, "now")
        path = Path(path)
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise ValueError("Execution budget receipt must be a regular file")
            record = json.loads(path.read_text())
            if record.get("format") != _FORMAT or record.get("phase") not in _PHASES:
                raise ValueError("Unsupported execution budget receipt")
            start = _number(record.get("started_at"), "started_at")
            remaining = _number(
                record.get("remaining_seconds"), "remaining_seconds", positive=True
            )
            if start > current + 5:
                raise ValueError("Execution start is later than the supervisor clock")
            # A late marker cannot revive an expired startup attempt.
            if self.execution_deadline is None and start > self.startup_deadline:
                return self.startup_deadline - current
            deadline = start + remaining
            self.execution_deadline = (
                deadline
                if self.execution_deadline is None
                else min(self.execution_deadline, deadline)
            )
            self.phase = record["phase"]
        deadline = (
            self.execution_deadline
            if self.execution_deadline is not None
            else self.startup_deadline
        )
        return deadline - current
