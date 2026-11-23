# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.


from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


class Backend(Protocol):
    """Minimal dispatch contract implemented by every backend."""

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> Any:
        """Run one primitive and return its simulator result."""

    def solved(self) -> bool:
        """Whether the environment's own success predicate has fired."""

    def artifact(self, name: str, step: int = -1) -> bytes:
        """Return the bytes of a recorded observation artifact."""

    def env_steps(self) -> int | None:
        """Total simulator steps executed so far (``None`` if unknown)."""

    def cancel(self) -> None:
        """Interrupt the primitive that is currently executing, if any."""


class ToolError(RuntimeError):
    """A primitive refused the call or failed while running.

    Raised for bad arguments, service failures and out-of-range lookups.
    Measured outcomes (a grasp that did not lift, a servo that stopped short,
    a segmentation with no mask) are *not* errors: they are returned as
    result fields so a policy can branch on them.
    """

    def __init__(self, tool: str, message: str, payload: dict[str, Any] | None = None):
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.message = message
        self.payload = payload or {}


class EpisodeFinished(BaseException):  # noqa: N818 - deliberate BaseException
    """The episode is over; the policy should return.

    This derives from ``BaseException`` (like ``KeyboardInterrupt``) so that a
    policy's ``except Exception`` blocks do not swallow it.  ``reason`` is one
    of ``"terminated"`` (task success), ``"truncated"`` (benchmark ended the
    episode), ``"timeout"`` (host time limit) or ``"cancelled"``.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def unwrap(result: Any, tool: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Split a backend result into ``(primitive_payload, state_envelope)``.

    The host replaces the result of every state-changing tool with an observation
    envelope (``step``, ``state``, ``terminated``, ``log.result`` ...).  The
    primitive's own return value lives at ``log.result``.  Read-only tools
    return their payload directly, in which case the envelope is ``None``.
    """

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
    """Raise :class:`ToolError` when a payload carries a backend error."""

    error = payload.get("error")
    if error:
        raise ToolError(tool, str(error), payload)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class CallRecord:
    """One ``robo.<tool>(...)`` call as seen by the library."""

    index: int
    tool: str
    kwargs: dict[str, Any]
    started_at: float
    elapsed_s: float
    ok: bool
    stateful: bool
    env_steps: int | None = None
    error: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["kwargs"] = {k: _jsonable(v) for k, v in self.kwargs.items()}
        record["summary"] = {k: _jsonable(v) for k, v in self.summary.items()}
        return record


class Ledger:
    """Append-only record of every library call in an episode."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[CallRecord]:
        return iter(self._records)

    @property
    def records(self) -> list[CallRecord]:
        return list(self._records)

    def open(self, tool: str, kwargs: dict[str, Any], *, stateful: bool) -> CallRecord:
        record = CallRecord(
            index=len(self._records),
            tool=tool,
            kwargs=dict(kwargs),
            started_at=time.time(),
            elapsed_s=0.0,
            ok=False,
            stateful=stateful,
        )
        self._records.append(record)
        return record

    def stateful_calls(self) -> int:
        return sum(1 for record in self._records if record.stateful)

    def env_steps(self) -> int:
        return sum(record.env_steps or 0 for record in self._records)

    def write_jsonl(self, path: str | Path) -> None:
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        with location.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


__all__ = [
    "Backend",
    "CallRecord",
    "EpisodeFinished",
    "Ledger",
    "ToolError",
    "raise_for_error",
    "unwrap",
]
