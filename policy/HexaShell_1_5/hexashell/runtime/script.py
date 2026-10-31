"""Scripted episodes: cells read from a file instead of asked of the model.

``HEXASHELL_SCRIPT=path.py`` replaces the model with the file's cells (split at lines that start
with ``# %%``), one per turn, then finishes with status ``keep_partial``.  Everything else - the
sandbox, the robot, the records - is the same, so a script can collect pictures for offline
benchmarks or check the tools in the real simulator without spending model calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .llm import Reply


def cells_of(text: str) -> list[str]:
    cells, current = [], []
    for line in text.splitlines():
        if line.startswith("# %%"):
            if any(s.strip() for s in current):
                cells.append("\n".join(current).strip("\n"))
            current = []
        else:
            current.append(line)
    if any(s.strip() for s in current):
        cells.append("\n".join(current).strip("\n"))
    return cells


class ScriptGateway:
    def __init__(self, path: str, *, log=None):
        self.path = Path(path).expanduser()
        self.cells = cells_of(self.path.read_text())
        self.turn = 0
        self.log = log or (lambda event: None)

    def settings(self) -> dict[str, Any]:
        return {"model": f"script:{self.path.name}", "cells": len(self.cells)}

    def ask(self, messages, tools, *, deadline=None, stop=None) -> Reply:
        i = self.turn
        self.turn += 1
        if i < len(self.cells):
            call = {"id": f"script_{i + 1}", "type": "function",
                    "function": {"name": "python", "arguments": json.dumps({"code": self.cells[i]})}}
        else:
            call = {"id": f"script_{i + 1}", "type": "function",
                    "function": {"name": "finish", "arguments": json.dumps(
                        {"status": "keep_partial", "summary": f"script {self.path.name} done"})}}
        return Reply(text="", calls=[call], finish_reason="tool_calls", usage={})

    def close(self) -> None:
        pass
