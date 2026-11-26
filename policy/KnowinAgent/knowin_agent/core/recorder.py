"""Append-only JSONL run records (one file per table) plus saved images.

Tables follow the run-log layout of KRA, our team's in-house ReAct robot-agent
framework: turns, tool_calls, llm_calls, events, delegations, skill_runs.
tool_calls doubles as the execution trace that offline skill consolidation
reads (scope, delegation id, arguments, rationale, outcome).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

TABLES = ("episode", "turns", "tool_calls", "llm_calls", "events", "delegations", "skill_runs", "observations")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 64 else f"<array shape={list(value.shape)}>"
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class RunRecorder:
    def __init__(self, root: Optional[Path], *, save_images: bool = True):
        self.root = Path(root) if root else None
        self.save_images = save_images and self.root is not None
        self._lock = threading.Lock()
        self._seq = 0
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "images").mkdir(exist_ok=True)

    def record(self, table: str, **row) -> dict:
        if table not in TABLES:
            raise ValueError(f"unknown table {table!r}")
        with self._lock:
            self._seq += 1
            row = {"seq": self._seq, "time": round(time.time(), 3), **_jsonable(row)}
            if self.root:
                with open(self.root / f"{table}.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def save_image(self, name: str, rgb: np.ndarray) -> Optional[str]:
        if not self.save_images:
            return None
        from PIL import Image

        path = self.root / "images" / f"{name}.jpg"
        if not path.exists():
            Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(path, quality=85)
        return str(path.relative_to(self.root))

    def write_json(self, name: str, payload: dict) -> None:
        if self.root:
            (self.root / name).write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def read_table(root: Path, table: str) -> list[dict]:
    path = Path(root) / f"{table}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
