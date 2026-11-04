"""What an episode leaves on disk.

``<result_dir>/<task>_s<seed>_l<layout>-<time>/`` (other benchmarks name the part before the time):

* ``result.json``      verdict, costs and settings (written at the end, rewritten if it changes)
* ``prompt.md``        the system prompt and the task card as sent
* ``transcript.json``  the conversation, pictures replaced by their files
* ``cells.jsonl``      one line per cell: code, output, error, time, robot calls, steps
* ``cells/NNN.py``     the code of each cell, for reading
* ``events.jsonl``     requests to the model, motions, pictures dropped, nudges ...
* ``states.jsonl``     the arms after every action (joints, link6 pose, gripper)
* ``pictures/``        every picture the model was shown (JPEG)
* ``frames/``          the three cameras after each cell that moved the robot (JPEG)
* ``workspace/``       the cell's working directory
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..vision.overlay import jpeg


def _clean(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "x"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class Records:
    def __init__(self, root: Path, info: dict[str, Any], name: str | None = None):
        stamp = time.strftime("%Y%m%dT%H%M%S")
        if name:
            name = f"{_clean(name)}-{stamp}"
        else:
            task = info.get("task_name") or "task"
            seed = info.get("eval_seed")
            layout = info.get("layout")
            name = _clean(task)
            if seed is not None:
                name += f"_s{seed}"
            if layout is not None:
                name += f"_l{layout}"
            name += f"-{stamp}"
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        n = 1
        while path.exists():
            n += 1
            path = root / f"{name}-{n}"
        self.dir = path
        for sub in ("cells", "pictures", "frames", "workspace"):
            (path / sub).mkdir(parents=True)
        self.workspace = path / "workspace"
        self._lock = threading.Lock()
        self._files: dict[str, Any] = {}
        self.t0 = time.time()

    def _append(self, name: str, record: dict[str, Any]) -> None:
        line = json.dumps(_plain(record), ensure_ascii=False)
        with self._lock:
            f = self._files.get(name)
            if f is None:
                f = self._files[name] = open(self.dir / name, "a", encoding="utf-8")
            f.write(line + "\n")
            f.flush()

    def event(self, record: dict[str, Any]) -> None:
        self._append("events.jsonl", {"t": round(time.time() - self.t0, 3), **record})

    def state(self, record: dict[str, Any]) -> None:
        self._append("states.jsonl", record)

    def cell(self, number: int, code: str, record: dict[str, Any]) -> None:
        (self.dir / "cells" / f"{number:03d}.py").write_text(code, encoding="utf-8")
        self._append("cells.jsonl", {"cell": number, "t": round(time.time() - self.t0, 3), "code": code, **record})

    def picture(self, number: int, kind: str, data: bytes) -> str:
        name = f"pictures/{number:03d}_{_clean(kind)}.jpg"
        (self.dir / name).write_bytes(data)
        return name

    def frames(self, label: str, images: dict[str, np.ndarray]) -> None:
        for cam, image in images.items():
            (self.dir / "frames" / f"{_clean(label)}_{cam}.jpg").write_bytes(jpeg(image, 85))

    def write_json(self, name: str, data: Any) -> None:
        tmp = self.dir / (name + ".tmp")
        tmp.write_text(json.dumps(_plain(data), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.dir / name)

    def write_text(self, name: str, text: str) -> None:
        (self.dir / name).write_text(text, encoding="utf-8")

    def close(self) -> None:
        with self._lock:
            for f in self._files.values():
                f.close()
            self._files.clear()
