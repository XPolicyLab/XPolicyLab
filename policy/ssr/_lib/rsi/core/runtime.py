from __future__ import annotations

import threading
import time
import traceback
from typing import Iterator

from . import actions, obs as obs_mod
from ..programs import select_program

CALL_BUDGET_S = 45.0
_torch_import: dict = {}


def import_torch() -> dict | None:
    if _torch_import:
        return None
    t0 = time.perf_counter()
    try:
        import torch

        _torch_import["seconds"] = round(time.perf_counter() - t0, 3)
    except Exception as e:
        _torch_import["error"] = f"{type(e).__name__}: {e}"[:300]
    return dict(_torch_import)


class Context:
    def __init__(self, first: obs_mod.Observation):
        self.first = first
        self.obs = first
        self.instruction = first.instruction
        self.memory: dict = {}


class EpisodeAgent:
    def __init__(self):
        self.ctx: Context | None = None
        self.gen: Iterator[list[dict]] | None = None
        self.steps = 0
        self.failed = False
        self.buffer: list[dict] = []
        self.background = False
        self._job = None
        self._last: dict | None = None
        self._held = 0
        self.call_budget_s = CALL_BUDGET_S

    def observe(self, raw: dict) -> None:
        o = obs_mod.parse(raw, self.steps)
        self.steps += 1
        if self.ctx is None:
            self.ctx = Context(o)
            program = select_program(o.instruction, o)
            self.background = bool(getattr(program, "background", False))
            self.call_budget_s = float(getattr(program, "call_budget_s", None) or CALL_BUDGET_S)
            if self.background:
                import_torch()
            self.gen = program.run(self.ctx)
        else:
            self.ctx.obs = o

    def prefetch(self) -> None:
        if self.background and self._job is None and (not self.buffer) and (self.gen is not None) and (not self.failed):
            self._start()

    def act(self, deadline: float | None = None) -> list[dict]:
        if self.ctx is None:
            return []
        if self.buffer:
            chunk, self.buffer = self.buffer, []
            return self._sent(chunk)
        if self.gen is not None and (not self.failed):
            kind, val = self._advance(deadline)
            if kind == "busy":
                self._held += 1
                return self._sent([self._last] if self._last is not None else actions.hold(self.ctx.obs, n=1))
            if self._held:
                self._held = 0
            if kind == "chunk":
                if val:
                    return self._sent(list(val))
            elif kind == "stop":
                self.gen = None
            else:
                self.failed = True
        return self._sent(actions.hold(self.ctx.obs, n=5))

    def _sent(self, chunk: list[dict]) -> list[dict]:
        if chunk:
            self._last = chunk[-1]
        return chunk

    def _step(self):
        try:
            return "chunk", next(self.gen)
        except StopIteration:
            return "stop", None
        except Exception:
            return "error", traceback.format_exc()[-4000:]

    def _start(self) -> None:
        done, box = threading.Event(), {}

        def work():
            try:
                box["r"] = self._step()
            finally:
                done.set()

        self._job = done, box
        threading.Thread(target=work, name="rsi-program", daemon=True).start()

    def _advance(self, deadline: float | None):
        if not (self.background and deadline is not None):
            if self._job is not None:
                self._job[0].wait()
                box, self._job = self._job[1], None
                return box["r"]
            return self._step()
        if self._job is None:
            self._start()
        done, box = self._job
        if not done.wait(max(0.0, deadline - time.monotonic())):
            return "busy", None
        self._job = None
        return box["r"]

    def push_back(self, rest: list[dict]) -> None:
        if rest:
            self.buffer = list(rest) + self.buffer
