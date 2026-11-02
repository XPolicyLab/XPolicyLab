"""Code-action sessions: an agent submits a program, the host runs it against the robot and returns a receipt.

One ProgramSession per agent session: its namespace (the program API, the preloaded manipulation, pour and
closed-loop libraries, and whatever earlier programs defined) persists across executions. Each execution runs the
include files and then the program; a stop (stall, failed checkpoint, budget refusal, episode end, deadline)
ends it, and the receipt says where and why.
"""

from __future__ import annotations

import time
from importlib import resources
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from knowin_agent.program.api import ProgramAPI
from knowin_agent.program.sandbox import (LIBRARY_PREFIX, PROGRAM_FILE, ProgramStop, Sandbox, SandboxError, callable_names,
                                  wrap_counted)

# H4/H5 include names whose functions are preloaded; including them is accepted and does nothing.
PRELOADED_INCLUDES = ("tools/high_level.py", "tools/manipulation.py", "tools/pour.py",
                      "primitives/closed_loop/motion.py")
LIBRARY_MODULES = ("manipulation", "pour", "closed_loop")
COUNTED_LIBRARY = frozenset({"pick", "place", "look_down", "screw_down", "press_at", "rim_transfer",
                             "release_retreat", "guarded_move", "handover"})
STOP_REASONS = ("completed", "exception", "checkpoint_failed", "motion_interrupted", "over_budget_refused",
                "episode_over", "timeout")


def library_source(name: str) -> str:
    return resources.files("knowin_agent.program.library").joinpath(f"{name}.py").read_text(encoding="utf-8")


class IncludeRejected(Exception):
    def __init__(self, message: str, name: str):
        super().__init__(message)
        self.name = name


class ProgramSession:
    def __init__(self, runtime, *, workspace: Optional[Union[str, Path]] = None, phase: str = "practice",
                 recorder=None, library=None, deadline_s: float = 1200.0,
                 score_feedback: Optional[Callable[[], object]] = None):
        self.runtime = runtime
        self.workspace = Path(workspace) if workspace is not None else None
        self.phase = phase
        self.recorder = recorder
        self.deadline_s = float(deadline_s)
        self.score_feedback = score_feedback
        self.api = ProgramAPI(runtime, phase=phase, library=library)
        self.sandbox = Sandbox()
        self.api.bind(self.sandbox)
        self.sandbox.define(self.api.namespace())
        self.library_functions: list[str] = []
        for module in LIBRARY_MODULES:
            before = list(self.sandbox.globals)
            self.sandbox.run(self.sandbox.compile(library_source(module), f"{LIBRARY_PREFIX}library/{module}.py"))
            for name in callable_names(self.sandbox.globals, before):
                self.library_functions.append(name)
                if name in COUNTED_LIBRARY:
                    self.sandbox.globals[name] = wrap_counted(name, self.sandbox.globals[name], self._count)
        self.executions = 0
        self.last_receipt: Optional[dict] = None

    def _count(self, name: str) -> None:
        self.api.calls[name] += 1

    # -- includes ---------------------------------------------------------------------------------

    def _resolve(self, include: Sequence[str]) -> tuple[list[tuple[str, str]], list[dict]]:
        if isinstance(include, str):
            include = [include]
        rows, wanted = [], []
        for entry in include or ():
            if not isinstance(entry, str):
                raise IncludeRejected("include must be a list of file names", str(entry))
            if entry.strip() in PRELOADED_INCLUDES:
                rows.append({"name": entry.strip(), "status": "preloaded"})
            else:
                wanted.append(entry)
        if not wanted:
            return [], rows
        if self.workspace is None:
            raise IncludeRejected("this session has no workspace to include files from", wanted[0])
        from knowin_agent.workspace.includes import IncludeError, resolve_includes

        try:
            sources = resolve_includes(self.workspace, wanted)
        except IncludeError as exc:
            raise IncludeRejected(str(exc), wanted[0]) from None
        rows += [{"name": name, "status": "loaded"} for name, _ in sources]
        return sources, rows

    # -- execution ----------------------------------------------------------------------------------

    def execute(self, code: str, include: Sequence[str] = ()) -> dict:
        rt = self.runtime
        self.executions += 1
        started = time.time()
        rt.checkpoints.begin()
        self.api.begin()
        self.sandbox.take_stdout()
        before = rt.backend.budget().ticks_used
        finished_before = {row["track"] for row in rt.executor.lanes(recent=1 << 30)["recently_finished"]}
        error, stop_reason, includes = None, "completed", []
        try:
            sources, includes = self._resolve(include)
            compiled = [(name, self.sandbox.compile(text, name)) for name, text in sources]
            compiled.append((PROGRAM_FILE, self.sandbox.compile(code, PROGRAM_FILE)))
        except IncludeRejected as exc:
            error = {"type": "IncludeError", "message": str(exc), "file": exc.name, "line": None}
            stop_reason, compiled = "exception", []
        except SandboxError as exc:
            error = {"type": "SandboxError", "message": str(exc), "file": exc.filename, "line": exc.lineno}
            stop_reason, compiled = "exception", []
        if compiled:
            self.sandbox.deadline = time.monotonic() + self.deadline_s
            try:
                for _, program in compiled:
                    self.sandbox.run(program)
            except ProgramStop as exc:
                stop_reason, error = exc.reason, self._error(exc)
                self.api.record_stop(exc, error)
            except Exception as exc:  # noqa: BLE001 - agent code errors become the receipt
                stop_reason, error = "exception", self._error(exc)
            finally:
                self.sandbox.deadline = None
            latched = self.api.latched
            if error is None and latched is not None:
                where = self.api.interruption or {}
                stop_reason = latched.reason
                error = {"type": type(latched).__name__, "message": str(latched), "file": where.get("file"),
                         "line": where.get("line"), "program_line": where.get("program_line"),
                         "caught_by_program": True}
        unfinished = self.api.settle(stopped=error is not None)
        if rt.executor.episode_over and stop_reason == "completed":
            stop_reason = "episode_over"
        receipt = self._receipt(stop_reason, error, includes, before, finished_before, unfinished, started)
        self.last_receipt = receipt
        if self.recorder is not None:
            self.recorder.record("tool_calls", scope="task", tool="exec",
                                 arguments={"code": code, "include": list(include or ())},
                                 result={k: v for k, v in receipt.items() if k != "stdout"}, ok=receipt["ok"])
        return receipt

    def _error(self, exc: BaseException) -> dict:
        where = self.sandbox.error_location(exc)
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return {"type": type(exc).__name__, "message": str(message)[:2000], **where}

    def _receipt(self, stop_reason, error, includes, before, finished_before, unfinished, started) -> dict:
        rt, api = self.runtime, self.api
        stdout, length = self.sandbox.take_stdout()
        after = rt.backend.budget().ticks_used
        finished = [row for row in rt.executor.lanes(recent=1 << 30)["recently_finished"]
                    if row["track"] not in finished_before]
        raw_steps = sum(1 for row in finished if row["label"].startswith("step"))
        tracks = [row for row in finished if not row["label"].startswith("step")]
        receipt = {
            "ok": error is None,
            "stop_reason": stop_reason,
            "error": error,
            "interruption": api.interruption,
            "stdout": stdout,
            "stdout_chars": length,
            "native_steps": {"before": before, "after": after, "used": api.native_used(), "resets": api.resets},
            "tracks": tracks,
            "raw_steps": raw_steps,
            "calls": dict(sorted(api.calls.items())),
            "checkpoints": [dict(row) for row in rt.checkpoints.log],
            "includes": includes,
            "budget_panel": api.budget_panel(),
            "execution": self.executions,
            "seconds": round(time.time() - started, 2),
        }
        if unfinished:
            receipt["unwaited_tracks"] = unfinished
        if self.phase == "practice":
            receipt["score_feedback"] = self.score_feedback() if self.score_feedback is not None else None
            if rt.executor.episode_over and not self._succeeded():
                receipt["attempt_over"] = (
                    f"The evaluator ended this attempt without success ({rt.executor.over_reason}). Practice "
                    "goes on: write into lessons/ what led to it (the scoring section names the conditions), "
                    "then submit a program that calls reset() to start the scene again. Until then every robot "
                    "call fails.")
        return receipt

    def _succeeded(self) -> bool:
        try:
            return bool(self.runtime.backend.evaluation().get("success"))
        except Exception:  # noqa: BLE001
            return False

    # -- images for the agent -------------------------------------------------------------------------

    def images(self, limit: int = 4) -> list:
        """Views taken by observe() during the last program (the agent context already shows the current
        head and wrist images); the latest `limit`."""
        return list(self.api.images)[-limit:]
