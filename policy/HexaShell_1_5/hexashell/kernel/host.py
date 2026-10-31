"""Host side of the cell sandbox: starts the worker process, runs cells in it, answers its calls.

The worker is a fresh interpreter (``python -I``: no environment variables, no user site) with a
minimal environment - in particular no API key - that confines itself before it runs any cell.
Robot calls arrive as ("call", id, name, args, kwargs) and are answered by ``dispatch``; the
episode ending during a call is answered with "stop", which ends the cell.

Watchdogs, besides the worker's own CPU limit: a cell that neither calls the robot nor finishes
for ``idle_s`` (sleeping, waiting on something) is interrupted, and so is one running longer than
``total_s`` in all.  A worker that does not come back from an interrupt is killed and replaced;
its variables are lost and the result says so.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

from . import wire

ROOT = Path(__file__).resolve().parents[2]
BOOT = ("import sys; sys.path.insert(0, sys.argv[3]); "
        "from hexashell.kernel.worker import main; main(sys.argv[1:])")
#: The module whose ``Robot`` the cells get, unless the benchmark names another (RoboDojo's).
DEFAULT_PROXY = "hexashell.robot.proxy"
#: Seconds the worker gets to stop after an interrupt before it is killed.
GRACE_S = 10.0


@dataclass
class CellResult:
    output: str
    error: str | None
    stopped: bool = False  # the episode ended during the cell
    restarted: bool = False  # the worker was replaced: the cell's variables are gone
    elapsed: float = 0.0
    calls: list[dict] = field(default_factory=list)


class CallFailed(Exception):
    """Raised by ``dispatch`` for a call the cell should see as an error of the given type."""

    def __init__(self, kind: str, text: str):
        super().__init__(text)
        self.kind = kind
        self.text = text


class Stop(Exception):
    """Raised by ``dispatch`` when the episode is over: the cell is ended."""


Dispatch = Callable[[str, tuple, dict], Any]


class Kernel:
    def __init__(self, workspace: Path, dispatch: Dispatch, *, cpu_s: float = 120.0, idle_s: float = 150.0,
                 total_s: float = 1800.0, python: str | None = None, proxy: str = DEFAULT_PROXY,
                 log: Callable[[dict], None] | None = None):
        self.workspace = Path(workspace).resolve()
        self.dispatch = dispatch
        self.cpu_s = cpu_s
        self.idle_s = idle_s
        self.total_s = total_s
        self.python = python or sys.executable
        if not (proxy.startswith("hexashell.") and all(part.isidentifier() for part in proxy.split("."))):
            raise ValueError(f"the robot proxy must be a module of the hexashell package, not {proxy!r}")
        self.proxy = proxy
        self.log = log or (lambda event: None)
        self.proc: subprocess.Popen | None = None
        self.conn: Connection | None = None
        self.sandbox_note = ""
        self.starts = 0
        self.stopped = False

    # -- process -----------------------------------------------------------------------------------
    def start(self) -> None:
        self.close()
        self.workspace.mkdir(parents=True, exist_ok=True)
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.workspace),
            "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "TMPDIR": str(self.workspace),
        }
        self.proc = subprocess.Popen(
            [self.python, "-I", "-B", "-c", BOOT, str(theirs.fileno()), str(self.workspace), str(ROOT),
             str(self.cpu_s), self.proxy],
            pass_fds=(theirs.fileno(),), env=env, cwd=str(self.workspace), stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=open(self.workspace.parent / "worker.err", "ab"),
            start_new_session=True)
        theirs.close()
        self.conn = Connection(ours.detach())
        self.starts += 1
        deadline = time.monotonic() + 120.0
        while not self.conn.poll(1.0):
            if self.proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"the cell worker did not start (exit {self.proc.poll()}); "
                                   f"see {self.workspace.parent / 'worker.err'}")
        msg = wire.recv_untrusted(self.conn)
        if msg[0] != "ready":
            raise RuntimeError(f"the cell worker sent {msg[0]!r} instead of 'ready'")
        self.sandbox_note = str(msg[2]) if len(msg) > 2 else ""
        self.log({"kind": "worker_start", "pid": msg[1], "start": self.starts, "sandbox": self.sandbox_note})

    def close(self) -> None:
        if self.conn is not None:
            try:
                wire.send(self.conn, ("quit",))
            except OSError:
                pass
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.wait(2.0)
            except subprocess.TimeoutExpired:
                self._kill()
        if self.conn is not None:
            self.conn.close()
        self.proc = None
        self.conn = None

    def _kill(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(5.0)
        except subprocess.TimeoutExpired:
            pass

    # -- cells -------------------------------------------------------------------------------------
    def run(self, code: str) -> CellResult:
        restarted = False
        if self.proc is None or self.proc.poll() is not None:
            restarted = self.starts > 0
            self.start()
        t0 = time.monotonic()
        wire.send(self.conn, ("exec", code))
        calls: list[dict] = []
        last = t0
        interrupted_at = None
        why = ""
        stop_reason = None
        while True:
            now = time.monotonic()
            if interrupted_at is None:
                if now - last > self.idle_s:
                    why = (f"the cell was stopped after {self.idle_s:.0f} s without finishing or calling the "
                           "robot (sleeping or waiting?). Variables defined so far are kept.")
                elif now - t0 > self.total_s:
                    why = (f"the cell was stopped after running {self.total_s / 60:.0f} minutes in all. "
                           "Variables defined so far are kept.")
                if why:
                    self.log({"kind": "cell_interrupt", "why": why})
                    os.kill(self.proc.pid, signal.SIGINT)
                    interrupted_at = now
            elif now - interrupted_at > GRACE_S:
                self._kill()
                self.close()
                self.log({"kind": "worker_killed", "why": why})
                return CellResult("", f"Interrupted: {why.split('. ')[0]}; the Python process did not stop "
                                  "and was restarted: all variables are lost.", stopped=stop_reason is not None,
                                  restarted=True, elapsed=now - t0, calls=calls)
            if not self.conn.poll(1.0):
                if self.proc.poll() is not None:
                    return self._died(t0, calls, restarted, stop_reason)
                continue
            try:
                msg = wire.recv_untrusted(self.conn)
            except (EOFError, OSError):
                return self._died(t0, calls, restarted, stop_reason)
            except Exception as exc:  # noqa: BLE001 - a message the host refuses to read
                self._kill()
                self.close()
                return CellResult("", f"SandboxError: the cell sent something the host does not accept ({exc}); "
                                  "the Python process was restarted: all variables are lost.",
                                  restarted=True, elapsed=time.monotonic() - t0, calls=calls)
            kind = msg[0] if isinstance(msg, tuple) and msg else None
            if kind == "done":
                _, out, err, stopped, elapsed = msg
                if why and err and err.startswith("Interrupted"):
                    err = f"Interrupted: {why}"
                return CellResult(str(out), None if err is None else str(err),
                                  stopped=bool(stopped) or stop_reason is not None, restarted=restarted,
                                  elapsed=time.monotonic() - t0, calls=calls)
            if kind != "call" or len(msg) != 5:
                continue
            _, call_id, name, args, kwargs = msg
            if stop_reason is not None:
                wire.send(self.conn, ("stop", stop_reason))
                continue
            t_call = time.monotonic()
            record = {"name": str(name), "t": round(t_call - t0, 3), "args": brief(args, kwargs)}
            try:
                value = self.dispatch(str(name), tuple(args), dict(kwargs))
                wire.send(self.conn, ("result", call_id, value))
            except Stop as exc:
                stop_reason = str(exc) or "the episode is over"
                record["stop"] = stop_reason
                self.stopped = True
                wire.send(self.conn, ("stop", stop_reason))
            except CallFailed as exc:
                record["error"] = f"{exc.kind}: {exc.text}"[:300]
                wire.send(self.conn, ("error", call_id, exc.kind, exc.text))
            except Exception as exc:  # noqa: BLE001 - a bug on our side must not end the episode
                text = f"robot.{name} failed inside the robot software: {type(exc).__name__}: {exc}"
                record["error"] = text[:300]
                record["traceback"] = traceback.format_exc()[-4000:]
                wire.send(self.conn, ("error", call_id, "RobotError", text))
            record["s"] = round(time.monotonic() - t_call, 3)
            calls.append(record)
            last = time.monotonic()

    def _died(self, t0, calls, restarted, stop_reason) -> CellResult:
        code = self.proc.poll() if self.proc is not None else None
        self.close()
        self.log({"kind": "worker_died", "exit": code})
        hint = " (out of memory?)" if code in (-9, 137) else ""
        return CellResult("", f"the Python process running the cells stopped unexpectedly (exit {code}){hint}; "
                          "it will be restarted for the next cell: all variables are lost.",
                          stopped=stop_reason is not None, restarted=True, elapsed=time.monotonic() - t0,
                          calls=calls)


def brief(args, kwargs, limit: int = 300) -> str:
    """A short, readable account of a call's arguments for the record (arrays by shape)."""

    def one(v, depth=0):
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            return f"<array {tuple(v.shape)} {v.dtype}>"
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, (list, tuple)) and depth < 2:
            inner = ", ".join(one(x, depth + 1) for x in list(v)[:8]) + (", ..." if len(v) > 8 else "")
            return f"[{inner}]" if isinstance(v, list) else f"({inner})"
        if isinstance(v, dict) and depth < 2:
            return "{" + ", ".join(f"{k}: {one(x, depth + 1)}" for k, x in list(v.items())[:8]) + "}"
        text = repr(v)
        return text if len(text) <= 80 else text[:77] + "..."

    try:
        parts = [one(a) for a in args] + [f"{k}={one(v)}" for k, v in kwargs.items() if v is not None]
        text = ", ".join(parts)
    except Exception:  # noqa: BLE001 - only for the record
        text = "?"
    return text if len(text) <= limit else text[: limit - 3] + "..."
