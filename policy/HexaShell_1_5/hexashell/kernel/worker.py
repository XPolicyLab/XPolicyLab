"""The sandboxed process that runs the model's Python cells.

Started by :class:`hexashell.kernel.host.Kernel` with one end of a socket pair.  It imports its
libraries, confines itself (see :mod:`.sandbox`), then serves "exec" requests: each cell runs in
one persistent namespace (variables survive between cells), its output is captured, and every
``robot.*`` call is forwarded to the host and waits for the answer.

Limits: a cell may use CELL_CPU_S seconds of CPU time (time spent waiting for robot calls does
not count); the host also stops cells that sit idle too long.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import linecache
import os
import resource
import signal
import sys
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

from . import wire
from .sandbox import cell_builtins, landlock, seccomp_lockdown

OUTPUT_CAP = 200_000
#: Address space of the worker (numpy arrays included).
MEMORY_CAP = 24 << 30


class CellTimeout(BaseException):
    """CPU time of one cell exhausted."""


class EpisodeStopped(BaseException):
    """The host ended the cell (the episode is over)."""


class _Capture(io.TextIOBase):
    def __init__(self):
        self.parts: list[str] = []
        self.size = 0
        self.dropped = 0

    def write(self, s):
        if not isinstance(s, str):
            s = str(s)
        if self.size < OUTPUT_CAP:
            self.parts.append(s[: OUTPUT_CAP - self.size])
            self.dropped += max(0, len(s) - (OUTPUT_CAP - self.size))
        else:
            self.dropped += len(s)
        self.size += len(s)
        return len(s)

    def text(self) -> str:
        out = "".join(self.parts)
        if self.dropped:
            out += f"\n[... {self.dropped} more characters of output dropped]"
        return out


def _help(obj=None):
    if obj is None:
        print("help(obj): signature and description of a function or object, e.g. help(robot.goto)")
        return
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = None
    if sig is None and not isinstance(obj, type):
        # an object: its public methods, one line each
        print(f"{type(obj).__name__}:")
        for name in dir(type(obj)):
            if name.startswith("_"):
                continue
            member = getattr(type(obj), name)
            doc = (inspect.getdoc(member) or "").split("\n")[0]
            try:
                msig = str(inspect.signature(member)).replace("(self, ", "(").replace("(self)", "()")
            except (TypeError, ValueError):
                msig = ""
            print(f"  .{name}{msig}  {doc}")
        return
    name = getattr(obj, "__name__", type(obj).__name__)
    print(f"{name}{sig or ''}")
    doc = inspect.getdoc(obj)
    if doc:
        print(doc)


class Worker:
    def __init__(self, conn: Connection, workspace: Path, cpu_s: float, note: str = "",
                 proxy: str = "hexashell.robot.proxy"):
        self.conn = conn
        self.note = note
        self.workspace = workspace
        self.cpu_s = cpu_s
        self.cell_index = 0
        self.call_id = 0
        from ..robot.proxy import Blocked, Dropped, NotExecuted, RobotError
        from ..robot.results import NotFound

        import math

        import numpy as np

        robot = importlib.import_module(proxy).Robot(self._rpc)
        self.ns = {"__name__": "__cell__", "__builtins__": cell_builtins(workspace)}
        self.ns["__builtins__"]["help"] = _help
        #: Env_ICL (hexashell.robot.env_icl): the scene facts the robot's results carry, plus the model's
        #: own plan and notes; only a proxy that keeps one has it
        self.env_icl = getattr(robot, "_env_icl", None)
        if self.env_icl is not None:
            self.ns["Env_ICL"] = self.env_icl
        self.ns.update(robot=robot, np=np, math=math, RobotError=RobotError, NotExecuted=NotExecuted,
                       Blocked=Blocked, Dropped=Dropped, NotFound=NotFound)

    # -- robot calls -----------------------------------------------------------------------------
    def _rpc(self, name, args, kwargs):
        from ..robot.proxy import Blocked, Dropped, NotExecuted, RobotError
        from ..robot.results import NotFound

        # robot calls do not count against the cell's computing time, and a timer signal must not
        # fall between a request and its answer
        remaining, _ = signal.setitimer(signal.ITIMER_PROF, 0)
        self.call_id += 1
        call_id = self.call_id
        try:
            wire.send(self.conn, ("call", call_id, name, args, kwargs))
            while True:
                msg = wire.recv_trusted(self.conn)
                kind = msg[0]
                if kind == "stop":
                    raise EpisodeStopped(msg[1])
                if kind not in ("result", "error") or msg[1] != call_id:
                    continue  # an answer to a call that was interrupted earlier
                if kind == "result":
                    return msg[2]
                _, _, etype, text = msg
                if etype == "NotFound":
                    raise NotFound(text)
                if etype == "NotExecuted":
                    raise NotExecuted(text)
                if etype == "Blocked":
                    raise Blocked(text)
                if etype == "Dropped":
                    raise Dropped(text)
                if etype == "ValueError":
                    raise ValueError(text)
                if etype == "TypeError":
                    raise TypeError(text)
                raise RobotError(text)
        finally:
            if remaining > 0:
                signal.setitimer(signal.ITIMER_PROF, remaining)

    # -- cells -----------------------------------------------------------------------------------
    def run_cell(self, code: str) -> tuple[str, str | None, bool]:
        """(output, error text or None, stopped)"""
        self.cell_index += 1
        filename = f"<cell {self.cell_index}>"
        linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
        capture = _Capture()
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = capture
        error = None
        stopped = False

        def on_cpu_limit(signum, frame):
            raise CellTimeout()

        signal.signal(signal.SIGPROF, on_cpu_limit)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.setitimer(signal.ITIMER_PROF, self.cpu_s)
        try:
            tree = ast.parse(code, filename, "exec")
            tail = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                tail = ast.Expression(tree.body.pop().value)
            exec(compile(tree, filename, "exec"), self.ns)
            if tail is not None:
                value = eval(compile(tail, filename, "eval"), self.ns)
                if value is not None:
                    print(repr(value))
        except CellTimeout:
            error = (f"CellTimeout: the cell used more than {self.cpu_s:.0f} s of computing time and was "
                     "stopped (robot calls do not count). Variables defined so far are kept.")
        except EpisodeStopped as exc:
            error = f"EpisodeOver: {exc}"
            stopped = True
        except KeyboardInterrupt:
            error = "Interrupted: the cell was idle too long (sleeping or waiting) and was stopped."
        except SyntaxError as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).rstrip()
        except BaseException as exc:  # noqa: BLE001 - everything the cell raises is reported
            error = _cell_traceback(exc, filename)
        finally:
            signal.setitimer(signal.ITIMER_PROF, 0)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            sys.stdout, sys.stderr = old_out, old_err
        out = capture.text()
        delta = self.env_icl.flush() if self.env_icl is not None else ""
        if delta:
            out = (out.rstrip("\n") + "\n" if out.strip() else "") + delta + "\n"
        return out, error, stopped

    def serve(self) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # the host interrupts cells, never the loop
        wire.send(self.conn, ("ready", os.getpid(), self.note))
        while True:
            try:
                msg = wire.recv_trusted(self.conn)
            except (EOFError, OSError):
                return
            if msg[0] == "exec":
                t0 = time.time()
                out, err, stopped = self.run_cell(msg[1])
                wire.send(self.conn, ("done", out, err, stopped, time.time() - t0))
            elif msg[0] == "quit":
                return


def _cell_traceback(exc: BaseException, filename: str) -> str:
    """The exception with only the cell's own frames (the worker's internals are noise)."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename.startswith("<cell")]
    lines = []
    if frames:
        lines.append("Traceback (cell lines):")
        for f in frames[-6:]:
            src = f"  {f.line.strip()}" if f.line else ""
            lines.append(f"  line {f.lineno}:{src}")
    text = str(exc)
    lines.append(f"{type(exc).__name__}: {text}" if text else type(exc).__name__)
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    fd, workspace, root = int(argv[0]), Path(argv[1]), Path(argv[2])
    cpu_s = float(argv[3]) if len(argv) > 3 else 120.0
    proxy = argv[4] if len(argv) > 4 else "hexashell.robot.proxy"
    conn = Connection(fd)
    workspace.mkdir(parents=True, exist_ok=True)
    os.chdir(workspace)
    # libraries cells may import, loaded before confinement
    import json  # noqa: F401
    import math  # noqa: F401
    import random  # noqa: F401
    import statistics  # noqa: F401

    import numpy  # noqa: F401
    try:
        import cv2  # noqa: F401
    except Exception:  # noqa: BLE001
        pass
    from ..robot import proxy as _proxy, results  # noqa: F401
    importlib.import_module(proxy)  # the benchmark's robot, and the result types it returns

    resource.setrlimit(resource.RLIMIT_FSIZE, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_CAP, MEMORY_CAP))
    resource.setrlimit(resource.RLIMIT_NPROC, (resource.getrlimit(resource.RLIMIT_NPROC)[0],) * 2)
    read_only = sorted({sys.prefix, sys.base_prefix, sys.exec_prefix, str(root / "hexashell"),
                        "/usr", "/lib", "/lib64", "/etc", "/dev/null", "/dev/urandom"})
    note = landlock(read_only, [str(workspace)]) + "; " + seccomp_lockdown()
    Worker(conn, workspace, cpu_s, note, proxy).serve()


if __name__ == "__main__":
    main(sys.argv[1:])
