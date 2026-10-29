"""A persistent Python kernel for model-written programs, in its own process.

The host sends cells over stdin and reads one JSON reply per cell from a
private descriptor, so anything a program prints cannot corrupt the protocol.
The kernel namespace comes from a bootstrap callable `module:function` that
receives JSON options and returns a session object with:

    namespace: dict       names visible to programs
    drain() -> dict       attachments and events gathered during the cell
    close() -> None
"""

import argparse
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

from .events import EpisodeEnded, Escalation, ProgramFinished


MAX_OUTPUT_CHARS = 8000
_SOURCE_ROOT = Path(__file__).resolve().parents[2]  # .../src
# Top-level packages of the source checkout that kernels import (skills, benches).
REPO_PACKAGES = ("skills", "benches")


class PlainSession:
    """A bootstrap with an empty namespace, for tests and debugging."""

    def __init__(self, options=None):
        self.namespace = {"__name__": "__program__"}

    def drain(self):
        """Nothing: a plain session gathers no attachments or events."""
        return {}

    def close(self):
        """Nothing to release."""
        pass


class KernelDied(RuntimeError):
    """The kernel process failed to start, is not running, or did not answer in time (it is
    then killed)."""
    pass


_SANDBOX = None


def sandbox_available() -> bool:
    """Whether this machine can run the kernel's sandbox (util-linux unshare with user,
    mount and PID namespaces)."""
    global _SANDBOX
    if _SANDBOX is None:
        try:
            result = subprocess.run(["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork",
                                     "--mount-proc", "true"], capture_output=True, timeout=30)
            _SANDBOX = result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _SANDBOX = False
    return _SANDBOX


class Kernel:
    """Host-side handle. One kernel lives for one episode."""

    def __init__(self, bootstrap: str, options: dict, *, python: str = sys.executable,
                 cwd: str | None = None, env: dict | None = None, log_path: Path | None = None,
                 start_timeout: float = 300.0):
        self._log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        if env is None:
            env = dict(os.environ)
            roots = [str(_SOURCE_ROOT), str(_SOURCE_ROOT.parent)]
            env["PYTHONPATH"] = os.pathsep.join(roots + [p for p in [env.get("PYTHONPATH")] if p])
        isolation = options.get("isolation")
        if isolation == "auto":
            # Containers often forbid user namespaces; an evaluation host need not allow them.
            # The guard inside the kernel (reachable services, no processes) needs none.
            isolation = True if sandbox_available() else {"sandbox": False}
            if isolation is not True:
                print("kernel: user namespaces are not available here; programs run without the filesystem "
                      "sandbox (the network and process guard stays)", file=sys.stderr, flush=True)
            options = {**options, "isolation": isolation}
        command = [python, "-u", "-m", "harness_policy.runtime.kernel",
                   "--bootstrap", bootstrap, "--options", json.dumps(options)]
        if isolation and not (isinstance(isolation, dict) and isolation.get("sandbox") is False):
            # App. A: the kernel sees only the Python installation and the harness
            # source, and reaches only the services in its options (isolation.py).
            from .isolation import hidden_roots, kernel_environment, package_dirs, sandbox_command

            prefix = Path(python).resolve().parents[1]
            source = [_SOURCE_ROOT, *(_SOURCE_ROOT.parent / name for name in REPO_PACKAGES), *package_dirs()]
            extra = isolation.get("hide", []) if isinstance(isolation, dict) else []
            command = sandbox_command(command, keep=[], read_only=[prefix, *source],
                                      hide=hidden_roots([prefix, _SOURCE_ROOT, *( [cwd] if cwd else [])], extra))
            env, cwd = kernel_environment(options, env), None
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            cwd=cwd, env=env, bufsize=0,
        )
        self._replies: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self.cells = 0
        ready = self._receive(start_timeout)
        if ready.get("status") != "ready":
            self.close()
            raise KernelDied(f"Kernel failed to start: {ready.get('error')}")
        self.names = ready.get("names", [])

    def _read(self):
        for line in self.process.stdout:
            try:
                self._replies.put(json.loads(line))
            except json.JSONDecodeError:
                self._replies.put({"status": "protocol_error", "error": line[:200].decode(errors="replace")})
        self._replies.put(None)

    def _receive(self, timeout):
        try:
            reply = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            raise KernelDied(f"Kernel did not answer within {timeout:.0f} s; it was killed") from None
        if reply is None:
            raise KernelDied(f"Kernel exited with code {self.process.wait()}")
        return reply

    def run(self, code: str, timeout: float) -> dict:
        """Run `code` as the next cell and return the kernel's reply: status (ok, error,
        escalated, episode_ended or finished), clipped stdout, error, seconds, and what the
        session drained (attachments, events, state, usage). Raises KernelDied when the
        kernel is not running, exits, or does not reply within `timeout` seconds."""
        if self.process.poll() is not None:
            raise KernelDied("Kernel is not running")
        self.cells += 1
        message = json.dumps({"op": "run", "cell": self.cells, "code": code}) + "\n"
        self.process.stdin.write(message.encode())
        self.process.stdin.flush()
        return self._receive(timeout)

    def kill(self):
        """Kill the kernel process if it still runs."""
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def close(self):
        """Ask the kernel to close its session and exit (killing it when it does not within
        30 s), then close the pipes and the log."""
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'{"op": "close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.kill()
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()
        self._reader.join(timeout=5)
        if self._log is not subprocess.DEVNULL:
            self._log.close()


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    half = MAX_OUTPUT_CHARS // 2
    return text[:half] + f"\n... [{len(text) - MAX_OUTPUT_CHARS} characters omitted] ...\n" + text[-half:]


def format_error(error: BaseException) -> str:
    """Traceback restricted to program frames, plus the frame that raised."""
    summary = traceback.TracebackException.from_exception(error)
    frames = [frame for frame in summary.stack if frame.filename.startswith("<cell-")]
    if summary.stack and (not frames or frames[-1] is not summary.stack[-1]):
        frames.append(summary.stack[-1])
    lines = ["Traceback (program frames):\n"]
    for frame in frames:
        lines.append(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n')
        if frame.line:
            lines.append(f"    {frame.line}\n")
    lines.extend(summary.format_exception_only())
    return "".join(lines)


def restore_api(namespace: dict, api: dict) -> list[str]:
    """Put back the session's API names that a program rebound or deleted (a model's
    own `def pick(m): ...` would otherwise break every later call of the skill) and
    return them, sorted."""
    missing = object()
    restored = []
    for name, value in api.items():
        current = namespace.get(name, missing)
        if current is value:
            continue
        try:  # a constant set to the same value again (TABLE_Z = 0.765) is not rebound
            same = not callable(value) and bool(current == value)
        except Exception:
            same = False
        if not same:
            namespace[name] = value
            restored.append(name)
    return sorted(restored)


def _serve(bootstrap: str, options: dict) -> None:
    channel = os.fdopen(os.dup(1), "w", buffering=1)
    # Stray writes to descriptor 1 from programs or libraries go to the log instead.
    os.dup2(2, 1)
    sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", buffering=0), write_through=True)

    def reply(data):
        channel.write(json.dumps(data, allow_nan=False, default=repr) + "\n")

    try:
        module_name, _, function_name = bootstrap.partition(":")
        session = getattr(importlib.import_module(module_name), function_name)(options)
        namespace = session.namespace
    except BaseException as error:
        reply({"status": "error", "error": format_error(error) if not isinstance(error, SystemExit)
               else "bootstrap exited"})
        return
    if options.get("isolation"):
        from .isolation import endpoints, install_network_guard

        install_network_guard(endpoints(options))
    reply({"status": "ready", "names": sorted(name for name in namespace if not name.startswith("_"))})
    # The names the session provides, restored after every cell that rebinds them.
    api = {name: value for name, value in namespace.items() if not name.startswith("__")}
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "close":
            break
        started = time.perf_counter()
        buffer = io.StringIO()
        result = {"status": "ok", "error": None, "escalation": None, "episode_ended": None}
        try:
            code = compile(request["code"], f"<cell-{request['cell']}>", "exec")
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(code, namespace)
        except Escalation as escalation:
            result["status"] = "escalated"
            result["escalation"] = escalation.trace.to_dict()
        except EpisodeEnded as ended:
            result["status"] = "episode_ended"
            result["episode_ended"] = ended.trace.to_dict()
        except ProgramFinished as finished:
            result["status"] = "finished"
            result["finish"] = finished.summary
        except KeyboardInterrupt:
            result["status"] = "error"
            result["error"] = "KeyboardInterrupt"
        except BaseException as error:
            result["status"] = "error"
            result["error"] = format_error(error)
        restored = restore_api(namespace, api)
        if restored:
            buffer.write(f"\nWarning: the program rebound the API name(s) {', '.join(restored)}; they were restored "
                         "to the built-in skills after it ended. Give your own functions and variables other names.\n")
        result["stdout"] = _clip(buffer.getvalue())
        result["seconds"] = time.perf_counter() - started
        try:
            result.update(session.drain())
        except Exception as error:
            result["drain_error"] = format_error(error)
        reply(result)
    session.close()


def main(argv=None):
    """The kernel process: build the session from `--bootstrap` with the JSON
    `--options`, then run cells read from stdin until a close request or end of input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", required=True)
    parser.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    _serve(args.bootstrap, json.loads(args.options))


if __name__ == "__main__":
    main()
