# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Subprocess RPC servers: spawn (parent side), death-watch (child side)."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from typing import Callable

from roborsi.utils.config import get_repo_root, get_rlinf_repo_path
from roborsi.utils.logging import get_logger

logger = get_logger("daemon")


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def watch_parent_death(on_death: Callable[[], None]) -> None:
    """Call ``on_death()`` once, from a background thread, when stdin hits EOF.

    Under :class:`ProcessDaemon` this fires exactly when the parent dies. If
    invoked from a terminal, ``read()`` blocks on user input and never fires;
    if stdin is redirected from ``/dev/null`` or already closed, it fires
    immediately. Read the file descriptor directly so interpreter shutdown
    cannot race a daemon thread holding the buffered stdin lock.
    """

    def _watch() -> None:
        try:
            while os.read(sys.stdin.fileno(), 4096):
                pass
        except (OSError, ValueError):
            pass
        on_death()

    threading.Thread(target=_watch, daemon=True).start()


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def pick_free_port(host: str = "127.0.0.1") -> int:
    """Return an unused TCP port on ``host``.

    There's a small race between the socket closing and the child binding;
    on 127.0.0.1 it's near-zero in practice.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


class ProcessDaemon:
    """Wraps a subprocess server with ready-detection and lifecycle."""

    def __init__(
        self,
        name: str,
        cmd: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
        log_path: str | None = None,
        cwd: str | None = None,
        prepend_rlinf_path: bool = True,
        inherit_environment: bool = True,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.subprocess_env = os.environ.copy() if inherit_environment else {}
        self.subprocess_env.update(env_overrides or {})
        if prepend_rlinf_path:
            # Native environment workers need the RLinf checkout first.
            rlinf_path = str(get_rlinf_repo_path())
            pythonpath = self.subprocess_env.get("PYTHONPATH", "")
            self.subprocess_env["PYTHONPATH"] = (
                f"{rlinf_path}{os.pathsep}{pythonpath}" if pythonpath else rlinf_path
            )
        self.log_path = log_path
        self.cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._log_f = None

    def poll(self) -> int | None:
        """Return the subprocess exit code, or None if it's still running.

        None also before ``start()``. A non-None value means the child has
        exited — used by ``wait_for_ready`` to fail fast on early crashes.
        """
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> None:
        """Spawn the subprocess. Returns immediately; does not wait for readiness."""
        self._log_f = (
            open(self.log_path, "a") if self.log_path else open(os.devnull, "w")
        )
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                # stdin pipe lets the child detect our death via EOF; see
                # watch_parent_death above.
                stdin=subprocess.PIPE,
                stdout=self._log_f,
                stderr=subprocess.STDOUT,
                env=self.subprocess_env,
                cwd=self.cwd or get_repo_root(),
            )
        except Exception:
            self._log_f.close()
            self._log_f = None
            raise
        logger.info("%s spawned (pid=%s)", self.name, self._proc.pid)

    def stop(self, timeout: float = 15.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._log_f is not None:
            try:
                self._log_f.close()
            except Exception:
                pass
            self._log_f = None
