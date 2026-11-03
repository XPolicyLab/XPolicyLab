"""Owned POSIX process groups, explicit environments and bounded teardown.

Adapted from the upstream daemon lifecycle; ownership extends to descendants.
No repository, model framework or PYTHONPATH is injected implicitly.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


def watch_parent_death(on_death):
    def watch():
        try:
            # Raw descriptor reads avoid holding BufferedReader locks while a
            # daemon watcher remains blocked during normal interpreter shutdown.
            while os.read(sys.stdin.fileno(), 65536):
                pass
        finally:
            on_death()

    threading.Thread(target=watch, daemon=True).start()


def pick_free_port(host="127.0.0.1"):
    """For tests/legacy servers; prefer binding port 0 and announcing it."""
    with socket.socket() as stream:
        stream.bind((host, 0))
        return stream.getsockname()[1]


class ManagedProcess:
    def __init__(self, name, cmd, *, env_overrides=None, log_path=None, cwd=None):
        self.name = name
        self.cmd = list(cmd)
        self.env = dict(os.environ, **(env_overrides or {}))
        self.log_path = Path(log_path) if log_path else None
        self.cwd = cwd
        self._proc = None
        self._log = None
        self._stopped = False

    @property
    def pid(self):
        return self._proc.pid if self._proc else None

    def poll(self):
        return self._proc.poll() if self._proc else None

    def start(self):
        if self._proc is not None:
            raise RuntimeError("Process handle cannot be started twice")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path or os.devnull, "ab")
        try:
            self._proc = subprocess.Popen(
                self.cmd,
                stdin=subprocess.PIPE,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                env=self.env,
                cwd=self.cwd,
                start_new_session=True,
            )
        except BaseException:
            self._log.close()
            self._log = None
            raise

    def _signal(self, value):
        try:
            os.killpg(self.pid, value)
        except ProcessLookupError:
            pass

    def _group_alive(self):
        # Linux may retain orphan zombies until init reaps them. They hold no
        # execution resources; count running members, including grandchildren.
        proc = Path("/proc")
        if proc.is_dir():
            for path in proc.glob("[0-9]*/stat"):
                try:
                    fields = path.read_text().rsplit(")", 1)[1].split()
                    if int(fields[2]) == self.pid and fields[0] != "Z":
                        return True
                except (OSError, ValueError, IndexError):
                    continue
            return False
        try:
            os.killpg(self.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def stop(self, timeout=5.0):
        if timeout <= 0:
            raise ValueError("Teardown timeout must be positive")
        if self._proc is None or self._stopped:
            return
        self._signal(signal.SIGTERM)
        until = time.monotonic() + timeout
        while self._group_alive() and time.monotonic() < until:
            time.sleep(0.02)
        if self._group_alive():
            self._signal(signal.SIGKILL)
        self._proc.wait(timeout=timeout)
        until = time.monotonic() + timeout
        while self._group_alive() and time.monotonic() < until:
            time.sleep(0.02)
        if self._group_alive():
            raise TimeoutError(f"Process group still active: {self.name}")
        if self._proc.stdin:
            self._proc.stdin.close()
        if self._log:
            self._log.close()
            self._log = None
        self._stopped = True
