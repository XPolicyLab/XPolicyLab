from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess

from .common import ROOT, event, stop_process


class Backend:
    """Private, serialized transport to the simulator's own Python environment."""

    def __init__(self, cfg, run, command=None, log_name="simulator.log"):
        self.cfg, self.run, self.command = cfg, run, command
        self.reset_restarts = 0
        self._spawn(log_name)

    def _spawn(self, log_name):
        self.timeout = self.cfg["request_seconds"]
        self.log = (self.run / log_name).open("w")
        cmd = self.command or [self.cfg["benchmark_python"], str(ROOT / "policylab/worker.py"), str(self.run / "config.json")]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.pop("PYTHONPATH", None)
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                                            env=env, start_new_session=True)
        except OSError:
            self.log.close()
            raise
        self.buffer = b""
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def request(self, _retry_reset=True, **request):
        timeout = (self.cfg.get("options", {}).get("reset_request_seconds", self.timeout)
                   if request.get("op") == "reset" else self.timeout)
        if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
            raise ValueError("Simulator request timeout must be finite and positive")
        self.process.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
        self.process.stdin.flush()
        import time
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            if not self.selector.select(max(0, deadline - time.monotonic())):
                raise TimeoutError(f"Simulator timed out; see {self.log.name}")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f"Simulator exited; see {self.log.name}")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        reply = json.loads(line)
        if "invalid_action" in reply:
            from .api import InvalidAction
            raise InvalidAction(reply["invalid_action"])
        if "error" in reply:
            if request.get("op") == "reset" and _retry_reset:
                previous_log = self.log.name
                self.close()
                self.reset_restarts += 1
                log_name = f"simulator-reset-retry-{self.reset_restarts:04d}.log"
                event(self.run / "backend_events.jsonl", phase="reset_retry",
                      previous_log=previous_log, log=log_name)
                self._spawn(log_name)
                self.request(op="initialize")
                return self.request(_retry_reset=False, **request)
            raise RuntimeError(f"Simulator failed; see {self.log.name}")
        return reply

    def close(self):
        if self.process.poll() is None:
            try:
                self.timeout = 10
                self.request(op="close")
            except Exception:
                pass
        stop_process(self.process)
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()
