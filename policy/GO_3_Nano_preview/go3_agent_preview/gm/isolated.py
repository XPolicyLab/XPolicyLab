"""Per-environment code processes: a deadline kills computation, not a thread.

Each worker owns its generator and latest observation. No simulator state crosses
the boundary. A dead worker is never reused within an episode.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from multiprocessing.connection import wait
import time
import os
import traceback
from collections.abc import Mapping


def _worker(connection, cfg, key, episode):
    from .model import GmModel
    model = GmModel(cfg)
    model._episode[key] = episode - 1
    connection.send(("ready", None))
    try:
        while True:
            command, value = connection.recv()
            if command == "close":
                break
            if command != "step":
                raise ValueError(command)
            started = time.monotonic()
            try:
                model.update_obs(value)
                chunk = model.get_action_batch([key])[0]
                status = model.status(key)
                status["compute_ms"] = (time.monotonic() - started) * 1000
                connection.send(("ok", (chunk, status)))
            except BaseException:
                connection.send(("error", traceback.format_exc()))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        model.close()
        connection.close()


class IsolatedGm:
    def __init__(self, cfg):
        self.cfg = cfg
        transport = float(os.environ.get("XPL_REQUEST_TIMEOUT_S", "120"))
        self.timeout_s = max(0.01, min(float(cfg.get("geact_request_timeout_s", 600)), transport - 10.0))
        # Worker startup imports the adapter in a fresh spawned process. Cold
        # page cache, a slow shared filesystem or a larger batch can exceed the
        # 30 s default; raise it with code.worker_startup_timeout_s, capped by
        # the enclosing RPC budget so the parent never waits past its deadline.
        code_cfg = cfg.get("code") if isinstance(cfg.get("code"), Mapping) else {}
        startup = float(code_cfg.get("worker_startup_timeout_s", 30.0))
        if not math.isfinite(startup) or startup <= 0:
            raise ValueError(
                f"code.worker_startup_timeout_s must be a finite positive number, got {startup!r}"
            )
        self.startup_timeout_s = min(startup, self.timeout_s)
        self.context = mp.get_context("spawn")
        self.workers = {}
        self.observations = {}
        self.statuses = {}
        self.episodes = {}
        self.latest = []

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, observations):
        self.latest = [int(obs.get("env_idx", i)) for i, obs in enumerate(observations)]
        for key, obs in zip(self.latest, observations):
            self.observations[key] = obs
        # Startup is a separate bounded observation RPC, before large payloads
        # are sent. Launch every worker first; wait only to one common deadline.
        pending = {}
        startup_started = time.monotonic()
        deadline = startup_started + self.startup_timeout_s
        for key in self.latest:
            if key in self.workers or self.statuses.get(key, {}).get("failure"):
                continue
            parent, child = self.context.Pipe()
            episode = self.episodes.get(key, -1) + 1
            self.episodes[key] = episode
            process = self.context.Process(target=_worker, args=(child, self.cfg, key, episode))
            process.start()
            child.close()
            self.workers[key] = (process, parent)
            pending[parent] = key
        while pending:
            ready = wait(list(pending), timeout=max(0, deadline - time.monotonic()))
            if not ready:
                break
            for connection in ready:
                key = pending.pop(connection)
                try:
                    kind, _ = connection.recv()
                    if kind != "ready":
                        raise RuntimeError("worker did not acknowledge startup")
                except (EOFError, OSError, RuntimeError) as exc:
                    self._fail(key, "code_worker_startup_error", str(exc),
                               startup_ms=(time.monotonic() - startup_started) * 1000)
        for key in pending.values():
            self._fail(key, "code_worker_startup_timeout", "startup deadline exceeded",
                       startup_ms=(time.monotonic() - startup_started) * 1000)

    def _fail(self, key, reason, detail, *, compute_ms=0.0, startup_ms=None):
        previous = self.statuses.get(key, {})
        self.drop_env(key)
        self.statuses[key] = {**previous, "done": True, "failure": reason,
                              "worker_error": detail, "events": [],
                              "compute_ms": float(compute_ms),
                              "compute_wait_ms": float(compute_ms),
                              "compute_measurement": "failed_parent_monotonic_wait" if startup_ms is None else "no_code_compute_startup_failed"}
        if startup_ms is not None:
            self.statuses[key]["startup_elapsed_ms"] = float(startup_ms)

    def get_action_batch(self, indices=None):
        indices = self.latest if indices is None else [int(i) for i in indices]
        pending, chunks = {}, {}
        requested_at = {}
        deadline = time.monotonic() + self.timeout_s
        for key in indices:
            if key not in self.observations:
                raise RuntimeError(f"no observation for env {key}")
            if key not in self.workers:
                previous = self.statuses.get(key, {})
                self.statuses[key] = {**previous, "compute_ms": 0.0, "compute_wait_ms": 0.0,
                                      "compute_measurement": "no_code_compute_controlled_hold"}
                chunks[key] = [self._hold(key)]
                continue
            process, connection = self.workers[key]
            requested_at[key] = time.monotonic()
            try:
                connection.send(("step", self.observations[key]))
                pending[connection] = key
            except (EOFError, OSError) as exc:
                self._fail(key, "code_worker_error", str(exc),
                           compute_ms=(time.monotonic() - requested_at[key]) * 1000)
                chunks[key] = [self._hold(key)]
        while pending:
            ready = wait(list(pending), timeout=max(0, deadline - time.monotonic()))
            if not ready:
                break
            for connection in ready:
                key = pending.pop(connection)
                try:
                    kind, value = connection.recv()
                    if kind == "ok":
                        chunks[key], self.statuses[key] = value
                        self.statuses[key]["compute_wait_ms"] = (time.monotonic() - requested_at[key]) * 1000
                        self.statuses[key]["compute_measurement"] = "successful_worker_monotonic_compute"
                    else:
                        self._fail(key, "code_worker_error", value,
                                   compute_ms=(time.monotonic() - requested_at[key]) * 1000)
                        chunks[key] = [self._hold(key)]
                except (EOFError, OSError) as exc:
                    self._fail(key, "code_worker_error", str(exc),
                               compute_ms=(time.monotonic() - requested_at[key]) * 1000)
                    chunks[key] = [self._hold(key)]
        for key in pending.values():
            self._fail(key, "code_compute_timeout", f"exceeded {self.timeout_s:g} s",
                       compute_ms=(time.monotonic() - requested_at[key]) * 1000)
            chunks[key] = [self._hold(key)]
        return [chunks[key] for key in indices]

    def _hold(self, key):
        state = self.observations[key]["state"]
        return {k: state[k].copy() if hasattr(state[k], "copy") else state[k]
                for k in ("left_arm_joint_state", "left_ee_joint_state",
                          "right_arm_joint_state", "right_ee_joint_state")}

    def status(self, key):
        return self.statuses.get(key, {"step": 0, "done": False, "failure": None, "events": []})

    def drop_env(self, key):
        self.statuses.pop(key, None)
        worker = self.workers.pop(key, None)
        if worker is not None:
            process, connection = worker
            # Terminate instead of queueing close behind potentially stuck code.
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.2)
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
            connection.close()
            process.close()

    def reset(self):
        for key in list(self.workers):
            self.drop_env(key)
        self.observations.clear()
        self.statuses.clear()
        self.latest.clear()

    close = reset
