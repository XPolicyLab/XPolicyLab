"""Episode isolation and bounded worker startup for the shared policy service."""

import asyncio
import copy
from dataclasses import dataclass
import json
import os
import time
from pathlib import Path

from client_server.ws.protocol.exceptions import ErrorCode, WsError
from XPolicyLab.policy.embodiedevo.model import Model
from XPolicyLab.policy.embodiedevo.runtime.files import write_json


class AgentCapacity:
    """Event-loop-owned reservations, including queued and disconnected agents."""

    def __init__(self, limit):
        self.limit = limit
        self.reserved = {}
        self.total = 0

    def reserve(self, identity, count):
        total = self.total - self.reserved.get(identity, 0) + count
        if total > self.limit:
            raise WsError(ErrorCode.CALL_FAILED, "Agent capacity exhausted; retry prepare_case later",
                          details={"limit": self.limit, "reserved": self.total,
                                   "retryable": True, "retry_after_sec": 1})
        self.total = total
        if count:
            self.reserved[identity] = count
        else:
            self.reserved.pop(identity, None)

    def release(self, identity):
        self.total -= self.reserved.pop(identity, 0)


class SessionModel(Model):
    """Fresh state for one client case, reading one immutable task snapshot.

    The template's Model constructor has validated config and robot dimensions.
    Do not re-register per-case atexit/signal callbacks or re-snapshot live source
    files: that would retain retired models and let concurrent layouts drift.
    Episode workspaces are copied only by the worker after the first observation.
    """

    def __init__(self, template, run_dir, *, startup_timeout):
        self.contexts, self.ready, self.next_episode = {}, {}, 0
        self.robot_dims = copy.deepcopy(template.robot_dims)
        self.frozen_files = dict(template.frozen_files)
        self.cfg = copy.deepcopy(template.cfg)
        self.run = Path(run_dir)
        self.run.mkdir(parents=True, exist_ok=False)
        self.cfg.update(run_dir=str(self.run), inputs_dir=str(template.run / "inputs"),
                        harness_dir=str(template.run / "frozen_harness"), managed_service=True,
                        supervisor_pid=os.getpid(), action_poll_timeout_sec=0,
                        worker_shutdown_grace_sec=10, agent_start_timeout_sec=startup_timeout)
        self.cfg["task_config"]["root"] = str(self.run)
        write_json(self.run / "resolved.json", self.cfg)

    def on_trial_end(self, result=None):
        result = result or {}
        results = result.get("results", [{"env_idx": 0, **result}])
        indices = [item["env_idx"] for item in results]
        if not indices or len(set(indices)) != len(indices) or not set(indices) <= self.contexts.keys():
            raise ValueError("trial_end must identify distinct active environments")
        for index in indices:
            context = self.contexts[index]
            # A client may cancel an episode while its agent is programming or
            # still queued. There is then no outstanding action to acknowledge.
            if not context.pending and not context.finished:
                context.close()
        super().on_trial_end(result)
        for index in indices:
            self.contexts[index].close()
            del self.contexts[index]


@dataclass
class Startup:
    task: asyncio.Task | None = None
    spawning: bool = False
    cancelled: bool = False
    ready: bool = False
    error: str | None = None


class StartupScheduler:
    """Reserve a startup lane until Docker is running, not just until Popen.

    Queued clients receive empty action chunks and can keep their normal poll
    loop/keepalive running. Cancelling queued work never launches a late worker;
    cancelling an in-progress spawn drains it before its owner cleans up.
    """

    def __init__(self, parallelism, timeout, run_model):
        self.semaphore = asyncio.Semaphore(parallelism)
        self.timeout = timeout
        self.run_model = run_model
        self.entries = {}
        self.starting = 0
        self.peak_starting = 0
        self.closed = False

    def ready(self, contexts):
        for context in contexts:
            if context.obs is None or context.step_limit is None:
                raise ValueError("prepare_case and update_obs must precede get_action")
        for context in contexts:
            entry = self.entries.get(context)
            if entry is None:
                if self.closed:
                    raise RuntimeError("Service is shutting down")
                entry = self.entries[context] = Startup()
                entry.task = asyncio.create_task(self._launch(context, entry))
            if entry.error:
                raise RuntimeError(entry.error)
        return all(self.entries[context].ready for context in contexts)

    async def _launch(self, context, entry):
        try:
            async with self.semaphore:
                if entry.cancelled:
                    return
                entry.spawning = True
                self.starting += 1
                self.peak_starting = max(self.peak_starting, self.starting)
                try:
                    await self.run_model(context.start_worker)
                    deadline = time.monotonic() + self.timeout
                    marker = context.run / "episodes" / f"episode_{context.episode:04d}" / "runtime.json"
                    while not entry.cancelled:
                        # Atomic runtime records are tiny; parsing is independent
                        # of the model/API and never waits on the worker pipe.
                        state = json.loads(marker.read_text()) if marker.exists() else {}
                        if state.get("status") in {"running", "finished", "agent_reconnecting"}:
                            entry.ready = True
                            return
                        if state.get("status") == "failed":
                            raise RuntimeError("Agent startup failed; inspect episode runtime.json")
                        if not context.worker.is_alive():
                            raise RuntimeError("Agent worker exited during startup")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Agent startup timed out")
                        await asyncio.sleep(0.05)
                finally:
                    self.starting -= 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            entry.error = f"{type(exc).__name__}: {exc}"

    async def detach(self, contexts):
        entries = []
        for context in list(contexts):
            entry = self.entries.get(context)
            if entry is not None:
                entry.cancelled = True
                if not entry.spawning:
                    entry.task.cancel()
                entries.append((context, entry))
        await asyncio.gather(*(entry.task for _, entry in entries), return_exceptions=True)
        for context, entry in entries:
            if self.entries.get(context) is entry:
                self.entries.pop(context)

    @property
    def queued(self):
        return sum(not e.spawning and not e.cancelled for e in self.entries.values())
