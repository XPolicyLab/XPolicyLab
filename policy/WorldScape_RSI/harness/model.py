"""Standard XPolicyLab adapter for a frozen development Harness release."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import threading

from XPolicyLab.model_template import ModelTemplate

from . import contracts
from .release import FrozenRelease, ROOT
from .plans import RepeatedAction


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = dict(model_cfg)
        for key, expected in (("bench_name", "RoboDojo"), ("env_cfg_type", "arx_x5"),
                              ("action_type", "joint"), ("policy_name", "WorldScape_RSI")):
            if self.model_cfg.get(key) != expected:
                raise ValueError("Unsupported deployment field: " + key)
        if self.model_cfg.get("eval_batch"):
            raise ValueError("Use one model process per environment; eval_batch=false")
        forbidden = {"routes", "backends", "checkpoint_path", "model", "effort",
                     "wam", "wam_skills", "route_mode", "jev", "jev_checkpoint"}
        if forbidden.intersection(self.model_cfg):
            raise ValueError("Change a release, not per-task model or routing overrides")
        path = Path(self.model_cfg.get("release_path", "harness/release.json"))
        self.release = FrozenRelease(path if path.is_absolute() else ROOT.parent / path,
                                     self.model_cfg.get("release_sha256"))
        if self.release.root != ROOT:
            raise ValueError("Install and launch the frozen adapter beside its release manifest")
        self.dims = contracts.action_dimensions(self.model_cfg["env_cfg_type"])
        self._lock = threading.RLock()
        self._task = self.model_cfg.get("task_name")
        self._backend = None
        self._backend_name = None
        self._queue = deque()
        self._repeat_action = None
        self._repeat_remaining = 0
        self._pending = None
        self._observation = None
        self._planned = self._executed = 0
        self._failed = False
        self._ready = False

    def prepare_case(self, meta):
        with self._lock:
            task = meta.get("task_name")
            case = meta.get("action_case_id")
            if task is None and isinstance(case, str) and case.endswith("_case"):
                task = case[:-5]
            if task is None:
                task = self._task
            if task is None:
                raise ValueError("Public task metadata is required for Skill loading")
            if case is not None and case != task + "_case":
                raise ValueError("Conflicting task and action-case metadata")
            if self._ready and task != self._task:
                raise RuntimeError("End the current trial before changing its task")
            self._task = task
            return {"ok": True}

    def reset(self):
        with self._lock:
            self._ready = False
            self._failed = True
            self._queue.clear()
            self._repeat_action = None
            self._repeat_remaining = 0
            self._pending = self._observation = None
            old, self._backend = self._backend, None
            if old is not None:
                old.close()
            self.release.verify()
            if self._task is None:
                raise ValueError("prepare_case or task_name is required before reset")
            name, spec = "inner_agent", self.release.agent_spec()
            from .backends import build_backend
            backend = build_backend(name, spec, self._task, self.model_cfg)
            try:
                backend.reset()
            except BaseException:
                backend.close()
                raise
            self._backend, self._backend_name = backend, name
            self._planned = self._executed = 0
            self._failed = False
            self._ready = True
            return {"ok": True}

    def _check(self):
        if self._failed or not self._ready:
            raise RuntimeError("Trial is unavailable; a confirmed reset is required")
        self.release.verify()

    def update_obs(self, raw):
        with self._lock:
            try:
                self._check()
                if self._observation is not None and self._pending is None:
                    raise RuntimeError("Duplicate observation without an intervening action")
                clean = contracts.observation(raw, self.dims)
                interrupted = self._backend.observe(clean, self._pending)
                if self._pending is not None:
                    self._executed += 1
                    if interrupted:
                        self._queue.clear()
                        self._repeat_remaining = 0
                    if not self._queue and not self._repeat_remaining:
                        self._backend.completed(self._executed, self._planned, interrupted)
                self._pending = None
                self._observation = clean
            except BaseException:
                self._failed = True
                raise

    def get_action(self):
        with self._lock:
            try:
                self._check()
                if self._observation is None or self._pending is not None:
                    raise RuntimeError("A fresh execution observation is required")
                if not self._queue and not self._repeat_remaining:
                    chunk = self._backend.plan()
                    if isinstance(chunk, RepeatedAction):
                        self._repeat_action = contracts.action(chunk.action, self.dims)
                        self._repeat_remaining = chunk.count
                        self._planned, self._executed = chunk.count, 0
                    else:
                        if not isinstance(chunk, (list, tuple)) or not chunk:
                            raise ValueError("Backend returned no native actions")
                        # Validate the entire proposal before exposing its first action.
                        self._queue.extend(contracts.action(item, self.dims) for item in chunk)
                        self._planned, self._executed = len(self._queue), 0
                if self._repeat_remaining:
                    next_action = contracts.action(self._repeat_action, self.dims)
                    self._repeat_remaining -= 1
                else:
                    next_action = self._queue.popleft()
                self._backend.before(next_action)
                self._pending = next_action
                return [{key: value.copy() for key, value in next_action.items()}]
            except BaseException:
                self._failed = True
                raise

    def update_obs_batch(self, *args, **kwargs):
        raise NotImplementedError("One independent policy process per environment")

    def get_action_batch(self, *args, **kwargs):
        raise NotImplementedError("One independent policy process per environment")

    def release_metadata(self):
        return {"release_sha256": self.release.sha256, "task": self._task,
                "executor": self._backend_name, "selection_scope": "agent_selected_skill",
                "official_single_entry_eligible": False, "eligibility_status": "unverified"}

    def close(self):
        with self._lock:
            self._ready = False
            self._queue.clear()
            self._repeat_action = None
            self._repeat_remaining = 0
            self._pending = self._observation = None
            old, self._backend = self._backend, None
            if old is not None:
                old.close()

    def on_trial_end(self, result):
        self.close()
        return {"ok": True, "session_released": True}
