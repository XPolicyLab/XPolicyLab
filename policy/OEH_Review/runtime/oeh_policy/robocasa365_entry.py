"""The OEH review policy inside the pinned official RoboCasa365 loop (``benchmarks/robocasa365/single_eval.py``).

The official loop owns the environment, the seed (``env.reset(seed=seed + trial)``), the step limit (RoboCasa's own
task horizon), success and its printed result; the evaluator's runner (``oeh_eval.robocasa365_client``) owns the
attempt's sidecar, the recorded observations and the native-episode record. This endpoint stands where the released
``OpenWAMRoboCasa365Policy`` stands in that loop (``reset``, ``act(obs, prompt)``, ``close``): the evaluator's
``_build_policy`` hook builds the released client exactly as the direct condition does, then hands it here. It owns
only the permitted policy input, one admitted native action12 per ``act`` and the OEH evidence at the actual native
step, which the runner's environment hook reports back (``after_native_step``).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

from .contracts import encoded, plain
from .model import Model
from .robocasa365 import OpenWAMBackend, RoboCasa365Contract, sliced_action
from .runtime import Runtime

_VALIDATED = {}
# The official loop's settings the frozen configuration must agree with (its merged policy_config.yml).
RUN_KEYS = ("task", "split", "seed")


def policy_config_path() -> Path:
    """The campaign's frozen config (OEH_EVAL_INTERNAL_CONFIG.policy_config), else OEH_POLICY_CONFIG."""
    evaluation = json.loads(os.environ.get("OEH_EVAL_INTERNAL_CONFIG") or "{}")
    path = evaluation.get("policy_config") or os.environ.get("OEH_POLICY_CONFIG")
    if not path:
        raise ValueError("OEH_POLICY_CONFIG must name a frozen RoboCasa365 configuration")
    return Path(path).resolve()


def accept_validated(path: Path, cfg) -> None:
    """The runner validated this exact file in this process; do not hash the checkpoint again."""
    _VALIDATED.update(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      cfg=cfg)


def _load_config(contract, path: Path):
    if _VALIDATED.get("path") == str(path) and \
            _VALIDATED.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
        contract.trust(_VALIDATED["cfg"])  # binds the mode; the files were checked moments ago
        return _VALIDATED["cfg"]
    return contract.validate_config(json.loads(path.read_text(encoding="utf-8")))


class RoboCasa365Endpoint(Model):
    """One RoboCasa365 episode under OEH review mode, in the official loop's policy seat.

    ``official`` is the released ``OpenWAMRoboCasa365Policy`` the loop's own ``_build_policy`` built (connected and
    pinged), ``interface`` the module it came from and ``run_cfg`` the merged config the loop runs with."""

    def __init__(self, official, interface, run_cfg):
        self.contract = RoboCasa365Contract()
        self.cfg = _load_config(self.contract, policy_config_path())
        native = self.cfg["robocasa365"]
        for key in RUN_KEYS:
            if str(run_cfg.get(key)) != str(native[key]):
                raise ValueError(f"the official RoboCasa365 loop differs from the frozen configuration: {key} "
                                 f"{run_cfg.get(key)!r}, frozen {native[key]!r}")
        self.attempt_id = os.environ.get("ROBOCASA365_RUN_ID")
        if self.attempt_id and (not re.fullmatch(r"[A-Za-z0-9_.-]+", self.attempt_id)
                                or self.attempt_id in {".", ".."}):
            raise ValueError("invalid attempt identity")
        events = os.environ.get("OEH_EVAL_EVENTS")
        self.events_path = Path(events).resolve() if events else None
        if self.events_path is None or not self.events_path.is_file():
            raise RuntimeError("the RoboCasa365 runner owns the attempt sidecar; start through "
                               "oeh_policy.robocasa365_runner")
        client_path = Path(native["official_client_path"]).resolve()
        if (Path(getattr(interface, "__file__", "") or "").resolve() != client_path
                or self.cfg["identity_files"].get(str(client_path)) is None):
            raise ValueError("the released RoboCasa365 client the official loop imported must be the pinned one")
        self.teacher_start = None
        teacher = (self.cfg.get("review") or {}).get("teacher") or {}
        if teacher.get("api") == "codex_app_server":
            from .codex_teacher import prestart
            from .teacher import prompt_text
            # As oeh_policy.model.Model does: the app-server's start (its bounded retries and preflight) overlaps the
            # kitchen's construction, and the reset waits for it before its call bound starts. The prompt is the one
            # make_client would give this runtime (RoboCasa365's prompt set).
            self.teacher_start = prestart(teacher, prompt_text(self.cfg["review"].get("schedule"),
                                                               self.contract.benchmark))
        self.backend = OpenWAMBackend(official, interface)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="oeh-robocasa365-runtime", daemon=True)
        self.thread.start()
        self.runtime = None
        self.used = False
        self.acts = 0
        self.pending = None  # (ticket, action) of the submitted action its native step has not acknowledged yet

    def get_action(self):
        # One native action per official act() call; rows of an admitted chunk follow in order.
        return [self._call(self.runtime.next_action())]

    def reset(self):
        """The official loop's ``policy.reset()`` before the episode's first step: the OEH runtime starts here (its
        start ends with the released client's reset, the server's RESET, as the direct condition's reset)."""
        if self.used:
            # num_trials is 1: a second reset would start an episode this endpoint does not serve.
            self._fatal("reset after this policy endpoint's one episode")
            raise RuntimeError("one RoboCasa365 episode per policy endpoint (num_trials 1)")
        self.used = True
        if self.teacher_start is not None:
            # Outside the call bound: nothing times the official loop's reset, and a slow app-server start would
            # otherwise spend the first call's call_seconds (RoboDojo's Model waits before reporting ready).
            concurrent.futures.wait([self.teacher_start])
        self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id,
                               contract=self.contract, teacher_start=self.teacher_start)
        self.teacher_start = None
        try:
            self._call(self.runtime.start())
        except BaseException as exc:
            try:
                self.finish("preflight_failed")
            finally:
                # As oeh_policy.model.Model: an endpoint that can serve no episode says so (the campaign counts a
                # start that never took a native step as the harness's fault, never the arm's).
                self._fatal(f"policy runtime failed to start: {type(exc).__name__}: {exc}")
            raise

    def act(self, obs, prompt):
        """The official loop's ``policy.act(obs, instruction)``: one admitted native action12, as the action dict the
        released client's ``slice_action`` hands ``env.step``."""
        if self.runtime is None:
            raise RuntimeError("the official loop must reset the policy before its first act")
        if self.pending is not None:
            raise RuntimeError("the previous RoboCasa365 action was never acknowledged by a native step")
        try:
            if self.acts == 0:
                native = self.cfg["robocasa365"]
                self.append("native_reset", {"task": native["task"], "seed": native["seed"], "native_step": 0})
            self.acts += 1
            self.update_obs({"native": obs, "instruction": prompt})
            action = self.contract.validate_action(self.get_action()[0], self.cfg["action_limits"])
            ticket = self.before_action(action)
            self.append("native_action_submitted", {"trace_id": ticket["trace_id"], "sha256": ticket["action_sha256"],
                                                    "action": action})
            self.pending = (ticket, action)
            # Exactly one submission: the official loop hands this dict to env.step, which the runner's hook reports.
            return sliced_action(self.backend.interface, action)
        except BaseException:
            if not self.loop.is_closed():
                self.finish("runner_failed")
            raise

    def after_native_step(self, obs, *, terminal, reason, native_steps, native_step_limit):
        """The runner's environment hook, after ``env.step`` ran the submitted action: acknowledge it with the state the
        step left (``terminal`` when the official loop stops after this step: success, done or its step limit)."""
        if self.pending is None:
            raise RuntimeError("a native RoboCasa365 step ran without an admitted OEH action")
        ticket, action = self.pending
        self.pending = None
        try:
            state = self.contract.project_state(obs)
            self.after_action({"ticket": ticket, "action": action, "state": state, "terminal": bool(terminal)})
            if terminal:
                self.append("native_terminal_observed", {"native_steps": native_steps,
                                                         "native_step_limit": native_step_limit, "reason": reason})
                self.finish(reason)
        except BaseException:
            if not self.loop.is_closed():
                self.finish("runner_failed")
            raise

    def close(self):
        """The official loop's ``policy.close()`` (run_eval's finally): end the runtime if it is still open, then close
        the released client's connection, as the direct condition's close does."""
        try:
            if not self.loop.is_closed():
                self.finish("episode_closed")
        finally:
            self.backend.policy.close()

    def append(self, kind, payload):
        row = {"kind": kind, "payload": plain(payload), "mono_ns": time.monotonic_ns()}
        with self.events_path.open("ab") as stream:
            stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, reason):
        """Close the runtime and copy its remaining events into the attempt's sidecar."""
        try:
            self.close_episode({"reason": reason})
        finally:
            if self.runtime is not None:
                for row in self.drain_events():
                    self.append(row["kind"], row["payload"])


def make_policy(official, interface, run_cfg):
    """The policy factory the evaluator's ``_build_policy`` hook calls for the OEH condition. An endpoint that cannot be
    built can serve no episode and says so (``policy-fatal.json``), as a runtime that cannot start does: the official
    loop has taken no native step, so the campaign counts it as the harness's start-up fault, never the arm's."""
    try:
        return RoboCasa365Endpoint(official, interface, run_cfg)
    except Exception as exc:
        Model._fatal(f"policy endpoint failed to start: {type(exc).__name__}: {exc}")
        raise
