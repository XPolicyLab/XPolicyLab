"""VLABench agent for the OEH conditions: the agent the pinned Evaluator calls, with the OEH runtime in this process.

``oeh_eval.vlabench_client.run`` (started by ``oeh_policy.vlabench_runner``) hooks OpenWAM's pinned single_eval.py so
the agent VLABench's ``Evaluator`` is given is a ``VLABenchEndpoint`` around the released client object single_eval.py
built from the same policy_config.yml as the direct condition (``OpenWAMVLABenchPolicy``): OpenWAM's request and the
conversion of its answer stay the released client's (``oeh_policy.vlabench.OpenWAMBackend``). The official loop owns
the scene, inverse kinematics, stepping, termination and every result file; the evaluator-side runner owns the
attempt's sidecar, the recorded observations and the native-episode record. This endpoint owns only the permitted
policy input, one admitted native row per ``predict`` and OEH's evidence at the native boundary: the evaluator's hook
on ``env.step`` acknowledges each row (``stepped``) with the state measured after the step.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import threading
import time

from .contracts import encoded, plain
from .model import Model
from .runtime import Runtime
from .vlabench import OpenWAMBackend, VLABenchContract

_VALIDATED = {}


def policy_config_path() -> Path:
    """The campaign's frozen config (OEH_EVAL_INTERNAL_CONFIG.policy_config), else OEH_POLICY_CONFIG."""
    evaluation = json.loads(os.environ.get("OEH_EVAL_INTERNAL_CONFIG") or "{}")
    path = evaluation.get("policy_config") or os.environ.get("OEH_POLICY_CONFIG")
    if not path:
        raise ValueError("OEH_POLICY_CONFIG must name a frozen VLABench configuration")
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


class VLABenchEndpoint(Model):
    """VLABench's agent duck type (``name``, ``control_mode``, ``reset``, ``predict``, ``close``) with the OEH runtime
    behind it. ``episode`` is the evaluator's view of the case (``track``, ``task_name``, ``episode``) and its
    ``finger_qpos()``, the two finger joints it measured with the observation being predicted on."""

    def __init__(self, official, episode):
        self.contract = VLABenchContract()
        self.cfg = _load_config(self.contract, policy_config_path())
        native = self.cfg["vlabench"]
        for key in ("track", "task_name", "episode"):
            if getattr(episode, key) != native[key]:
                raise ValueError(f"the VLABench case differs from the frozen configuration: {key}")
        self.episode = episode
        self.attempt_id = os.environ.get("VLABENCH_RUN_ID")
        if self.attempt_id and (not re.fullmatch(r"[A-Za-z0-9_.-]+", self.attempt_id)
                                or self.attempt_id in {".", ".."}):
            raise ValueError("invalid attempt identity")
        self.events_path = Path(os.environ["OEH_EVAL_EVENTS"]).resolve()
        if not self.events_path.is_file():
            raise RuntimeError("the VLABench runner owns the attempt sidecar; start through oeh_policy.vlabench_runner")
        client_path = Path(native["official_client_path"]).resolve()
        if self.cfg["identity_files"].get(str(client_path)) is None:
            raise ValueError("actual VLABench OpenWAM client must be pinned")
        if Path(inspect.getfile(type(official))).resolve() != client_path:
            raise ValueError("the released VLABench client object does not come from the pinned "
                             "openwam2vlabench_interface.py")
        self.teacher_start = None
        teacher = (self.cfg.get("review") or {}).get("teacher") or {}
        if teacher.get("api") == "codex_app_server":
            from .codex_teacher import prestart
            from .teacher import prompt_text
            # As oeh_policy.model.Model does: the app-server's start overlaps the scene build, and the first reset
            # waits for it before its call bound starts. The prompt is the one make_client would give this runtime.
            self.teacher_start = prestart(teacher, prompt_text(self.cfg["review"].get("schedule"),
                                                               self.contract.benchmark))
        self.backend = OpenWAMBackend(official)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="oeh-vlabench-runtime", daemon=True)
        self.thread.start()
        self.runtime = None
        self.used = False
        self.pending = None  # (ticket, row) of the row returned by the last predict, until its step is acknowledged
        self.finished = False
        self.closed = False

    # ---- VLABench's agent duck type (evaluation/evaluator/base.py: name, control_mode, reset, predict) ----
    @property
    def name(self) -> str:
        return self.backend.official.name

    @property
    def control_mode(self) -> str:
        return "ee"

    def reset(self):
        # The pinned Evaluator resets its agent once before each episode (base.py:78); single_eval.py runs one.
        if self.used:
            self._fatal("reset after this policy endpoint's one episode")
            raise RuntimeError("one VLABench episode per policy endpoint; start a fresh attempt for the next")
        self.used = True
        if self.teacher_start is not None:
            # Outside the call bound, as RoboTwin's endpoint does: a slow app-server start would otherwise spend the
            # first call's call_seconds.
            concurrent.futures.wait([self.teacher_start])
        self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id,
                               contract=self.contract, teacher_start=self.teacher_start)
        self.teacher_start = None
        try:
            self._call(self.runtime.start())  # ends with the released client's reset (RESET to the server)
        except BaseException as exc:
            self.finish("preflight_failed")
            self._fatal(f"policy runtime failed to start: {type(exc).__name__}: {exc}")
            raise

    def predict(self, observation, **kwargs):
        """One native row for the Evaluator's ``predict`` (base.py:147): rows of an admitted chunk are served in
        order across calls; each is admitted against this observation's projection just before it is returned."""
        if self.runtime is None or self.finished:
            raise RuntimeError("predict outside the VLABench episode")
        if self.pending is not None:
            raise RuntimeError("the previous VLABench row was never acknowledged by a native step")
        try:
            self.update_obs({"native": observation, "finger_qpos": self.episode.finger_qpos()})
            action = self.contract.validate_action(self._call(self.runtime.next_action()), self.cfg["action_limits"])
            ticket = self.before_action(action)
            self.append("native_action_submitted", {"trace_id": ticket["trace_id"],
                                                    "sha256": ticket["action_sha256"], "action": action})
            self.pending = (ticket, action)
        except BaseException:
            self.finish("runner_failed")
            raise
        # The released client's return shape (target_pos, target_euler, gripper_state), float64 copies of the row.
        return action[:3].copy(), action[3:6].copy(), action[6:8].copy()

    def close(self):
        """single_eval.py's ``policy.close()`` (its finally): the episode's runtime first, then the client socket."""
        try:
            if self.runtime is not None and not self.finished:
                self.finish("episode_closed")
        finally:
            if not self.closed:
                self.closed = True
                self.backend.official.close()

    # ---- the evaluator's native boundary (oeh_eval.vlabench_client) ----
    def stepped(self, state, *, terminal: bool, success: bool, native_steps: int, step_limit: int):
        """The native step of the last row completed: acknowledge it with the measured state; end on terminal."""
        if self.pending is None or self.finished:
            raise RuntimeError("a native VLABench step without an admitted row")
        ticket, action = self.pending
        self.pending = None
        try:
            self.after_action({"ticket": ticket, "action": action, "state": self.contract.project_state(state),
                               "terminal": bool(terminal)})
            if terminal:
                self.append("native_terminal_observed", {"native_steps": native_steps,
                                                         "native_step_limit": step_limit, "success": bool(success)})
                self.finish("native_terminal" if success else "native_step_limit")
        except BaseException:
            self.finish("runner_failed")
            raise

    def abandon(self, reason: str):
        """The episode ended by an exception the official loop swallows (base.py:90-92); close with its reason."""
        self.finish(reason)

    def append(self, kind, payload):
        row = {"kind": kind, "payload": plain(payload), "mono_ns": time.monotonic_ns()}
        with self.events_path.open("ab") as stream:
            stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, reason):
        """Close the OEH episode once and move its events to the attempt sidecar."""
        if self.finished:
            return
        self.finished = True
        try:
            self.close_episode({"reason": reason})
        finally:
            if self.runtime is not None:
                for row in self.drain_events():
                    self.append(row["kind"], row["payload"])


def make_policy(official, episode):
    """``oeh_eval.vlabench_client.run``'s agent factory for the OEH conditions."""
    return VLABenchEndpoint(official, episode)
