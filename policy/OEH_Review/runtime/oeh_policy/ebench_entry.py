"""OEH's driver for OpenWAM's released EBench bridge loop.

The released loop (``benchmarks/ebench/openwam2ebench_interface.py`` ``run_worker``: reset, one ``act`` per
observation, ``client.step``, until GenManip ends the episode) runs unchanged; only its ``EBenchOpenWAMDriver`` is
this one (``oeh_eval.ebench_client`` installs it, ``oeh_policy.ebench_runner`` starts the client). The evaluator's
client hooks own the attempt's sidecar, the recorded observations and the native-episode record, and call this driver
around each native submission (``submitting`` before the action reaches the server, ``submitted`` after its response),
as RoboTwin's eval callback surrounds ``take_action``. This driver owns only the permitted policy input, one admitted
action per submission and the OEH evidence of the native boundary.

GenManip returns no observation with the response that ends the episode (``isaac_worker_pool.py`` ``step`` answers a
finished worker with its pending reset, which resolves to ``obs: None``), so the last receipt binds the last observed
state.
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

from .contracts import digest, encoded, plain
from .ebench import EBenchContract, OpenWAMBackend
from .model import Model
from .runtime import Runtime

_VALIDATED = {}


def policy_config_path() -> Path:
    """The campaign's frozen config (OEH_EVAL_INTERNAL_CONFIG.policy_config), else OEH_POLICY_CONFIG."""
    evaluation = json.loads(os.environ.get("OEH_EVAL_INTERNAL_CONFIG") or "{}")
    path = evaluation.get("policy_config") or os.environ.get("OEH_POLICY_CONFIG")
    if not path:
        raise ValueError("OEH_POLICY_CONFIG must name a frozen EBench configuration")
    return Path(path).resolve()


def accept_validated(path: Path, cfg) -> None:
    """The runner validated this exact file in this process; do not hash the checkpoint again."""
    _VALIDATED.update(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      cfg=cfg)


def _load_config(contract, path: Path):
    if _VALIDATED.get("path") == str(path) and \
            _VALIDATED.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
        contract.trust(_VALIDATED["cfg"])
        return _VALIDATED["cfg"]
    return contract.validate_config(json.loads(path.read_text(encoding="utf-8")))


class EBenchEndpoint(Model):
    """One EBench episode's OEH runtime behind the released driver's interface (``act``, ``invalidate_episode``,
    ``steps``, ``episodes``)."""

    def __init__(self, official, south, *, send_state):
        self.contract = EBenchContract()
        self.cfg = _load_config(self.contract, policy_config_path())
        native = self.cfg["ebench"]
        if send_state is not True:
            raise ValueError("the released EBench checkpoint is proprio-conditioned: the bridge must send state")
        self.attempt_id = os.environ.get("EBENCH_RUN_ID")
        if self.attempt_id and (not re.fullmatch(r"[A-Za-z0-9_.-]+", self.attempt_id)
                                or self.attempt_id in {".", ".."}):
            raise ValueError("invalid attempt identity")
        self.events_path = Path(os.environ["OEH_EVAL_EVENTS"]).resolve()
        if not self.events_path.is_file():
            raise RuntimeError("the EBench client owns the attempt sidecar; start through oeh_policy.ebench_runner")
        bridge = Path(native["official_bridge_path"]).resolve()
        if self.cfg["identity_files"].get(str(bridge)) is None:
            raise ValueError("the released EBench bridge must be pinned")
        self.teacher_start = None
        teacher = self.cfg["review"]["teacher"]
        if teacher.get("api") == "codex_app_server":
            from .codex_teacher import prestart
            from .teacher import prompt_text
            # As RoboTwin's endpoint: the app-server starts while the server assigns the episode, and the first
            # observation waits for it before its call bound starts.
            self.teacher_start = prestart(teacher, prompt_text(self.cfg["review"].get("schedule"),
                                                               self.contract.benchmark))
        self.backend = OpenWAMBackend(official(south, send_state=True))
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="oeh-ebench-runtime", daemon=True)
        self.thread.start()
        self.runtime = None
        self.used = False
        self.closed = False
        self.offered = None  # the action act() returned, until the client submits it
        self.pending = None  # (ticket, action) between submission and response
        self.observed_state = None
        self.native_calls = 0
        self.steps = 0  # read by the released loop's last log line
        self.episodes = 0

    def start(self):
        if self.used:
            raise RuntimeError("one EBench episode per policy driver")
        self.used = True
        self.episodes = 1
        if self.teacher_start is not None:
            concurrent.futures.wait([self.teacher_start])
        self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id,
                               contract=self.contract, teacher_start=self.teacher_start)
        self.teacher_start = None
        try:
            self._call(self.runtime.start())
        except BaseException:
            self.close_episode({"reason": "preflight_failed"})
            raise

    def act(self, inner_obs):
        """One native observation in, one OEH-authorized native action out (submitted by the released loop)."""
        if self.closed:
            raise RuntimeError("the OEH episode has closed")
        if self.offered is not None or self.pending is not None:
            raise RuntimeError("the previous action was never submitted and acknowledged")
        try:
            if self.runtime is None:
                self.start()
            if self.native_calls == 0:
                self.append("native_reset", {"episode_id": inner_obs.get("episode_id"),
                                             "native_step": inner_obs.get("timestep")})
            self.native_calls += 1
            self.update_obs({"native": inner_obs})
            self.observed_state = self.contract.project_state(inner_obs)
            action = self.contract.validate_action(self._call(self.runtime.next_action()), self.cfg["action_limits"])
            self.offered = action
            self.steps += 1
            return action
        except BaseException:
            self.failed("runner_failed")
            raise

    def submitting(self, action):
        """The client is about to send ``action``: authorize it (exactly the action act returned)."""
        if self.closed or self.offered is None:
            raise RuntimeError("no OEH-authorized action is waiting for submission")
        try:
            if digest(action) != digest(self.offered):
                raise ValueError("the submitted action differs from the one OEH returned")
            ticket = self.before_action(self.offered)
            self.append("native_action_submitted", {"trace_id": ticket["trace_id"],
                                                    "sha256": ticket["action_sha256"], "action": self.offered})
        except BaseException:
            self.failed("runner_failed")
            raise
        self.pending, self.offered = (ticket, self.offered), None

    def submitted(self, inner_obs, terminal):
        """GenManip answered the submission: acknowledge it with the state it reports (the last observed state when
        the response ends the episode and carries none)."""
        if self.pending is None:
            raise RuntimeError("no OEH submission is waiting for its response")
        (ticket, action), self.pending = self.pending, None
        try:
            state = self.contract.project_state(inner_obs) if inner_obs is not None else self.observed_state
            self.after_action({"ticket": ticket, "action": action, "state": state, "terminal": bool(terminal)})
            if terminal:
                self.append("native_terminal_observed", {"native_steps": self.steps,
                                                         "native_step_limit": self.cfg["native_step_limit"]})
                self.finish("native_terminal")
        except BaseException:
            self.failed("runner_failed")
            raise

    def invalidate_episode(self):
        raise RuntimeError("one EBench episode per attempt: the released reconnect would start another")

    def append(self, kind, payload):
        row = {"kind": kind, "payload": plain(payload), "mono_ns": time.monotonic_ns()}
        with self.events_path.open("ab") as stream:
            stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())

    def failed(self, reason):
        if not self.closed:
            try:
                self.finish(reason)
            except Exception:  # the original failure is the one to report
                pass

    def finish(self, reason):
        if self.closed:
            return
        self.closed = True
        try:
            self.close_episode({"reason": reason})
        finally:
            if self.runtime is not None:
                for row in self.drain_events():
                    self.append(row["kind"], row["payload"])


def driver_class(official, episode):
    """The class the released loop instantiates as ``EBenchOpenWAMDriver(south, send_state=...)``: OEH's driver on
    the released one, registered with the evaluator's episode for its submission hooks."""

    class OEHDriver(EBenchEndpoint):
        def __init__(self, south, *, send_state=True):
            super().__init__(official, south, send_state=send_state)
            episode.listener = self

    return OEHDriver
