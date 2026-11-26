"""RoboCasa GR1 OEH policy for the pinned official ``single_eval.py`` loop (the review condition).

The official loop owns the simulator, the episode's seed, the step loop, success and its ``[RESULT]`` lines; the
evaluator's client (``oeh_eval.robocasa_gr1_client``) owns the attempt's sidecar, the recorded observations and the
native-episode record. Where the loop constructs OpenWAM's released ``OpenWAMRoboCasaGR1Policy``, the client
constructs this endpoint instead, handing it the released class and the loop's own arguments: the endpoint builds the
released client from them, runs the OEH review runtime in this process and answers the loop's ``reset``, ``act`` and
``close`` as the released client does. Each ``act`` returns the released client's own conversion of one admitted EEF33
row (``robocasa_gr1.OpenWAMBackend.convert``); the evaluator reports the native step that action ran
(``native_stepped``), the acknowledgement OEH's gate records, and the endpoint never steps the simulator itself.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from .contracts import encoded, plain
from .model import Model
from .robocasa_gr1 import CLIENT_KEYS, OpenWAMBackend, RoboCasaGR1Contract
from .runtime import Runtime

_VALIDATED = {}


def policy_config_path() -> Path:
    """The campaign's frozen config (OEH_EVAL_INTERNAL_CONFIG.policy_config), else OEH_POLICY_CONFIG."""
    evaluation = json.loads(os.environ.get("OEH_EVAL_INTERNAL_CONFIG") or "{}")
    path = evaluation.get("policy_config") or os.environ.get("OEH_POLICY_CONFIG")
    if not path:
        raise ValueError("OEH_POLICY_CONFIG must name a frozen RoboCasa GR1 configuration")
    return Path(path).resolve()


def accept_validated(path: Path, cfg) -> None:
    """The runner validated this exact file in this process; do not hash the checkpoint again."""
    _VALIDATED.update(path=str(Path(path).resolve()), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                      cfg=cfg)


def _load_config(contract, path: Path):
    if _VALIDATED.get("path") == str(path) and \
            _VALIDATED.get("sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
        contract.trust(_VALIDATED["cfg"])  # binds the review rows and camera; the files were checked moments ago
        return _VALIDATED["cfg"]
    return contract.validate_config(json.loads(path.read_text(encoding="utf-8")))


def loop_settings(cfg) -> dict:
    """What ``single_eval.run_eval`` runs from its merged config, read with its own defaults."""
    return {"env_id": cfg.get("env_id", "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"),
            "seed": int(cfg.get("seed", 0)), "num_episodes": int(cfg.get("num_episodes", 1)),
            "max_steps": int(cfg.get("max_steps", 720)), "settle_steps": int(cfg.get("settle_steps", 0))}


class RoboCasaGR1Endpoint(Model):
    """The released client's ``reset``/``act``/``close`` with the OEH review runtime between OpenWAM and the env."""

    def __init__(self, official_cls, args, kwargs, loop_cfg):
        self.contract = RoboCasaGR1Contract()
        self.cfg = _load_config(self.contract, policy_config_path())
        native = self.cfg["robocasa_gr1"]
        for key in CLIENT_KEYS:
            if kwargs.get(key) != native[key]:
                raise ValueError(f"RoboCasa GR1 loop differs from the frozen configuration: {key} "
                                 f"{kwargs.get(key)!r}, frozen {native[key]!r}")
        loop = loop_settings(loop_cfg)
        expected = {"env_id": native["env_id"], "seed": native["seed"], "num_episodes": 1,
                    "max_steps": self.cfg["native_step_limit"], "settle_steps": 0}
        for key, value in expected.items():
            if loop[key] != value:
                raise ValueError(f"RoboCasa GR1 loop differs from the frozen configuration: {key} {loop[key]!r}, "
                                 f"frozen {value!r}")
        module = sys.modules.get(getattr(official_cls, "__module__", ""))
        client_path = Path(native["official_client_path"]).resolve()
        if module is None or Path(module.__file__).resolve() != client_path or \
                self.cfg["identity_files"].get(str(client_path)) is None:
            raise ValueError("the released RoboCasa GR1 client the loop constructs must be the pinned one")
        self.attempt_id = os.environ.get("ROBOCASA_GR1_RUN_ID")
        if self.attempt_id and (not re.fullmatch(r"[A-Za-z0-9_.-]+", self.attempt_id)
                                or self.attempt_id in {".", ".."}):
            raise ValueError("invalid attempt identity")
        self.events_path = Path(os.environ["OEH_EVAL_EVENTS"]).resolve()
        if not self.events_path.is_file():
            raise RuntimeError("the RoboCasa GR1 client owns the attempt sidecar; start through "
                               "oeh_policy.robocasa_gr1_runner")
        self.teacher_start = None
        teacher = (self.cfg.get("review") or {}).get("teacher") or {}
        if teacher.get("api") == "codex_app_server":
            from .codex_teacher import prestart
            from .teacher import prompt_text
            # As RoboTwin's endpoint: the app-server's start overlaps the released client's connection and the first
            # reset waits for it before its call bound starts (the prompt is the one make_client gives this runtime).
            self.teacher_start = prestart(teacher, prompt_text(self.cfg["review"].get("schedule"),
                                                               self.contract.benchmark))
        self.official = official_cls(*args, **kwargs)  # the released client: its server ping, its kinematics
        self.backend = OpenWAMBackend(self.official)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="oeh-robocasa-gr1-runtime", daemon=True)
        self.thread.start()
        self.runtime = None
        self.used = self.finished = False
        self.pending = None  # (ticket, row) of the action the loop is stepping
        self.native_steps = 0

    def reset(self):
        """The loop's ``policy.reset()`` after ``env.reset``: the one episode's runtime starts, and its start ends with
        the released client's own reset (``OpenWAMBackend.reset``)."""
        if self.used:
            raise RuntimeError("one RoboCasa GR1 episode per policy endpoint")
        self.used = True
        if self.teacher_start is not None:
            # Outside the call bound: nothing times the loop's reset, and a slow app-server start would otherwise
            # spend the first call's call_seconds (RoboDojo's Model waits before reporting ready).
            concurrent.futures.wait([self.teacher_start])
        self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id,
                               contract=self.contract, teacher_start=self.teacher_start)
        self.teacher_start = None
        try:
            self._call(self.runtime.start())
        except BaseException:
            self.finish("preflight_failed")
            raise

    def act(self, obs):
        """One loop step: the released payload for ``obs`` is the policy input, the runtime admits one EEF33 row, and
        the released client converts exactly that row into the env action the loop steps."""
        if self.runtime is None or self.finished:
            raise RuntimeError("act outside this endpoint's one episode")
        if self.pending is not None:
            raise RuntimeError("the evaluator reported no native step for the previous action; run through "
                               "oeh_policy.robocasa_gr1_runner")
        try:
            request = self.backend.capture(obs)
            self.backend.bind(obs, request)
            self.update_obs({"native": obs, "eef33": request["state"], "instruction": request["prompt"]})
            row = self.contract.validate_action(self._call(self.runtime.next_action()), self.cfg["action_limits"])
            if self.native_steps >= self.cfg["native_step_limit"]:
                raise RuntimeError("native RoboCasa GR1 episode ended before action authorization")
            ticket = self.before_action(row)
            self.append("native_action_submitted", {"trace_id": ticket["trace_id"],
                                                    "sha256": ticket["action_sha256"], "action": row})
            action = self.backend.convert(obs, row)
            self.pending = (ticket, row)
            return action
        except BaseException:
            self.finish("runner_failed")
            raise

    def native_stepped(self, obs, reward, terminated, truncated, info):
        """The evaluator's report of the native step the last ``act``'s action ran: the gate's acknowledgement with
        the state after it. The step that ends the loop (success as ``single_eval.py`` reads it, or the step limit)
        ends the runtime's episode."""
        if self.pending is None or self.finished:
            raise RuntimeError("a native step without an authorized action")
        (ticket, row), self.pending = self.pending, None
        try:
            self.native_steps += 1
            ended = bool(info.get("success", False) or reward > 0) or bool(terminated) or bool(truncated)
            terminal = ended or self.native_steps >= self.cfg["native_step_limit"]
            state = self.contract.project_state({"eef33": self.backend.measured(obs)})
            self.after_action({"ticket": ticket, "action": row, "state": state, "terminal": terminal})
            if terminal:
                self.append("native_terminal_observed", {"native_steps": self.native_steps,
                                                         "native_step_limit": self.cfg["native_step_limit"]})
                self.finish("native_terminal" if ended else "native_step_limit")
        except BaseException:
            self.finish("runner_failed")
            raise

    def append(self, kind, payload):
        row = {"kind": kind, "payload": plain(payload), "mono_ns": time.monotonic_ns()}
        with self.events_path.open("ab") as stream:
            stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, reason):
        """Close the episode once and move the runtime's events into the attempt's sidecar."""
        if self.finished:
            return
        self.finished = True
        try:
            self.close_episode({"reason": reason})
        finally:
            if self.runtime is not None:
                for row in self.drain_events():
                    self.append(row["kind"], row["payload"])

    def close(self):
        """The loop's ``policy.close()`` (its ``finally``): an episode it left open ends as runner_failed, then the
        released client closes its WebSocket."""
        try:
            self.finish("runner_failed")
        finally:
            self.official.close()


def make_policy(official_cls, args, kwargs, loop_cfg):
    """What the evaluator's client constructs where ``single_eval.run_eval`` constructs the released client."""
    return RoboCasaGR1Endpoint(official_cls, args, kwargs, loop_cfg)
