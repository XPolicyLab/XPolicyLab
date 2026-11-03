"""RoboTwin policy_name entry for the pinned official eval_policy.py loop.

The official loop owns expert seed selection, termination and `_result.txt`; the evaluator's
runner (`oeh_eval.robotwin_client`) owns the attempt's sidecar, the recorded observations and
the native-episode record. This callback owns only permitted policy input, one admitted EE16
submission, and OEH evidence from the actual native `take_action` boundary.
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
from .robotwin import OpenWAMBackend, RoboTwinContract, import_official_client
from .runtime import Runtime

_VALIDATED = {}


def policy_config_path(usr_args) -> Path:
    """The campaign's frozen config (OEH_EVAL_INTERNAL_CONFIG.policy_config), else the launcher's or OEH_POLICY_CONFIG."""
    evaluation = json.loads(os.environ.get("OEH_EVAL_INTERNAL_CONFIG") or "{}")
    path = (evaluation.get("policy_config") or usr_args.get("oeh_policy_config")
            or os.environ.get("OEH_POLICY_CONFIG"))
    if not path:
        raise ValueError("OEH_POLICY_CONFIG must name a frozen RoboTwin configuration")
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


def native_endpose(env):
    """Arm poses after a step as `get_obs()` reads them, without its re-render (which can draw from np.random)."""
    return {"left_endpose": env.get_arm_pose("left"), "left_gripper": env.robot.get_left_gripper_val(),
            "right_endpose": env.get_arm_pose("right"), "right_gripper": env.robot.get_right_gripper_val()}


class RoboTwinEndpoint(Model):
    def __init__(self, usr_args):
        self.contract = RoboTwinContract()
        self.cfg = _load_config(self.contract, policy_config_path(usr_args))
        native = self.cfg["robotwin"]
        for key in ("host", "port", "send_state", "state_dim", "action_type", "task_name",
                    "task_config", "seed", "instruction_type"):
            if key in native and str(usr_args.get(key)) != str(native[key]):
                raise ValueError(f"RoboTwin launcher differs from frozen configuration: {key}")
        self.attempt_id = os.environ.get("ROBOTWIN_RUN_ID")
        if self.attempt_id and (not re.fullmatch(r"[A-Za-z0-9_.-]+", self.attempt_id)
                                or self.attempt_id in {".", ".."}):
            raise ValueError("invalid attempt identity")
        self.events_path = Path(os.environ["OEH_EVAL_EVENTS"]).resolve()
        if not self.events_path.is_file():
            raise RuntimeError("the RoboTwin runner owns the attempt sidecar; start through oeh_policy.robotwin_runner")
        client_path = Path(native["official_client_path"]).resolve()
        if self.cfg["identity_files"].get(str(client_path)) is None:
            raise ValueError("actual RoboTwin OpenWAM client must be pinned")
        root = str(Path(native["openwam_root"]).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        self.teacher_start = None
        teacher = (self.cfg.get("review") or {}).get("teacher") or {}
        if self.cfg["mode"] == "review" and teacher.get("api") == "codex_app_server":
            from .codex_teacher import prestart
            from .teacher import prompt_text
            # As oeh_policy.model.Model does: the app-server's start (its bounded retries and preflight) overlaps the
            # client's connection and RoboTwin's expert check, and the first reset waits for it before its call bound
            # starts. The prompt is the one make_client would give this runtime (RoboTwin's prompt set).
            self.teacher_start = prestart(teacher, prompt_text(self.cfg["review"].get("schedule"),
                                                               self.contract.benchmark))
        official = import_official_client(client_path)
        self.backend = OpenWAMBackend(official.get_model(usr_args), official)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="oeh-robotwin-runtime", daemon=True)
        self.thread.start()
        self.runtime = None
        self.used = False
        self.native_calls = 0

    def get_action(self):
        # One native action per official eval() call; rows of an admitted slow chunk follow in order.
        return [self._call(self.runtime.next_action())]

    def reset(self):
        if self.used:
            raise RuntimeError("one RoboTwin episode per policy endpoint")
        self.used = True
        if self.teacher_start is not None:
            # Outside the call bound: nothing times the official loop's reset_model, and a slow app-server start
            # would otherwise spend the first call's call_seconds (RoboDojo's Model waits before reporting ready).
            concurrent.futures.wait([self.teacher_start])
        self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id,
                               contract=self.contract, teacher_start=self.teacher_start)
        self.teacher_start = None
        try:
            return self._call(self.runtime.start())
        except BaseException:
            self.close_episode({"reason": "preflight_failed"})
            raise

    def append(self, kind, payload):
        row = {"kind": kind, "payload": plain(payload), "mono_ns": time.monotonic_ns()}
        with self.events_path.open("ab") as stream:
            stream.write(encoded(row))
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, reason):
        try:
            self.close_episode({"reason": reason})
        finally:
            try:
                for row in self.drain_events():
                    self.append(row["kind"], row["payload"])
            finally:
                self.backend.client._client.close()


def get_model(usr_args):
    return RoboTwinEndpoint(usr_args)


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    """The official callback shape; native judge remains in eval_policy.py."""
    if not hasattr(TASK_ENV, "_oeh_last_setup_seed"):
        raise RuntimeError("the evaluator's native reset hook was not installed")
    instruction = str(TASK_ENV.get_instruction())
    value = {"native": observation, "instruction": instruction}
    if model.native_calls == 0:
        model.append("native_reset", {"actual_seed": TASK_ENV._oeh_last_setup_seed,
                                      "task": TASK_ENV.task_name, "native_step": TASK_ENV.take_action_cnt})
    model.native_calls += 1
    try:
        model.backend.official._apply_step_lim_override(TASK_ENV)
        model.update_obs(value)
        actions = model.get_action()
        if len(actions) != 1:
            raise ValueError("RoboTwin official client expects exactly one native action")
        action = model.contract.validate_action(actions[0], model.cfg["action_limits"])
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            raise RuntimeError("native RoboTwin episode ended before action authorization")
        ticket = model.before_action(action)
        model.append("native_action_submitted", {"trace_id": ticket["trace_id"],
                     "sha256": ticket["action_sha256"], "action": action})
        # Exactly one submission. An ambiguous exception is not retried.
        previous_step = TASK_ENV.take_action_cnt
        TASK_ENV.take_action(action, action_type="ee")
        if TASK_ENV.take_action_cnt != previous_step + 1:
            raise RuntimeError("native RoboTwin action submission was not acknowledged")
        terminal = bool(TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim)
        state = model.contract.project_state(native_endpose(TASK_ENV))
        model.after_action({"ticket": ticket, "action": action, "state": state,
                            "terminal": terminal})
        if terminal:
            model.append("native_terminal_observed", {"native_steps": TASK_ENV.take_action_cnt,
                                                       "native_step_limit": TASK_ENV.step_lim})
            model.finish("native_terminal" if TASK_ENV.eval_success else "native_step_limit")
    except BaseException:
        if not model.loop.is_closed():
            model.finish("runner_failed")
        raise
