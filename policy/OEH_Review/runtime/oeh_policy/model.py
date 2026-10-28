"""The same XPolicyLab Model endpoint serves fast and slow frozen configurations."""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import importlib
import json
import os
from pathlib import Path
import re
import threading

from .contracts import digest, file_digest, validate_config
from .openwam_protocol import SESSION_ENV, remote_xpolicylab_model
from .runtime import Runtime

IDENTITY_KEYS = ("ckpt_dir", "openwam_root", "openwam_deploy_config", "env_cfg_type", "action_type")
PATH_KEYS = ("ckpt_dir", "openwam_root", "openwam_deploy_config")
OEH_ONLY_KEYS = frozenset({"oeh_policy_config"})


def official_arguments(frozen, launcher) -> dict:
    """The mapping handed to the official OpenWAM Model: the launcher's model_cfg with the frozen block as defaults.

    A launcher value of None counts as unset and takes the frozen value. Identity-critical fields present in both
    must agree (paths compared resolved). The result is what report_policy records, so the campaign can compare it
    with the direct condition's report field by field.
    """
    frozen = copy.deepcopy(dict(frozen or {}))
    if not frozen.get("ckpt_dir") or not frozen.get("openwam_root"):
        raise ValueError("freeze explicit OpenWAM checkpoint and source paths")
    official = dict(frozen)
    for key, value in launcher.items():
        if key in OEH_ONLY_KEYS or value is None:
            continue
        if key in IDENTITY_KEYS and key in frozen:
            if key in PATH_KEYS:
                same = str(Path(str(value)).expanduser().resolve()) == str(Path(str(frozen[key])).expanduser().resolve())
            else:
                same = str(value) == str(frozen[key])
            if not same:
                raise ValueError(f"launcher differs from frozen OpenWAM configuration: {key}")
        official[key] = value
    if _flag(official.get("allow_dummy_policy", False)):
        raise ValueError("dummy policy cannot run this experiment")
    return official


def _flag(value) -> bool:
    """XPolicyLab's launcher passes YAML/shell booleans as strings ("false"); read them the way it means them."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class Model:
    def __init__(self, model_cfg):
        evaluation = json.loads(os.environ.get("OEH_EVAL_INTERNAL_CONFIG", "{}"))
        config_path = evaluation.get("policy_config") or model_cfg.get("oeh_policy_config") or os.environ.get("OEH_POLICY_CONFIG")
        if not config_path:
            raise ValueError("OEH_POLICY_CONFIG must select a frozen policy configuration")
        self.cfg = validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
        self.attempt_id = os.environ.get("ROBODOJO_RUN_ID")
        if self.attempt_id and not re.fullmatch(r"[A-Za-z0-9_.-]+", self.attempt_id):
            raise ValueError("invalid native run identity")
        if self.attempt_id in {".", ".."}:
            raise ValueError("invalid native run identity")
        if model_cfg.get("action_type", "ee") != "ee":
            raise ValueError("this policy implements the official OpenWAM EE action contract")
        module = importlib.import_module("XPolicyLab.policy.OpenWAM.model")
        source = str(Path(module.__file__).resolve())
        if self.cfg["identity_files"].get(source) != file_digest(Path(source)):
            raise ValueError("the actual imported OpenWAM adapter must be in identity_files")
        # The official OpenWAM Model receives the launcher's own model_cfg, exactly as the direct condition does
        # (parity by construction); the frozen `openwam` block supplies defaults and the identity-critical fields
        # must agree, so a launcher naming other weights, sources or robot fails before loading anything.
        official = official_arguments(self.cfg.get("openwam", {}), model_cfg)
        checkpoint = Path(official["ckpt_dir"]).resolve()
        files = {str(p.relative_to(checkpoint)): file_digest(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()}
        if not files or digest(files) != self.cfg["checkpoint_sha256"]:
            raise ValueError("checkpoint directory identity mismatch")
        normalizer = checkpoint / "normalization_stats.npy"
        if file_digest(normalizer) != self.cfg["normalization_sha256"]:
            raise ValueError("normalization identity mismatch")
        source_root = Path(official["openwam_root"]).resolve()
        sources = {str(p.relative_to(source_root)): file_digest(p)
                   for p in sorted((source_root / "openwam").rglob("*.py"))}
        if not sources or digest(sources) != self.cfg.get("openwam_source_sha256"):
            raise ValueError("OpenWAM source identity mismatch")
        from oeh_eval.runtime_config import report_policy
        report_policy(official)
        self.teacher_start = None
        teacher = (self.cfg.get("review") or {}).get("teacher") or {}
        if self.cfg["mode"] == "review" and teacher.get("api") == "codex_app_server":
            from .codex_teacher import prestart
            from .teacher import prompt_text
            # Overlaps the OpenWAM load; the first reset takes it. The prompt is the one make_client would give.
            self.teacher_start = prestart(teacher, prompt_text(self.cfg["review"].get("schedule")))
        session = os.environ.get(SESSION_ENV)
        if session:
            # Shared OpenWAM serving: the same official adapter, its engine a session of the GPU's shared service
            # that the attempt's launcher leased (oeh_eval.robodojo_launch).
            self.backend = remote_xpolicylab_model(module.Model)(official, session)
        else:
            self.backend = module.Model(official)
        if self.teacher_start is not None:
            # The rest of the start finishes here, before the server reports ready: what is left of it at the first
            # reset counts against that call's bound (call_seconds + 15 s), which a slow app-server start exceeds
            # (comparison lock c93d419: three review/hybrid attempts timed out at their first reset). Every step of
            # the start is bounded (LAUNCH_ATTEMPTS of LAUNCH_TIMEOUT_S, then two 60 s requests); a failure stays
            # in the future for the runtime's start to report.
            concurrent.futures.wait([self.teacher_start])
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name="oeh-policy-runtime", daemon=True)
        self.thread.start()
        self.runtime = None
        self.used = False
        self.reset_response = None
        if self.cfg["mode"] == "review-stmem":
            # Load perception and authenticate before advertising the policy server as
            # ready. Official reset/action calls retain their <120-second deadline.
            self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id)
            future = asyncio.run_coroutine_threadsafe(self.runtime.start(), self.loop)
            try:
                self.reset_response = future.result(timeout=self.cfg["stmem"]["startup_seconds"] + 180)
                self.used = True  # official resets use repeat_reset before any observation
            except BaseException:
                future.cancel()
                asyncio.run_coroutine_threadsafe(self.runtime.close("prewarm_failed"), self.loop).result(timeout=15)
                raise

    def _timeout(self):
        """Bound on one call. A hybrid get_action may hold a refused fast chunk, a slow decision and the fast
        decision that follows a failed one, each bounded by decision_seconds, plus the wait for the slow turn
        to end; before the first decision, add what first_decision_seconds allows beyond decision_seconds.
        A review get_action ends within review.call_seconds (the OpenWAM inference, the wait for the teacher and
        the hold that replaces a late decision all fit in it); the first call may add first_call_extra_seconds.
        Mirrored in oeh_eval.matrix_emit.model_call_seconds."""
        if self.cfg["mode"] in {"review", "review-stmem"}:
            review = self.cfg["review"]
            first = self.runtime is None or self.runtime.calls == 0
            return review["call_seconds"] + (review["first_call_extra_seconds"] if first else 0) + 15
        seconds = self.cfg["decision_seconds"]
        decisions = 4 if self.cfg["mode"] == "hybrid" else 1
        warmup = 0
        if self.runtime is None or self.runtime.calls == 0:
            warmup = max(0, self.cfg.get("first_decision_seconds", seconds) - seconds)
        return seconds * decisions + warmup + 15

    def _call(self, coro):
        if self.loop.is_closed():
            coro.close()
            raise RuntimeError("policy server episode has closed")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=self._timeout())
        except concurrent.futures.TimeoutError:
            future.cancel()
            if self.runtime:
                asyncio.run_coroutine_threadsafe(self.runtime.close("caller_timeout"), self.loop)
            raise RuntimeError("OEH policy call timed out; discard this server process") from None

    def reset(self):
        # The pilot has eval_num=1. Never reuse potentially detached model work or
        # carry episode memory across trials. A new trial starts a new server.
        # One episode, two official resets: RoboDojo's eval client resets the policy at the end of every
        # environment reset (src/eval_client/eval_env.py:245 at ee67a14), then XPolicyLab's official OpenWAM
        # episode resets it again before its first observation (policy/OpenWAM/deploy.py:2 at fa431ec). OpenWAM's
        # own reset only clears its batch buffers, so the direct condition gets two resets with the effect of
        # one. A reset before this episode's first observation therefore repeats the backend reset, as the direct
        # condition does, and keeps the episode; a reset after it, or after the episode stopped, is refused.
        # When this server can serve no further episode (its runtime failed to start, or the official loop asks for a
        # second episode), it says so in policy-fatal.json: RoboDojo's client would otherwise move on seed after seed
        # (src/eval_client/main.py at ee67a14) until the attempt's wall, and the evaluator stops the attempt instead.
        if self.used:
            runtime = self.runtime
            if (runtime is None or self.loop.is_closed() or runtime.observation is not None or runtime.calls
                    or runtime.gate.stopped):
                self._fatal("reset after this policy server's one episode")
                raise RuntimeError("one episode per policy server; start a fresh server for the next attempt")
            self._call(runtime.repeat_reset())
            return self.reset_response
        self.used = True
        self.runtime = Runtime(self.cfg, self.backend, attempt_id=self.attempt_id, teacher_start=self.teacher_start)
        self.teacher_start = None
        try:
            self.reset_response = self._call(self.runtime.start())
        except BaseException as exc:
            self.close_episode({"reason": "preflight_failed"})
            self._fatal(f"policy runtime failed to start: {type(exc).__name__}: {exc}")
            raise
        return self.reset_response

    @staticmethod
    def _fatal(reason):
        from oeh_eval.runtime_config import report_policy_fatal
        report_policy_fatal(reason)

    def update_obs(self, obs):
        return self._call(self.runtime.update_obs(obs))

    def get_action(self):
        return self._call(self.runtime.get_action())

    def before_action(self, action):
        return self._call(self.runtime.before_action(action))

    def after_action(self, obs):
        if set(obs) != {"ticket", "action", "state", "terminal"}:
            raise ValueError("native acknowledgment requires ticket, action, state and terminal")
        return self._call(self.runtime.after_action(**obs))

    def drain_events(self):
        if self.loop.is_closed():
            values = copy.deepcopy(self.runtime.events[self.runtime.drain_cursor:])
            self.runtime.drain_cursor = len(self.runtime.events)
            return values
        return self._call(self.runtime.drain_events())

    def close_episode(self, obs=None):
        if getattr(self, "teacher_start", None) is not None:
            from .codex_teacher import discard
            discard(self.teacher_start)
            self.teacher_start = None
        if self.runtime and not self.loop.is_closed():
            self._call(self.runtime.close((obs or {}).get("reason", "episode_closed")))
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("policy runtime did not stop")
            self.loop.close()
        return {"closed": True}

    def update_obs_batch(self, obs):
        raise ValueError("first experiment requires eval_batch=false and one environment")

    def get_action_batch(self, env_idx_list=None):
        raise ValueError("first experiment requires eval_batch=false and one environment")
