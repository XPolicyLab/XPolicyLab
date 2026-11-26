"""Keep XPolicyLab's official loop; guard its sole native submission boundary.

This requires the companion Model methods before_action/after_action. The generic
OEH Eval external wrapper alone cannot hold and acknowledge OEH execution leases.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading

from .contracts import CAMERAS, digest, plain, project_state


class NativeTerminal(Exception):
    """Legacy loop-stop signal; AgentTermination names opt-in policy-driven closure."""


class AgentTermination(NativeTerminal):
    """Explicit policy declaration, never a benchmark success result."""

    def __init__(self, control):
        self.control = control
        super().__init__(control["status"])


class EpisodeControlClient:
    """Intercept the opt-in control envelope before the official loop treats it as action rows."""

    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def call(self, *args, **kwargs):
        result = self.source.call(*args, **kwargs)
        method = kwargs.get("func_name") or (args[0] if args else None)
        if method == "get_action" and isinstance(result, dict) and "oeh_episode_control" in result:
            from .q1_actions import validate_finish
            control = result["oeh_episode_control"]
            expected = {"kind", "observation_id", "status", "reason", "native_step", "claim_source", "benchmark_outcome"}
            if (set(result) != {"oeh_episode_control"} or not isinstance(control, dict) or set(control) != expected
                    or control["kind"] != "agent_termination" or control["claim_source"] != "agent"
                    or control["benchmark_outcome"] != "owned_by_evaluator"
                    or type(control["native_step"]) is not int or control["native_step"] < 0):
                raise ValueError("invalid agent termination envelope")
            validate_finish({key: control[key] for key in ("observation_id", "status", "reason")})
            raise AgentTermination(control)
        return result


def _finalize_agent_termination(task_env):
    """Ask pinned RoboDojo to judge an early stop, then let run_eval save its result.

    RoboDojo ee67a14 starts ``success`` as True: returning from its policy loop
    without this check would count an unfinished episode as a success. Its
    existing failure-state input to is_episode_end requests a final reward check;
    that native check may still judge success. This is the same hook used by
    RoboProbe 3c0bc683's _mark_incomplete_episode_failed, with explicit single-env
    compatibility checks. Neither the model's status nor a fabricated step-limit
    change supplies the outcome. No motion or model call is made here.
    """
    if (getattr(task_env, "num_envs", None) != 1
            or not callable(getattr(task_env, "is_episode_end", None))
            or not callable(getattr(task_env, "get_running_env_idx_list", None))
            or not isinstance(getattr(task_env, "success", None), list)
            or len(task_env.success) != 1 or type(task_env.success[0]) is not bool
            or not isinstance(getattr(task_env, "end_flag", None), list)
            or len(task_env.end_flag) != 1 or type(task_env.end_flag[0]) is not bool):
        raise RuntimeError("agent termination requires RoboDojo's single-environment native finalization hook")
    if task_env.is_episode_end():
        if not task_env.end_flag[0]:
            raise RuntimeError("native finalization returned terminal without an end flag")
        return "already_native_terminal"
    if task_env.get_running_env_idx_list() != [0] or task_env.end_flag[0]:
        raise RuntimeError("agent termination found an inconsistent native running environment")
    # This is a request for native final_check, not the saved success judgment.
    # is_episode_end owns that judgment, including success on its final check.
    task_env.success[0] = False
    if not task_env.is_episode_end() or not task_env.end_flag[0]:
        raise RuntimeError("RoboDojo did not finalize the agent's early-stop request")
    return "native_final_check"


# Runtime events cross to the sidecar once per OpenWAM chunk. A single drain at the end of a 564- and a 1056-step
# RoboDojo episode (2026-09-25, finding L13) reset the policy server connection while the reply was sent; the
# exception then took the native result with it and the attempts sealed aborted.
DRAIN_EVERY_STEPS = 32
# The cleanup calls after the official episode (the last step's acknowledgement, close_episode, the final drain) are
# OEH's alone; direct makes none. A policy server that crashed there left XPolicyLab's client reconnecting for about
# 900 s, past the evaluator's 600 s watch, so RoboDojo never wrote the episode's native result (harness audit
# 2026-10-02, ATTR-1). Each call now gets this long; a call that does not return is abandoned on its daemon thread, the
# rest are skipped (the client is still busy with it), and the sidecar says so.
CLEANUP_SECONDS = 60.0
# Written by this gateway alone when the official loop has ended its episode; the runtime's drained events may not use
# it, so the evaluator can tell "the episode ended, then cleanup lost the result" from a run that never finished.
EPISODE_ENDED = "official_episode_ended"
RESERVED_KINDS = frozenset({EPISODE_ENDED, "agent_episode_ended", "agent_episode_finalized"})


class _Cleanup:
    """Runs each cleanup call with a time limit and records every failure in the sidecar; never raises."""

    def __init__(self, log, seconds=None):
        self.log, self.seconds, self.hung = log, CLEANUP_SECONDS if seconds is None else seconds, None

    def gated(self, log):
        """The log a cleanup call writes through: once a call is abandoned, its thread's late rows are dropped, so they
        cannot interleave with the rows written after it."""
        cleanup = self

        class Gated:
            def __getattr__(self, key):
                return getattr(log, key)

            def append(self, kind, payload):
                if cleanup.hung is None:
                    log.append(kind, payload)

        return Gated()

    def run(self, stage, call):
        if self.hung is not None:
            self.log.append("cleanup_failure", {"stage": stage, "error": "skipped",
                                                "detail": f"{self.hung} did not return"})
            return
        failure = []
        done = threading.Event()

        def target():
            try:
                call()
            except BaseException as exc:  # noqa: BLE001 - recorded, never raised: the native result comes first
                failure.append(type(exc).__name__)
            finally:
                done.set()

        threading.Thread(target=target, daemon=True, name=f"oeh-cleanup-{stage}").start()
        if not done.wait(self.seconds):
            self.hung = stage
            self.log.append("cleanup_failure", {"stage": stage, "error": "timeout", "seconds": self.seconds})
        elif failure:
            self.log.append("cleanup_failure", {"stage": stage, "error": failure[0]})


class NativeGateway:
    """RoboDojo's task environment as the official episode sees it; every native submission goes through OEH.

    The official loop (XPolicyLab deploy.py) drives the environment in its own order: take_action for every row of a
    chunk, get_obs between rows and before the next chunk, is_episode_end only between chunks. RoboDojo's
    is_episode_end is not a query (ee67a14 eval_env.py: it scores the reward, sets the end and success flags and
    renders a last frame when they change), so calling it at every step would judge a mid-chunk success the direct
    condition never sees. This gateway adds no environment call: it never calls is_episode_end, the receipt of a
    native step is sent when the official loop reads the next observation (that step's post-step state), and the
    episode's last step is acknowledged by `finish()` once the loop has ended.
    """

    def __init__(self, env, client, log, *, drain_every=DRAIN_EVERY_STEPS, stmem_calibration=False):
        self.env, self.client, self.log = env, client, log
        self.submitted = None  # (ticket, action) of the last native step until its post-step observation
        self.drain_every = drain_every
        self.acknowledged = 0
        self.submissions = 0
        self.stmem_calibration = stmem_calibration

    def __getattr__(self, key):
        return getattr(self.env, key)

    def _acknowledge(self, obs):
        ticket, action = self.submitted
        self.submitted = None
        self.client.call(func_name="after_action", obs={"ticket": ticket, "action": action,
                         "state": project_state(obs["state"]), "terminal": False})
        self.acknowledged += 1
        if self.drain_every and self.acknowledged % self.drain_every == 0:
            # Evidence transfer, not control: a failed periodic drain must not end the episode (review 2026-09-28,
            # e9). The row makes the evaluator flag the attempt's evidence; the runtime's own events keep the rest.
            try:
                self.drain()
            except Exception as exc:
                self.log.append("cleanup_failure", {"stage": "periodic_drain", "error": type(exc).__name__})

    def drain(self):
        """Append the runtime's events since the last drain to the sidecar, in the runtime's order."""
        for event in self.client.call(func_name="drain_events"):
            if event["kind"] in RESERVED_KINDS:
                raise ValueError(f"the policy runtime sent an event of reserved kind {event['kind']!r}")
            self.log.append(event["kind"], event["payload"])

    def get_obs(self):
        obs = self.env.get_obs()
        if self.submitted is not None:
            self._acknowledge(obs)
        # Native wire image arrays are decoded by XPolicyLab on the server. Keep
        # that representation and cadence, but project all nested keys here too.
        projected = {"state": project_state(obs["state"]), "vision": {
            camera: {"color": obs["vision"][camera]["color"]} for camera in CAMERAS}}
        if self.stmem_calibration:
            from .stmem_calibration import project_calibration
            for camera in CAMERAS:
                if camera=='cam_head' or obs['vision'][camera].get('stmem_calibration') is not None:
                    projected['vision'][camera]['stmem_calibration']=project_calibration(obs['vision'][camera])
        for key in ("instruction", "task_instruction", "env_idx"):
            if key in obs:
                projected[key] = obs[key]
        # ModelClientProxy records exactly what crosses the policy boundary.
        # Logging here as well would create a second observation for one call.
        return projected

    def take_action(self, action):
        if self.submitted is not None:
            raise RuntimeError("a native step was submitted before the previous one was acknowledged")
        ticket = self.client.call(func_name="before_action", obs=action)
        self.log.append("native_action_submitted", {"action": plain(action),
                         "sha256": ticket["action_sha256"], "trace_id": ticket["trace_id"]})
        # Exactly one call, with no retry if the result is ambiguous.
        result = self.env.take_action(action)
        self.submitted = (ticket, action)
        self.submissions += 1
        return result

    def finish(self):
        """After the official loop has ended: acknowledge its last native step with the final observation."""
        if self.submitted is not None:
            self.acknowledge(self.env.get_obs())

    def acknowledge(self, obs):
        """Acknowledge the last native step with an observation already read from the environment."""
        if self.submitted is not None:
            self._acknowledge(obs)


def eval_one_episode(task_env, model_client):
    from XPolicyLab.policy.OpenWAM.deploy import eval_one_episode as official_episode
    from oeh_eval.direct_bridge import BridgeLog, ModelClientProxy
    from oeh_eval.runtime_config import report_env

    event_path = os.environ.get("OEH_EVAL_EVENTS")
    if not event_path:
        raise RuntimeError("OEH_EVAL_EVENTS must name the attempt's fresh evidence sidecar")
    permitted = os.environ.get("OEH_EVAL_PERMITTED_OBSERVATIONS")
    stmem_calibration = os.environ.get("OEH_STMEM_CALIBRATION") == "1"
    log = BridgeLog(Path(event_path), set(permitted.split(",")) if permitted else None,
                    stmem_calibration=stmem_calibration)
    report_env(task_env)
    gateway = NativeGateway(task_env, model_client, log, stmem_calibration=stmem_calibration)
    reason = "native_terminal"
    cleanup = _Cleanup(log)
    gateway.log = cleanup.gated(log)
    try:
        try:
            official_episode(gateway, EpisodeControlClient(ModelClientProxy(model_client, log)))
        except AgentTermination as ended:
            reason = "agent_requested_termination"
            log.append("agent_episode_ended", ended.control)
            finalized_by = _finalize_agent_termination(task_env)
            control_sha256 = digest(ended.control)
            log.append("agent_episode_finalized", {"method": finalized_by,
                       "benchmark_outcome": "owned_by_evaluator", "control_sha256": control_sha256})
        except NativeTerminal:
            pass
        ended_payload = {"native_submissions": gateway.submissions}
        if reason == "agent_requested_termination":
            ended_payload["termination_source"] = "agent"
            ended_payload["control_sha256"] = control_sha256
        log.append(EPISODE_ENDED, ended_payload)
        # The last step's acknowledgement is cleanup too: a failure here used to re-raise and take the native result
        # with it (ATTR-1); the missing receipt is recorded instead. The final observation is read here, on the
        # official loop's thread: the simulator is never touched from a cleanup thread that may be abandoned, and
        # only the RPC to the OEH runtime is bounded.
        if gateway.submitted is not None:
            try:
                final = task_env.get_obs()
            except Exception as exc:  # noqa: BLE001 - recorded, never raised: the native result comes first
                log.append("cleanup_failure", {"stage": "final_observation", "error": type(exc).__name__})
            else:
                cleanup.run("final_acknowledgement", lambda: gateway.acknowledge(final))
    except BaseException:
        reason = "runner_failed"
        raise
    finally:
        # The official loop has ended: its native result is the evaluator's, and a failure to close or drain the
        # policy side must not take it with it. Such a failure is recorded in the sidecar, where the campaign's
        # evidence validation sees it; missing receipts are never synthesized.
        cleanup.run("close_episode", lambda: model_client.call(func_name="close_episode", obs={"reason": reason}))
        # Preserve events even when inference/admission/native stepping failed.
        cleanup.run("drain_events", gateway.drain)
        # The observation the model still holds is the episode's last: BridgeLog stores it whole, then closes. Local
        # file writes only (no policy-side call), so not bounded like the calls above; a failure is recorded.
        try:
            log.close()
        except Exception as exc:  # noqa: BLE001 - recorded, never raised: the native result comes first
            log.append("cleanup_failure", {"stage": "close_evidence", "error": type(exc).__name__})


def eval_one_episode_batch(task_env, model_client):
    raise ValueError("OEH first experiment requires eval_batch=false")
