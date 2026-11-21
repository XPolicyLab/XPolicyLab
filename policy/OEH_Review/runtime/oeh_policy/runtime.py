"""One episode, one OEH kernel, one decision at a time, one native command owner."""
from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import uuid

from PIL import Image
from oeh_sdk import Brain, Client, Plugin

from . import brain_preflight
from .actions import MODE_FIELDS, Method, check_slow_settings, intent_schema, mode_fields, validate_action
from .escalation import Escalation
from .contracts import (CAMERAS, CAPABILITY, digest, encoded, file_digest,
                        observation_digest, plain, project_observation, project_state, validate_config)
from .gateway import ExecutionGate
from .review import ARMS, ReviewMethod, executed_record
from .review_guard import ReviewGuard
from .takeover import Segment, Takeover, trigger_view
from .teacher import WINDOW_CLOSED
from .teacher_bridge import FROM_LANE



def serving_timing(backend) -> dict:
    """{"openwam_serving": {calls, generates, queue_ms, gpu_ms}} for a backend whose OpenWAM is a shared service's
    session (its time waiting behind other sessions is not this model call's own); {} for an in-process OpenWAM."""
    take = getattr(backend, "take_serving_timing", None)
    timing = take() if callable(take) else None
    return {"openwam_serving": timing} if timing is not None else {}

class FastRefused(RuntimeError):
    """OEH refused a fast chunk. In hybrid mode this escalates instead of ending the episode."""


class SlowFailed(RuntimeError):
    """A hybrid slow decision ended before anything reached the gate; the fast path takes the same observation."""

    def __init__(self, reason, *, no_action=False, **detail):
        super().__init__(reason)
        self.kind = "slow_decision_no_action" if no_action else "slow_decision_failed"
        self.evidence = {"reason": reason, **detail}


FAST_TICK = "obs/fast_tick"
# RoboTwin runs this runtime inside its simulator process, where native calls (the first render, planner setup) can
# hold the GIL for seconds and the runtime's loop thread then sends nothing. The kernel disposes a plugin 3 x
# heartbeat_ms after its last beat (oeh-rpc conn.rs; the plugin's requested period is honoured) and cancels an
# execution lease_ttl_s after its last lease beat, so both tolerate a 30 s stall: on the pod (2026-09-25) a 7 s
# first-frame stall disposed the 1 s-heartbeat plugin before the first decision. They bound only how long a dead
# plugin goes unnoticed; admission is unchanged. The same values serve every benchmark.
PLUGIN_HEARTBEAT_MS = 10_000
LEASE_TTL_S = 30.0
EXECUTE_ACCEPT_MS = 30_000
MAX_EVENT_BYTES = 8 * 1024 * 1024
# review.digest_max_chunks (G5), when the design leaves it out: how many unreviewed chunks the first packet after them
# carries in full (the newest; the counts in takeover.openwam_alone_since_last_packet cover all of them). [proposal:
# 16 chunks of 32 steps is 20 s at 25 Hz, above the 10 s hand-back cooldown and a segment of 8 chunks; the study's
# question for it: "how many unreviewed chunks does the first packet after them carry in full?"]
DIGEST_MAX_CHUNKS = 16


class RoboDojoContract:
    """Default native contract. Other benchmarks supply these exact boundaries."""

    action_contract = "XPolicyLab OpenWAM native EE, env-relative world, wxyz"
    ledger_kind = "obs/arx_x5_ledger"
    benchmark = "robodojo"
    robot_id = "arx_x5"
    persona_path = Path(__file__).with_name("method.txt")

    def validate_config(self, value):
        return validate_config(value)

    def make_method(self, cfg, backend=None):
        return ReviewMethod(cfg) if cfg.get("mode") in {"review", "review-stmem"} else Method(cfg)

    def intent_schema(self, max_steps, modes=None, *, pose_control=False):
        """``modes`` (brain.intent_modes, slow mode): the enum and fields the slow brain is shown; None, all."""
        return intent_schema(max_steps, modes, pose_control=True) if pose_control else (
            intent_schema(max_steps) if modes is None else intent_schema(max_steps, modes))

    def project_observation(self, obs):
        return project_observation(obs)

    def project_state(self, state):
        return project_state(state)

    def validate_action(self, action, limits):
        return validate_action(action, limits)

    def motion_features(self, state, action):
        """Arm positions observed after a step and the grippers it commanded, binarized at 0.5
        (previous_command semantics: 1 open, 0 closed), so only open/closed crossings count as reversals."""
        return ([state[arm + "_ee_pose"][:3] for arm in ("left", "right")],
                [float(float(action[arm + "_ee_joint_state"][0]) > 0.5) for arm in ("left", "right")])

    def camera_payload(self, runtime):
        return runtime._robodojo_camera_payload()

    # The review port (review mode; docs/benchmark-adapters/README.md): where the candidate and the teacher's views come
    # from. A benchmark's contract implements these with make_method's review.ReviewMethod (its action space).
    review_cameras = CAMERAS  # the views every review packet carries, in order

    def review_chunk(self, backend):
        """The review candidate: one fresh OpenWAM chunk of native rows on the current observation."""
        return backend.get_action()

    def review_views(self, observation):
        """The camera views the teacher sees, exactly ``review_cameras`` in order: name -> HxWx3 uint8 RGB."""
        return {name: observation["vision"][name]["color"] for name in self.review_cameras}

    def model_observation(self, runtime, projected):
        """What the VLA backend receives for a new native observation."""
        return projected

    def subgoal_observation(self, runtime, subgoal):
        """The backend input for a slow VLA decision: this decision's prefix only."""
        obs = copy.deepcopy(runtime.observation)
        obs["instruction"] = subgoal
        obs.pop("task_instruction", None)
        return obs

    def escalated(self, runtime):
        """Hook run once the slow brain's escalation payload has been built."""

    def ledger_payload(self, runtime, projected):
        return {"episode_id": runtime.episode_id, "seq": runtime.gate.step, "registration": 1,
                "gripper_cmd": runtime.method.gripper_commands,
                "eef": {arm: {"position": plain(projected["state"][arm + "_ee_pose"][:3]),
                               "quaternion_wxyz": plain(projected["state"][arm + "_ee_pose"][3:])}
                        for arm in ("left", "right")}}


class MotionPlugin(Plugin):
    def __init__(self, runtime):
        super().__init__(name="oeh-policy-motion", provide=["adapter.oeh-policy"], heartbeat_ms=PLUGIN_HEARTBEAT_MS)
        self.runtime = runtime
        if runtime.cfg["mode"] in {"review", "review-stmem"}:
            schema = runtime.method.review_schema()
        elif (runtime.cfg.get("brain") or {}).get("pose_control"):
            schema = runtime.contract.intent_schema(runtime.cfg["max_chunk_steps"],
                        runtime.cfg["brain"].get("intent_modes"), pose_control=True)
        elif (runtime.cfg.get("brain") or {}).get("intent_modes") is not None:
            # G4/M3: the design's intent vocabulary is what the slow brain is shown and what admission takes.
            schema = runtime.contract.intent_schema(runtime.cfg["max_chunk_steps"], runtime.cfg["brain"]["intent_modes"])
        else:
            schema = runtime.contract.intent_schema(runtime.cfg["max_chunk_steps"])
        self.capability(CAPABILITY, "motion", "high", schema, lease_ttl_s=LEASE_TTL_S)(self.act)
        if runtime.slow_settings.get("agent_termination"):
            from .q1_actions import FINISH_CAPABILITY, finish_schema
            self.capability(FINISH_CAPABILITY, "control", "low", finish_schema(),
                            lease_ttl_s=LEASE_TTL_S)(self.finish)

    async def finish(self, params, ctx):
        from .q1_actions import validate_finish
        runtime = self.runtime
        params = validate_finish(params)
        if (not runtime.slow_settings.get("agent_termination") or runtime.active != "slow"
                or runtime.consumed or params["observation_id"] != runtime.observation_id
                or runtime.gate.pending is not None or runtime.gate.inflight is not None
                or runtime.gate.stopped or ctx.cancelled.is_set()):
            raise ValueError("finish needs the current unconsumed slow observation and no motion in flight")
        runtime.consumed = True
        runtime.agent_termination = {**params, "native_step": runtime.gate.step,
            "claim_source": "agent", "benchmark_outcome": "owned_by_evaluator"}
        envelope = {"oeh_episode_control": {"kind": "agent_termination", **runtime.agent_termination}}
        runtime.emit("agent_control_accepted", {"capability": "oeh_finish",
            "kernel_trace_id": ctx.trace_id, "kernel_exec_id": ctx.exec_id,
            "observation_id": runtime.observation_id,
            "observation_sha256": observation_digest(runtime.observation),
            "native_step": runtime.gate.step,
            "control_sha256": digest(envelope["oeh_episode_control"])})
        runtime.emit("agent_termination", runtime.agent_termination)
        runtime.gate.stop("agent_requested_termination")
        if runtime.ready is not None and not runtime.ready.done():
            runtime.ready.set_result(envelope)
        return {"accepted": True, "executed_steps": 0, "benchmark_outcome": "owned_by_evaluator"}

    async def act(self, params, ctx):
        runtime = self.runtime
        if runtime.cfg["mode"] in {"review", "review-stmem"}:
            return await self._act_review(params, ctx)
        hybrid = runtime.cfg["mode"] == "hybrid"
        if hybrid and (runtime.consumed or params.get("observation_id") != runtime.observation_id):
            # A late intent from a slow turn whose decision already went to the fast path: refused without
            # touching the current decision or the gate.
            runtime.emit("stale_intent_refused", {"trace_id": ctx.trace_id, "exec_id": ctx.exec_id,
                                                  "active": runtime.active})
            raise ValueError("intent has a stale or consumed observation")
        task = asyncio.current_task()
        runtime.executing = task
        ready = runtime.ready  # this decision's result; a later decision has its own
        slow = runtime.active == "slow"
        offered = False
        async def heartbeat():
            while True:
                ctx.heartbeat()
                await asyncio.sleep(.2)
        lease = asyncio.create_task(heartbeat())
        # Slow mode with brain.on_call_failure fill_once: an intent the method itself refuses (a rule the shown schema
        # cannot express, e.g. a phase transition that skips a checkpoint) is a refused intent like one kernel
        # admission refuses, so it takes the fill once instead of stopping the attempt (round-1 smoke, 2026-10-20).
        method_refused = late = False
        try:
            try:
                runtime.method.check_intent(params)
            except ValueError:
                method_refused = (slow and runtime.cfg["mode"] == "slow"
                                  and runtime.slow_settings.get("on_call_failure") == "fill_once")
                raise
            if runtime.consumed or params["observation_id"] != runtime.observation_id:
                raise ValueError("intent has a stale or consumed observation")
            runtime.consumed = True
            if ctx.cancelled.is_set() or runtime.gate.stopped:
                raise RuntimeError("intent cancelled")
            runtime.emit("policy_intent", {"params": params, "trace_id": ctx.trace_id,
                                           "exec_id": ctx.exec_id, "brain": "slow" if slow else "fast"})
            fill = runtime.fill_kind if runtime.cfg["mode"] == "slow" and not slow else None
            fill_source = runtime.fill_source if fill else None  # read now: the runtime clears it when the decision returns
            if params["mode"] == "vla":
                if slow:
                    obs = runtime.contract.subgoal_observation(runtime, params["subgoal"])
                    await asyncio.to_thread(runtime.backend.update_obs, obs)
                    runtime.backend_input_sha256 = observation_digest(obs)
                start = time.monotonic_ns()
                request_id = "vla:" + ctx.exec_id
                runtime.emit("model_call", {"model_id": runtime.cfg["model_id"],
                             "method": "vla_inference", "request_id": request_id,
                             "input_sha256": runtime.backend_input_sha256})
                try:
                    rows = await asyncio.to_thread(runtime.backend.get_action)
                except BaseException as exc:
                    runtime.emit("model_failure", {"request_id": request_id, "error": type(exc).__name__})
                    raise
                finally:
                    runtime.emit("model_timing", {"request_id": request_id,
                                 "duration_ms": (time.monotonic_ns() - start) / 1e6,
                                 **serving_timing(runtime.backend)})
                if not isinstance(rows, list) or not rows or len(rows) > runtime.cfg["max_chunk_steps"]:
                    raise ValueError("model chunk violates frozen horizon")
                try:
                    rows = [runtime.contract.validate_action(row, runtime.cfg["action_limits"]) for row in rows]
                except ValueError as exc:
                    if hybrid and not slow:
                        runtime.emit("fast_chunk_rejected", {"reason": str(exc), "request_id": request_id})
                        raise FastRefused(str(exc)) from None
                    raise
                runtime.emit("vla_full_proposal", {"actions": rows, "sha256": digest(rows)})
                if slow:
                    rows = rows[:params["execute_steps"]]
            else:
                # The one exception: slow mode's on_no_call hold, a one-step wait the harness asked the fast brain for.
                if not slow and not (fill == "hold" and params["mode"] == "wait" and params["execute_steps"] == 1):
                    raise ValueError("fast condition cannot select analytic primitives")
                try:
                    rows = runtime.method.primitives(params, runtime.observation["state"])
                except ValueError:
                    # A bound the primitive builder enforces (e.g. a relative move over 3 cm) refuses the intent like
                    # check_intent does: under fill_once it takes the fill instead of ending the attempt (round-2
                    # smoke, 2026-10-21: make_kong l36 R2-LLMFirst sealed unjudged on such a refusal).
                    method_refused = (slow and runtime.cfg["mode"] == "slow"
                                      and runtime.slow_settings.get("on_call_failure") == "fill_once")
                    raise
            if (ctx.cancelled.is_set() or self.disposed.is_set() or time.monotonic() >= runtime.decision_deadline
                    or params["observation_id"] != runtime.observation_id):
                late = (slow and runtime.cfg["mode"] == "slow"
                        and runtime.slow_settings.get("on_call_failure") == "fill_once")
                raise RuntimeError("intent expired during generation")
            # The decision deadline ends when the intent is accepted: slow rows run to the episode deadline
            # and native step limit, however long the model took (reviewer m3).
            done = runtime.gate.offer(rows, ctx, runtime.gate.deadline if slow else runtime.decision_deadline)
            offered = True
            ready.set_result(copy.deepcopy(list(rows)))
            result = await done  # lease is held until the native acknowledgements
            if slow:
                if params["phase"] != runtime.method.phase:
                    runtime.emit("phase_transition", {"from": runtime.method.phase, "to": params["phase"],
                                                      "native_step": runtime.gate.step})
                runtime.method.phase = params["phase"]
                runtime.method.target = params["target"]
            if runtime.cfg["mode"] == "slow":
                runtime.remember(params, result, "slow" if slow else f"{fill_source}:{fill}")
            runtime.previous_receipt = result
            return result  # only executed count, never native outcome or score
        except FastRefused as exc:
            # Nothing was offered to the native gate; the episode continues via escalation.
            if ready and not ready.done():
                ready.set_exception(FastRefused(str(exc)))
            raise
        except BaseException as exc:
            if (hybrid or method_refused or late) and slow and not offered:
                # Nothing reached the gate: only this slow decision failed; the fast path takes over.
                if isinstance(exc, ValueError):
                    runtime.emit("slow_intent_refused", {"source": "oeh_method", "reason": str(exc),
                                                         "trace_id": ctx.trace_id})
                if ready and not ready.done():
                    ready.set_exception(SlowFailed(f"intent refused: {exc}" if isinstance(exc, ValueError)
                                                   else f"intent failed: {type(exc).__name__}: {exc}"))
                raise
            runtime.gate.stop(type(exc).__name__)
            if ready and not ready.done():
                ready.set_exception(RuntimeError(type(exc).__name__))
            raise
        finally:
            runtime.executing = None
            lease.cancel()
            await asyncio.gather(lease, return_exceptions=True)

    async def _act_review(self, params, ctx):
        """Review mode: build the rows of one admitted decision and hand them to the gate.

        A teacher intent is taken only while the runtime is waiting for one (the window) and only for the current
        candidate; a harness intent (hold or fallback, proposed by the fast brain) is the one the runtime asked
        for. Everything is decided under the review lock, so a teacher decision and a hold never both reach the
        gate for one call of the official loop.
        """
        runtime = self.runtime
        state = runtime.review
        harness = runtime.fast_request is not None and digest(params) == digest(runtime.fast_request)
        runtime.executing = asyncio.current_task()
        offered = False
        async def heartbeat():
            while True:
                ctx.heartbeat()
                await asyncio.sleep(.2)
        lease = asyncio.create_task(heartbeat())
        try:
            runtime.method.check_intent(params)
            async with state.lock:
                packet = state.packet
                if (packet is None or params["candidate_id"] != state.candidate_id
                        or params["observation_id"] != state.observation_id):
                    raise ValueError("stale candidate: the decision does not match the current proposal")
                if harness:
                    ready = runtime.ready
                elif not state.window.is_set() or state.ready is None or state.ready.done():
                    raise ValueError(f"{WINDOW_CLOSED}: the harness is holding the robot for this call; "
                                     "the same decision is proposed at the next call")
                else:
                    ready = state.ready
                if ctx.cancelled.is_set() or runtime.gate.stopped or getattr(runtime, "closed", False):
                    raise RuntimeError("intent cancelled")
                if params["mode"] != "hold" and getattr(runtime, "context_recovery_pending", False):
                    raise ValueError("memory context recovery required before motion")
                if params["mode"] != "hold" and getattr(runtime, "task_tree", None) is not None:
                    # Recheck at the actual admission boundary, including delayed proposals.
                    if harness:
                        raise ValueError("automatic task-tree motion has no main-LLM leaf binding")
                    runtime.task_tree.check_binding(getattr(runtime, "task_action_binding", None))
                rows = runtime.method.rows(params, state.candidate, runtime.observation["state"])
                record = runtime.method.executed_rows(rows, runtime.observation["state"])
                clip = getattr(runtime.method, "clip", None)
                if clip:
                    kind = "hold_clipped" if clip["mode"] == "hold" else "start_clipped"
                    runtime.emit("review_harness_action", {"kind": kind, "mode": clip["mode"],
                                                           "request_id": packet["request_id"],
                                                           "reason": "measured pose outside the workspace bounds",
                                                           "arms": clip["arms"], "native_step": runtime.gate.step})
                runtime.emit("policy_intent", {"params": params, "trace_id": ctx.trace_id, "exec_id": ctx.exec_id,
                                               "brain": "fast" if harness else "teacher",
                                               "request_id": packet["request_id"]})
                done = runtime.gate.offer(rows, ctx, runtime.gate.deadline)
                offered = True
                if not harness:
                    state.window.clear()
                source = "teacher" if not harness else "auto" if packet.get("auto") else "harness"
                if params["mode"] != "hold":
                    # The decision is consumed: the next call of the official loop gets a fresh proposal.
                    state.finished[packet["request_id"]] = {"teacher": "executed", "harness": "fallback",
                                                            "auto": "auto"}[source]
                    state.packet = None
                ready.set_result(copy.deepcopy(list(rows)))
            result = await done  # lease is held until the native acknowledgements
            state.record(params, record, result["executed_steps"], source, packet)
            segment = getattr(runtime, "segment", None)  # a runtime stand-in (tests' SimpleNamespace) may lack it
            if segment is not None and source in ("teacher", "harness") and params["mode"] != "hold":
                # G13: the decision for this packet executed (the LLM component's, or the fallback): its segment begins.
                look = params.get("next_look")
                chunks = segment.start(runtime.gate.step, look["after_chunks"] if look else None)
                runtime.emit("segment_started", {"request_id": packet["request_id"], "native_step": runtime.gate.step,
                                                 "after_chunks": chunks, "named": look is not None,
                                                 "source": source})
            elif getattr(runtime, "takeover", None) is not None and source == "teacher" and params["mode"] != "hold":
                # A full chunk: every row of this candidate OpenWAM may run in one decision.
                full = min(len(state.candidate or ()), runtime.method.limits["student_max_steps"])
                why = runtime.takeover.teacher_executed(runtime.gate.step, state.previous, full)
                if why is not None:
                    runtime.emit("teacher_handback", {"request_id": packet["request_id"], "why": why,
                                                      "native_step": runtime.gate.step,
                                                      "teacher_decisions": runtime.takeover.decisions})
            runtime.previous_receipt = result
            return result  # only executed count, never native outcome or score
        except BaseException:
            if offered:
                runtime.gate.stop("review_execution_failed")
            raise
        finally:
            runtime.executing = None
            lease.cancel()
            await asyncio.gather(lease, return_exceptions=True)


def _report_refusal(reason, provider_error):
    """The evaluator's policy-fatal report with the provider's own answer (status, type, code; never its message):
    the evaluator, not this runtime, decides whether that answer is a provider refusal (infrastructure)."""
    try:
        from oeh_eval.runtime_config import report_policy_fatal
        report_policy_fatal(reason[:800], provider_error=provider_error)
    except Exception:  # noqa: BLE001 - the runtime error the caller raises carries the reason as well
        pass


class ReviewState:
    """One review episode's decision bookkeeping. Touched only on the runtime's event loop."""

    COUNTERS = ("student", "edit", "eef", "home", "hold", "fallback")

    def __init__(self, auto=False, arms=ARMS, describe=None, digest_max=None, interval_memory=None):
        self.arms = arms  # the review method's, for the previous-result records
        # How an executed decision reads in the next packet: the review method's ``describe_executed``, by default
        # review.executed_record for these arms.
        self.describe = describe
        self.decision = 0  # index of the next packet
        self.packet = self.images = None
        self.candidate = self.candidate_id = self.observation_id = None
        self.started = None
        self.window = asyncio.Event()  # set while a call of the official loop waits for a teacher decision
        self.lock = asyncio.Lock()
        self.ready = None  # rows for the waiting call
        self.finished = {}  # request_id -> executed | fallback | abandoned
        self.failed = {}  # request_id -> why the teacher gave up on it
        self.assessments = {}
        self.previous = None  # what the last decision executed, for the next packet
        self.holds = 0  # hold rows since the current packet
        self.counters = {f"{key}_steps": 0 for key in self.COUNTERS}
        self.counters["decisions"] = 0
        # OEH hybrid v4 (on_stall): OpenWAM's chunks that ran without the teacher, and those since its last packet.
        self.auto = {"chunks": 0, "steps": 0} if auto else None
        if auto:
            self.counters["auto_steps"] = 0
        # G5 (review.digest per_chunk): one entry per chunk OpenWAM ran alone since the last packet, the newest
        # digest_max kept; the entry of the pass-through chunk being executed waits in pending_digest.
        self.digest_max = digest_max
        self.interval_memory = interval_memory
        self.measurement_source = lambda: None
        self.digest = [] if digest_max is not None else None
        self.pending_digest = None
        self.receipts = []  # G13: the receipts of the pass-through chunks since the last packet (the segment report)
        self.waiting = set()  # request ids whose validated teacher decision waits for the window
        self.fatal = None  # the provider refused the key, model or quota: the attempt stops
        self.fatal_error = None  # that refusal as the provider answered it (status, type, code)
        self.shutdown = False

    def closed(self, request_id) -> bool:
        return (self.shutdown or request_id is None or request_id in self.finished or self.packet is None
                or self.packet["request_id"] != request_id)

    def give_up(self, request_id, reason):
        if request_id is not None and request_id not in self.finished:
            self.failed[request_id] = reason

    async def wait_window(self, request_id):
        while not self.closed(request_id) and not self.window.is_set():
            try:
                await asyncio.wait_for(self.window.wait(), 1.0)
            except asyncio.TimeoutError:
                continue

    def record(self, params, rows, executed, source, packet):
        mode = params["mode"]
        if mode == "hold":
            self.holds += 1
            self.counters["hold_steps"] += executed
            return
        if source == "auto":
            # OpenWAM alone (on_stall): not a decision of the teacher's, and nothing for it to assess.
            self.counters["auto_steps"] += executed
            self.auto["chunks"] += 1
            self.auto["steps"] += executed
            self.receipts.append({"request_id": packet["request_id"], "native_step": packet["native_step"],
                                  "requested_steps": params["steps"], "executed_steps": executed, "source": source})
            if self.digest is not None and self.pending_digest is not None:
                self.digest.append({**self.pending_digest, "executed_steps": executed,
                                    "gripper_ledger": self.ledger()})
                del self.digest[:-self.digest_max]
            if self.interval_memory is not None and self.pending_digest is not None:
                entry = {**self.pending_digest, "executed_steps": executed, "gripper_ledger": self.ledger()}
                measurement = self.measurement_source()
                if executed > 0 and measurement and measurement["native_step"] == entry["native_step"] + executed:
                    entry["measured_after"] = measurement["positions"]
                self.interval_memory.record(entry, request_id=packet["request_id"], requested_steps=params["steps"])
            self.pending_digest = None
            self.previous = self._described(None, params, rows, executed, source, None, None)
            self.previous["holds_before_execution"] = self.holds
            self.holds = 0
            return
        self.counters["fallback_steps" if source == "harness" else f"{mode}_steps"] += executed
        self.counters["decisions"] += 1
        # Only an executed teacher decision carries the teacher's assessment; a fallback carries none.
        info = self.assessments.pop(packet["request_id"], {})
        if source != "teacher":
            info = {}
        self.previous = self._described(packet["decision"], params, rows, executed, source, info.get("trigger"),
                                        info.get("assessment"))
        self.previous["holds_before_execution"] = self.holds
        self.holds = 0

    def ledger(self):
        """The gripper commands after the last executed row, as a packet shows them (eef_view's rounding)."""
        commands = self.ledger_source() if self.ledger_source is not None else {}
        return {arm: round(float(value), 3) for arm, value in (commands or {}).items()}

    ledger_source = None  # the runtime's: a callable returning its method's gripper ledger

    def _described(self, decision_index, params, rows, executed, source, trigger, assessment):
        if self.describe is not None:
            return self.describe(decision_index, params, rows, executed, source, trigger, assessment)
        return executed_record(decision_index, params, rows, executed, source, trigger, assessment, self.arms)


class FastBrain(Brain):
    """Deterministic harness dispatcher. No model, planner, or changed VLA instruction."""
    def __init__(self, runtime):
        super().__init__("fast")
        self.runtime = runtime

    async def on_turn(self, ctx):
        r = self.runtime
        if r.cfg["mode"] in {"review", "review-stmem"}:
            # The harness's own review-mode actions: one hold row while the teacher decides, or the teacher-less
            # fallback (the student's chunk), exactly as the runtime requested them.
            try:
                params = copy.deepcopy(r.fast_request)
                if params is not None:
                    outcome = await ctx.propose(CAPABILITY, params, timeout=r.decision_budget)
                    if outcome.get("outcome") != "executed" and r.ready and not r.ready.done():
                        r.ready.set_exception(RuntimeError("OEH refused or failed the harness action"))
                await ctx.done()
            except Exception:
                if r.ready and not r.ready.done():
                    r.ready.set_exception(RuntimeError("harness dispatch failed"))
                raise
            return
        try:
            # Phase and target are carried unchanged: the fast path never advances them.
            if r.cfg["mode"] == "slow" and r.fill_kind == "hold":
                # Slow mode's on_no_call hold (G4): one row at the measured pose with the ledger's gripper commands.
                params = {"observation_id": r.observation_id, "mode": "wait", "phase": r.method.phase,
                          "target": r.method.target, "execute_steps": 1}
            else:
                params = {
                    "observation_id": r.observation_id, "mode": "vla", "phase": r.method.phase,
                    "target": r.method.target if r.cfg["mode"] in ("hybrid", "slow") else "full task",
                    "execute_steps": r.cfg["max_chunk_steps"],
                    "subgoal": r.observation.get("instruction", r.observation.get("task_instruction")),
                }
            outcome = await ctx.propose(CAPABILITY, params, timeout=r.decision_budget)
            if outcome.get("outcome") != "executed" and r.ready and not r.ready.done():
                r.ready.set_exception(FastRefused("OEH refused or failed the fast action")
                                      if r.cfg["mode"] == "hybrid" else
                                      RuntimeError("OEH refused or failed the fast action"))
            await ctx.done()
        except Exception:
            if r.ready and not r.ready.done():
                r.ready.set_exception(RuntimeError("fast dispatch failed"))
            raise


class Runtime:
    def __init__(self, config, backend, *, attempt_id=None, contract=None, teacher_model=None, teacher_start=None):
        self.teacher_model_client = teacher_model  # review mode: a stand-in model client (tests); None = the API
        self.teacher_start = teacher_start  # review mode: a Codex app-server started with the policy server
        self.contract = contract if contract is not None else RoboDojoContract()
        self.cfg = self.contract.validate_config(config)
        self.backend = backend
        self.output = Path(self.cfg["output"]).resolve()
        if attempt_id:
            self.output = self.output / attempt_id / self.cfg["mode"]
        self.output.mkdir(parents=True, exist_ok=False)
        self.episode_id = uuid.uuid4().hex
        self.llm_budget=None
        if self.cfg.get("review",{}).get("llm_token_budget"):
            from .llm_budget import EpisodeLLMBudget
            self.llm_budget=EpisodeLLMBudget(self.output/"llm-token-budget.json",self.cfg["review"]["llm_token_budget"])
        self.spatial_observation_sequence=0
        self.latest_spatial_context=None
        self.events = []
        self.kernel_events = []
        self.drain_cursor = 0
        self.started = time.monotonic()
        self.gate = ExecutionGate(self.emit, native_step_limit=self.cfg["native_step_limit"],
                                  deadline=self.started + self.cfg["wall_seconds"])
        if any(key in (self.cfg.get("brain") or {}) for key in
               ("pose_control", "phase_policy", "context", "execution_feedback", "agent_termination", "notes_store")) and (
                self.cfg["mode"] != "slow" or self.contract.benchmark != "robodojo"):
            raise ValueError("Q1 reference settings are opt-in RoboDojo slow-mode settings")
        self.method = self.contract.make_method(self.cfg, backend)
        self.stmem = None
        if self.cfg["mode"] == "review-stmem":
            from .stmem_runtime import SpatialWorkerClient
            self.stmem = SpatialWorkerClient(self)
        self.context_recovery_pending = False
        self.task_tree = None  # Initialized with the first public instruction/TaskSpec packet.
        self.observation = None
        self.resets_repeated = 0
        self.observed_step = None
        self.observation_id = None
        self.backend_input_sha256 = None
        self.previous_receipt = None
        self.ready = None
        self.turn_ended = None
        self.executing = None
        self.consumed = True
        self.kernel = self.client = self.watcher = self.brain_client = self.plugin = self.fast = None
        self.kernel_log = self.short_socket = None
        self.closed = False
        self.sealed = False
        self.sim_ready_sent = False
        self.backend_connected = False
        self.cleanup_errors = []
        self.calls = 0
        self.slow_calls = 0
        self.active = "slow" if self.cfg["mode"] == "slow" else "fast"
        # The slow role's brain-row settings (G4/G6/G8; actions.check_slow_settings): absent, today's slow mode.
        self.slow_settings = check_slow_settings(self.cfg["brain"]) if self.cfg["mode"] == "slow" else {}
        self.slow_context = self.slow_context_identity = None
        if "context" in self.slow_settings:
            from .slow_reference import context_packet
            self.slow_context, self.slow_context_identity = context_packet(
                self.slow_settings["context"], self.cfg["brain"].get("task"))
        self.slow_feedback_pending = None
        self.agent_termination = None
        self.environment_terminal_seen = False
        self.fill_kind = None  # slow mode: the on_no_call fill being executed (fast_chunk | hold)
        self.fill_source = None  # ... and why: on_no_call (a silent turn) or on_call_failure (a failed one)
        self.fills = 0  # slow mode: decisions the harness filled after a silent turn (not LLM calls)
        self.decisions_made = []  # slow mode: every executed decision, for the payload's recent_decisions (G6)
        self.escalation = Escalation(self.cfg["escalation"]) if self.cfg["mode"] == "hybrid" else None
        self.slow_outcome = None  # executed | failed | no_action, for the return_to_fast record
        self.slow_error = self.slow_refusal = None  # why the current slow decision's model call or intent failed
        self.denied = None  # the escalation reason whose denial was already recorded (m6)
        self.current_request = None
        self.request_started = {}
        self.slow_request_timed = set()
        self.decision_deadline = self.started
        self.decision_budget = self.cfg["decision_seconds"]
        review_mode = self.cfg["mode"] in {"review", "review-stmem"}
        schedule = self.cfg["review"].get("schedule") if review_mode else None
        on_stall = schedule == "on_stall"
        segment = schedule == "segment"
        # G5 (review.digest per_chunk): the interval digest's bound; absent the key, DIGEST_MAX_CHUNKS.
        digest_max = (self.cfg["review"].get("digest_max_chunks", DIGEST_MAX_CHUNKS)
                      if review_mode and self.cfg["review"].get("digest") == "per_chunk" else None)
        interval_memory = None
        self.last_action_measurement = None
        if review_mode and self.cfg["review"].get("digest") == "event_summary":
            if self.contract.benchmark != "robodojo":
                raise ValueError("event_summary currently requires RoboDojo measured end-effector telemetry")
            from .interval_memory import IntervalMemory
            interval_memory = IntervalMemory(arms=self.method.arms)
        self.review = (ReviewState(auto=on_stall or segment, arms=self.method.arms,
                                   describe=getattr(self.method, "describe_executed", None), digest_max=digest_max,
                                   interval_memory=interval_memory)
                       if review_mode else None)
        if self.review is not None:
            self.review.ledger_source = lambda: self.method.gripper_commands
            self.review.measurement_source = lambda: self.last_action_measurement
        # OEH hybrid v4: the teacher is consulted only when OpenWAM stalls (oeh_policy.takeover). The segment schedule
        # (G13, oeh_policy.takeover.Segment) shares its detectors and pass-through path.
        if segment:
            if self.cfg["review"]["teacher"].get("channel", "python") != "python":
                raise ValueError("review.schedule segment runs on the python teacher channel on this commit (the "
                                 "brain lane's tool has no next_look)")
            self.takeover = Segment(self.cfg["review"]["on_stall"], self.cfg["review"]["segment"],
                                    self.cfg["review"]["control_hz"])
        else:
            self.takeover = (Takeover(self.cfg["review"]["on_stall"], self.cfg["review"]["control_hz"])
                             if on_stall else None)
        self.segment = self.takeover if segment else None
        # G7 (review.notes_store): the read-only store, loaded at its sealed sha256 before anything runs.
        notes_owner = self.cfg.get("brain", {}) if self.cfg["mode"] == "slow" else self.cfg.get("review", {})
        self.notes = self._load_notes() if "notes_store" in notes_owner else None
        # The review guard (oeh_policy.review_guard): only when review.guard is set, under either schedule.
        guard = self.cfg["review"].get("guard") if self.review is not None else None
        self.guard = (ReviewGuard(guard, self.method.limits["student_max_steps"],
                                  {arm: self.cfg["action_limits"][arm + "_xyz_bounds"] for arm in ("left", "right")})
                      if guard is not None else None)
        # review.guard.final_home_steps: the last native steps in which home passes the takeover gate untriggered.
        self.final_home_steps = self.guard.cfg["final_home_steps"] if self.guard is not None else 0
        self.fast_request = None  # review mode: the harness action the fast brain is asked to propose
        self.teacher = self.teacher_client = None
        # review.teacher.channel: where the teacher's model loop runs (python: TeacherBrain in this process; brain:
        # oeh-brain's `brain-teacher` lane in the kernel, answered by a TeacherBridge here).
        self.teacher_channel = (self.cfg["review"]["teacher"].get("channel", "python")
                                if self.cfg["mode"] in {"review", "review-stmem"} else None)
        if self.teacher_channel == "brain":
            from .llm_contract import has_v2_experience
            if interval_memory is not None or has_v2_experience(self):
                raise ValueError("reference experience/event_summary currently require the Python teacher channel")
        self.teacher_bridge = self.teacher_prestart = None
        self.teacher_relayed = self.teacher_relay_dir = self.teacher_adopt = None  # brain channel: the prestart
        self.teacher_accounts = None
        self.identity = {"schema": "oeh.policy.identity.v1", "configuration_sha256": digest(self.cfg),
                         "kernel_sha256": file_digest(Path(self.cfg["kernel"])),
                         "method_sha256": file_digest(self.contract.persona_path)
                         if self.contract.persona_path is not None else None,
                         "source_sha256": {p.name: file_digest(p) for p in Path(__file__).parent.glob("*.py")},
                         "checkpoint_sha256": self.cfg["checkpoint_sha256"],
                         "normalization_sha256": self.cfg["normalization_sha256"],
                         "mode": self.cfg["mode"], "model_id": self.cfg["model_id"],
                         "identity_files": self.cfg["identity_files"],
                         "action_contract": self.contract.action_contract,
                         "brain": self.cfg.get("brain"), "data_kind": self.cfg.get("data_kind", "simulation")}
        if self.review is not None:
            from .teacher import prompt_identity
            self.identity["review"] = {"teacher_prompt": self._prompt_identity(prompt_identity),
                                       "teacher_model": self.cfg["review"]["teacher"]["model"],
                                       "reasoning_effort": self.cfg["review"]["teacher"]["reasoning_effort"],
                                       "limits": self.method.limits}
            if self.takeover is not None:
                self.identity["review"]["schedule"] = schedule
            if self.segment is not None:
                self.identity["review"]["segment"] = {"max_segment_chunks": self.segment.max_chunks,
                                                      "default_after_chunks": self.segment.default_chunks}
            if digest_max is not None:
                self.identity["review"]["digest"] = {"mode": "per_chunk", "max_chunks": digest_max}
            if interval_memory is not None:
                self.identity["review"]["digest"] = {"mode": "event_summary", "schema": "oeh.interval-memory.v1",
                    "max_events": interval_memory.max_events, "max_recent": interval_memory.max_recent,
                    "motion_epsilon_m": interval_memory.motion_epsilon_m}
            if self.notes is not None:
                self.identity["review"]["notes_store"] = {"path": self.notes["path"], "sha256": self.notes["sha256"],
                                                          "count": self.notes["count"]}
            if self.guard is not None:
                self.identity["review"]["guard"] = dict(self.guard.cfg)
            # The declared context manifest and the Codex pin (docs/agent-icl-study/declared-factor-mode.md §1.4):
            # recorded only when the lock declares them, so a legacy identity is byte for byte what it was.
            if "context" in self.cfg["review"]:
                from .contracts import context_identity, packet_field_set
                self.identity["review"]["context"] = {**context_identity(self.cfg["review"]["context"]),
                                                      "field_set": packet_field_set(self.cfg["review"])}
            if "codex_version" in self.cfg["review"]["teacher"]:
                self.identity["review"]["teacher_codex_version"] = self.cfg["review"]["teacher"]["codex_version"]
        if self.slow_context_identity is not None:
            self.identity["slow_context"] = self.slow_context_identity
        if self.slow_settings.get("pose_control"):
            self.identity["slow_reference_prompt_sha256"] = digest(self._slow_persona())
        if self.cfg["mode"] == "slow" and self.notes is not None:
            self.identity["slow_notes_store"] = {"path": self.notes["path"], "sha256": self.notes["sha256"],
                                                  "count": self.notes["count"], "delivery": "each_observation"}
        if "design" in self.cfg:
            # The condition registry's entry (oeh_policy.mechanisms; docs/agent-icl-study/condition-registry.md),
            # sealed by the compiler at the config's top level: recorded only when declared.
            self.identity["design"] = copy.deepcopy(self.cfg["design"])
        (self.output / "identity.json").write_bytes(encoded(self.identity))
        (self.output / "configuration.json").write_bytes(encoded(self.cfg))

    def _prompt_identity(self, prompt_identity):
        """The teacher prompt's identity. The declared manifest is passed only when the lock declares one (absent, the
        call and its result are the frozen ones). Under the segment schedule the tool the teacher is given is the
        segment tool (review.tool_spec schedule segment, with next_look), so its hash is that tool's."""
        review = self.cfg["review"]
        args = (review.get("schedule"), self.contract.benchmark, self.method.arms, getattr(self.method, "gripper", True))
        identity = prompt_identity(*args, context=review["context"]) if "context" in review else prompt_identity(*args)
        from .llm_contract import has_v2_experience
        if review.get("digest") == "event_summary" or has_v2_experience(self) or review.get("exploration_probe"):
            from .teacher import runtime_prompt
            identity = copy.deepcopy(identity)
            identity["prompt_sha256"] = digest(runtime_prompt(self))
            identity["reference_contract_sha256"] = file_digest(Path(__file__).with_name("llm_interaction_contract.md"))
        return identity

    def _load_notes(self):
        """G7's read path: the notes store at the sha256 the lock sealed (oeh_policy.notes_store); a changed, missing
        or malformed store refuses the start."""
        from . import notes_store
        owner = "brain" if self.cfg["mode"] == "slow" else "review"
        cfg = self.cfg[owner]["notes_store"]
        if cfg["scope"] == "layout":
            raise ValueError("review.notes_store scope layout needs the case's layout, which the runtime never reads "
                             "(it is an oracle key of the packet boundary); use global or task")
        store = notes_store.load_store(cfg["path"], cfg["sha256"])
        if owner == "brain":
            marker = Path(cfg["path"]) / "store.json"
            if not marker.is_file() or json.loads(marker.read_text()) != {"schema": "oeh.notes_store.v2"}:
                raise ValueError("brain.notes_store requires a sealed v2 evidence store")
            if any(note.get("schema") != "oeh.task_note.v2" for note in store["notes"]):
                raise ValueError("brain.notes_store cannot mix legacy notes into a v2 experience store")
        return store

    def _task_notes(self):
        from . import notes_store
        owner = "brain" if self.cfg["mode"] == "slow" else "review"
        cfg = self.cfg[owner]["notes_store"]
        task = self.cfg[owner].get("task")
        if cfg["scope"] == "task" and not task:
            raise ValueError(f"{owner}.notes_store scope task needs {owner}.task")
        return notes_store.notes_for(self.notes, task, cfg["scope"])

    def emit(self, kind, payload):
        if self.sealed:
            return  # nothing is recorded after the seal
        row = {"kind": kind, "payload": plain(payload), "mono_ns": time.monotonic_ns()}
        self.events.append(row)
        with (self.output / "events.jsonl").open("ab") as stream:
            stream.write(encoded(row))

    def _event(self, event):
        self.kernel_events.append(event)
        with (self.output / "kernel-events.jsonl").open("ab") as stream:
            stream.write(encoded(event))
        kind, payload = event.get("kind"), event.get("payload", {})
        trace = event.get("trace_id")
        if (kind == "act/admitted" and self.slow_settings.get("agent_termination")
                and payload.get("capability") == "oeh_finish"):
            self.emit("control_kernel_admission", {"kernel_trace_id": trace,
                "kernel_exec_id": payload.get("proposal_id"), "capability": payload["capability"],
                "brain": payload.get("brain_id"), "source": "kernel_act_admitted"})
        if kind == "act/refused" and payload.get("brain_id") == "slow" and payload.get("stage") == "vocabulary":
            # The slow brain's intent failed the capability schema: a brain protocol failure, not a safety event.
            self.slow_refusal = payload.get("reason")
            self.emit("slow_intent_refused", {"source": "kernel_admission", "reason": payload.get("reason"),
                                              "trace_id": trace})
        elif kind and (kind in {"safety/estop", "act/refused", "act/revoked", "act/preempted"}
                       or kind.startswith("health/")):
            self.emit("safety_event", {"kernel_kind": kind, "trace_id": trace, "evidence": payload})
        if (kind == "brain/request-error" and self.active == "slow" and self.cfg.get("brain")
                and not (self.ready and self.ready.done())):
            # After the intent is accepted, the brain's step timer still spans the rows' execution; a
            # "step timeout: tools" then is not a model failure (the rows run to the episode deadline).
            self.slow_error = payload.get("reason")
            self.emit("model_failure", {"model_id": self.cfg["brain"]["model"], "request_id": self.current_request,
                                        "reason": payload.get("reason")})
        if kind == FROM_LANE and self.teacher_bridge is not None:
            # The brain-channel teacher's lane asking for state or validation, or a record TeacherBrain would make.
            self.teacher_bridge.on_event(payload)
        if kind == "brain/policy_request":
            request_id = f"{trace}:{payload['step']}"
            self.current_request = request_id
            self.request_started[request_id] = time.monotonic_ns()
            path = self.output / f"brain-request-{len(self.request_started):06d}.json"
            path.write_bytes(encoded(payload))
            self.emit("model_call", {"model_id": self.cfg["brain"]["model"], "request_id": request_id,
                      "input_sha256": digest(payload), "input_artifact": path.name,
                      "method": "internal_native_brain"})
        if kind == "brain/provider_attempt" and self.slow_settings.get("pose_control"):
            self.emit("provider_call", {**payload, "request_id": f"{trace}:{payload['step']}",
                                       "source": "provider_dispatch"})
        if kind == "brain/model_warning":
            # A provider notice attached to one call (Codex warning items: fallback model metadata, a model
            # reroute, config notices): recorded, not a failure (supervisor decision).
            self.emit("codex_warning", {"model_id": self.cfg["brain"]["model"],
                                        "request_id": f"{trace}:{payload['step']}", "message": payload["message"]})
        if kind == "brain/usage":
            request_id = f"{trace}:{payload['step']}"
            self.emit("usage", {"model_id": self.cfg["brain"]["model"], "request_id": request_id,
                      "input_tokens": payload["input_tokens"], "output_tokens": payload["output_tokens"],
                      "cached_input_tokens": payload["cached_input_tokens"],
                      "reasoning_output_tokens": payload.get("reasoning_output_tokens"),
                      "call_complete": payload.get("complete"),
                      "total_tokens": payload["input_tokens"] + payload["output_tokens"],
                      "cost_usd": None, "source": "provider_reported"})
            # codex exec 0.155.1 cannot cap output tokens: max_tokens is a measured budget (decision log).
            limit = self.cfg["brain"].get("max_tokens", 2048)
            if payload["output_tokens"] > limit:
                self.emit("budget_exceeded", {"kind": "max_tokens", "request_id": request_id,
                          "output_tokens": payload["output_tokens"],
                          "reasoning_tokens": payload.get("reasoning_output_tokens"),
                          "limit": limit, "enforced": False})
        if kind == "brain/model_output" and self.current_request:
            self.emit("model_timing", {"request_id": self.current_request,
                      "duration_ms": (time.monotonic_ns() - self.request_started[self.current_request]) / 1e6})
            self.slow_request_timed.add(self.current_request)
        if kind == "brain/turn_end" and self.slow_settings.get("pose_control"):
            for request_id, started in self.request_started.items():
                if request_id.startswith(f"{trace}:") and request_id not in self.slow_request_timed:
                    self.emit("model_timing", {"request_id": request_id,
                        "duration_ms": (time.monotonic_ns() - started) / 1e6,
                        "ended_at": "turn_end", "turn_end_reason": payload.get("reason")})
                    self.slow_request_timed.add(request_id)
        if (kind == "brain/turn_end" and self.turn_ended and not self.turn_ended.done()
                and payload.get("brain_id") in (None, self.active)):  # hybrid: the deciding brain's own turn
            self.turn_ended.set_result(event)
            if self.ready and not self.ready.done():
                slow_turn = self.cfg["mode"] == "slow" and self.active == "slow"
                silent = slow_turn and "on_no_call" in self.slow_settings and payload.get("reason") == "end_turn"
                # brain.on_call_failure fill_once: a failed call or a refused intent is handed to the fill too.
                failed = (slow_turn and self.slow_settings.get("on_call_failure") == "fill_once"
                          and payload.get("reason") != "end_turn")
                self.ready.set_exception(self._slow_turn_failure(payload.get("reason"))
                                         if self.cfg["mode"] == "hybrid" and self.active == "slow" or silent or failed
                                         else RuntimeError("brain turn ended without an admitted action"))

    def _slow_turn_failure(self, reason):
        """Why a hybrid slow turn ended before an intent reached the gate."""
        if reason == "end_turn":  # the documented no-intervention answer ({"tool": null} on the Codex wire)
            return SlowFailed("the slow brain answered with no intervention", no_action=True)
        if self.slow_refusal:
            return SlowFailed(f"intent refused by kernel admission: {self.slow_refusal}", turn_end=reason)
        if self.slow_error:
            return SlowFailed(f"model call failed: {self.slow_error}", turn_end=reason)
        return SlowFailed(f"brain turn ended without an admitted intent: {reason}", turn_end=reason)

    async def start(self):
        if self.stmem is not None:
            await self.stmem.start()
        if self.review is not None:
            # The key must read the frozen model before the episode starts: a refused key, a wrong model or an
            # unreachable API is a failed start, never a silent student-only run.
            from .teacher import TeacherUnavailable, make_client
            if self.teacher_channel == "brain":
                await self._brain_teacher_prestart()
            else:
                if self.teacher_model_client is None and self.teacher_start is None:
                    self.teacher_model_client = make_client(self.cfg["review"]["teacher"], self)
                if hasattr(self.teacher_model_client, "logs_dir"):
                    self.teacher_model_client.logs_dir = self.codex_logs
                try:
                    if self.teacher_start is not None:
                        # Started with the policy server (Model waits for it before reporting ready).
                        self.teacher_model_client, checked = await self._take_prestart(self.teacher_start)
                        self.teacher_model_client.workdir = self.output / "teacher"
                        self.teacher_model_client.workdir.mkdir(parents=True, exist_ok=True)
                    else:
                        checked = await asyncio.to_thread(self.teacher_model_client.preflight)
                except (TeacherUnavailable, OSError, ValueError) as exc:
                    raise self._teacher_preflight_failed(exc) from None
                self.emit("teacher_preflight", checked)
        if self.cfg["mode"] in {"slow", "hybrid"}:
            # Fail before the episode, not at the first escalation (reviewer m5).
            try:
                checked = await asyncio.to_thread(brain_preflight.check, self.cfg["brain"])
            except RuntimeError as exc:
                self.emit("brain_preflight_failed", {"reason": str(exc)})
                raise
            self.emit("brain_preflight", checked)
        self.short_socket = tempfile.TemporaryDirectory(prefix="oeh-policy-")
        socket = Path(self.short_socket.name) / "rpc.sock"
        turn_seconds = max(self.cfg["decision_seconds"], self.cfg.get("first_decision_seconds", 0))
        permitted_capabilities = [CAPABILITY]
        if self.slow_settings.get("agent_termination"):
            from .q1_actions import FINISH_CAPABILITY
            permitted_capabilities.append(FINISH_CAPABILITY)
        if self.review is not None:
            # A teacher turn spans its model calls, any hold rows and the execution of its own rows.
            turn_seconds = max(turn_seconds, self.cfg["review"]["turn_seconds"])
        rows = [{"id": "clock"}, {"id": "world-log", "config": {"dir": "./kernel-world", "retain": 32768}},
                {"id": "frame-store", "config": {"window_s": 2}}, {"id": "state-registry"},
                {"id": "robot-fsm"},
                {"id": "distill-estop", "config": {"flag_path": "./estop"}},
                {"id": "brains", "config": {"turn_timeout_ms": int(turn_seconds * 1000)}},
                {"id": "admission", "config": {"estop_flag": "./estop", "required_rules":
                    ["rule-whitelist", "rule-fsm", "rule-run-mode", "rule-estop"]}},
                {"id": "rule-whitelist", "config": {"allowed": permitted_capabilities}}, {"id": "rule-fsm"},
                {"id": "rule-run-mode", "config": {"mode": "active"}}, {"id": "rule-estop"},
                {"id": "approval-auto", "config": {"allow": False}},
                # The camera event carries three PNG views of up to 320x240 (~1 MiB of base64 for textured
                # frames); the kernel's 64 KiB default refused it at the first decision. 8 MiB is over 3x the
                # 2 MiB per-image cap the native brain enforces (oeh-brain model.rs ImageInput::validate).
                # execute_accept_ms bounds only the plugin's acknowledgement of an admitted intent, like the heartbeat
                # and lease above it tolerates a 30 s stall of this process's loop: at 5 s, a review attempt on the pod
                # (dev2 build_tower l5, 2026-09-29) lost an admitted OpenWAM chunk to "capability/execute: request
                # timed out" and the episode ended on "kernel is not ready for motion".
                {"id": "rpc-server", "config": {"uds": str(socket), "ws": None,
                    "peer_uids": [os.geteuid()], "cancel_grace_ms": 5000, "execute_accept_ms": EXECUTE_ACCEPT_MS,
                    "max_event_bytes": MAX_EVENT_BYTES}}]
        if self.contract.benchmark == "robodojo":
            rows.append({"id": "distill-arx-ledger"})
        else:
            rows.append({"id": "distill-sim-ready", "config": {"benchmark": self.contract.benchmark}})
        wake = {"default": "ignore", "rules": {"obs/robot_rgb": "wake", "safety/estop": "interrupt"},
                "budget": {"max_consecutive_wakes": 10000, "min_interval_ms": 0}}
        fast_wake = wake
        if self.cfg["mode"] in {"hybrid", "review", "review-stmem"} or self._slow_fills():
            # Fast decisions wake only the fast dispatcher; camera frames (hybrid, slow) or teacher packets (review)
            # wake only the model brain.
            fast_wake = {**wake, "rules": {FAST_TICK: "wake", "safety/estop": "interrupt"}}
        if self.cfg["mode"] in {"slow", "hybrid"}:
            brain = self.cfg["brain"]
            wake_path = self.output / "wake.json"
            wake_path.write_bytes(encoded(wake))
            # G4: the slow role's row settings come from cfg["brain"] when the design froze them there; absent, the
            # literals the hand-written slow and hybrid configs ran with (the bundle is then byte for byte as before).
            rows.append({"id": "brain-native", "config": {
                "brain_id": "slow", "wake": str(wake_path),
                "persona": self._slow_persona(),
                "isolated_camera_turns": brain.get("isolated_camera_turns", True),
                **({"record_policy_requests": True, "require_complete_observation": True}
                   if self.slow_settings.get("pose_control") else {}),
                "allowed_tools": ["robot_" + name for name in permitted_capabilities],
                "model": {"provider": brain["provider"], "model": brain["model"]},
                "providers": brain["providers"], "max_steps": brain.get("max_steps", 1),
                "max_tokens": brain.get("max_tokens", 2048),
                "step_timeout_ms": int(self.cfg["decision_seconds"] * 1000),
                "oeh_home": str(self.output / "empty-knowledge"), "robot_id": self.contract.robot_id,
                "protected_paths": [str(self.output), self.cfg["repository"]],
                "review": {"every_turns": 0, "on_session_end": False, "max_steps": 1, "timeout_ms": 1000},
                "retry": {"max_attempts": brain.get("retry_max_attempts", 1), "base_ms": 100, "cap_ms": 100}}})
        if self.teacher_bridge is not None:
            # The brain-channel teacher: its lane's row is the bundle's last, after the services it injects.
            rows.append(self.teacher_bridge.row(self.teacher_accounts))
        bundle = self.output / "bundles"
        bundle.mkdir()
        (bundle / "policy.yaml").write_bytes(encoded(rows))
        profile = self.output / "profile.yaml"
        profile.write_bytes(encoded({"bundles": ["policy"], "patch": []}))
        self.kernel_log = (self.output / "kernel.log").open("w", encoding="utf-8")
        # The brain-channel teacher starts its app-server from the kernel: the kernel gets this process's
        # os.environ, which is what CodexClient gives the app-server (the C-level environment can hold more).
        environment = dict(os.environ) if self.teacher_bridge is not None else None
        self.kernel = await asyncio.create_subprocess_exec(self.cfg["kernel"], "run", "--profile", str(profile),
                         "--bundles-dir", str(bundle), cwd=self.output, env=environment,
                         stdout=self.kernel_log, stderr=self.kernel_log)
        for _ in range(200):
            if socket.exists():
                break
            if self.kernel.returncode is not None:
                raise RuntimeError("kernel startup failed; see kernel.log")
            await asyncio.sleep(.025)
        else:
            raise RuntimeError("kernel socket did not appear")
        self.watcher = await Client.connect_uds(socket, default_timeout=10)
        self.watcher.on_notification("event", self._event)
        await self.watcher.call("log/tail", {"prefixes": ["brain", "act", "safety", "health"]})
        self.client = await Client.connect_uds(socket, default_timeout=self.cfg["decision_seconds"])
        if self.teacher_bridge is not None:
            await self._brain_teacher_preflight()
        self.plugin = MotionPlugin(self)
        await self.plugin.register(self.client)
        if self.cfg["mode"] in {"fast", "hybrid", "review", "review-stmem"} or self._slow_fills():
            self.brain_client = await Client.connect_uds(socket, default_timeout=self.cfg["decision_seconds"])
            self.fast = FastBrain(self)
            self.fast.wake_policy = fast_wake
            await self.fast.attach(self.brain_client)
        if self.teacher_bridge is not None:
            self.teacher = self.teacher_bridge  # the lane attached itself as brain `teacher` with the kernel
        elif self.review is not None:
            from .teacher import TeacherBrain
            self.teacher_client = await Client.connect_uds(socket, default_timeout=self.cfg["review"]["turn_seconds"])
            self.teacher = TeacherBrain(self, client=self.teacher_model_client)
            await self.teacher.attach(self.teacher_client)
        await asyncio.to_thread(self.backend.reset)
        self.backend_connected = True
        self.emit("policy_identity", self.identity)
        return self.identity

    def _teacher_preflight_failed(self, exc):
        """Record and report a teacher preflight that failed; the RuntimeError the start raises."""
        from .teacher import PREFLIGHT_UNREACHABLE, TeacherFatal, TeacherUnavailable
        self.emit("teacher_preflight_failed", {"reason": str(exc)[:800]})
        if isinstance(exc, TeacherFatal) and exc.errors:
            # Reported before anything raises: the first policy-fatal report is the one the evaluator keeps.
            _report_refusal(f"teacher preflight refused: {exc}", exc.errors[-1])
        elif isinstance(exc, TeacherUnavailable) and not isinstance(exc, TeacherFatal):
            # Unreachable before the episode's first step: the operator's infrastructure, not this arm's
            # failure (an outage during the episode stays the policy's to hold or fall back through).
            _report_refusal(f"teacher unreachable at preflight: {exc}",
                            {"status": None, "type": "teacher_unavailable", "code": PREFLIGHT_UNREACHABLE})
        return RuntimeError(f"teacher preflight failed: {exc}")

    @property
    def codex_logs(self):
        """Where a Codex client keeps its app-server's stderr at close (diagnosis only; codex_teacher.keep_logs)."""
        return self.output / "codex-app-server"

    async def _take_prestart(self, start):
        """The prestart's (client, preflight result); a failed one keeps its app-server's stderr in the attempt."""
        try:
            client, checked = await asyncio.wrap_future(start)
        except BaseException as exc:
            failed = getattr(exc, "codex_client", None)
            if failed is not None:
                failed.logs_dir = self.codex_logs
                failed.close()
            raise
        client.logs_dir = self.codex_logs
        return client, checked

    async def _brain_teacher_prestart(self):
        """The brain channel before the kernel starts: the prestarted app-server's result, then the lane's inputs.

        TIME-1: the policy server starts the app-server and runs its preflight while OpenWAM loads (model.py), and
        this first reset takes that result, reporting a refusal or an unreachable teacher exactly as the python
        channel does, before the kernel starts. The lane then adopts that same process instead of starting its own
        inside this reset's call bound: the prestarted client hands it over on a Unix socket (``Rpc.hand_over``),
        and the lane's first "app-server" is a relay onto it, with the prestart's preflight result, account, rate
        limits and next request id, so it carries on where the python channel's client would (thread/start at the
        first packet). Only a later start (the app-server died, or a login switch) is the lane's own."""
        from .codex_teacher import read_accounts
        from .teacher import TeacherUnavailable
        from .teacher_bridge import TeacherBridge
        if self.teacher_start is not None:
            start, self.teacher_start = self.teacher_start, None
            try:
                prestarted, self.teacher_prestart = await self._take_prestart(start)
            except (TeacherUnavailable, OSError, ValueError) as exc:
                raise self._teacher_preflight_failed(exc) from None
            self.teacher_relayed = prestarted
            # A short path: a Unix socket's is limited to ~100 bytes, the attempt directory's can be longer.
            self.teacher_relay_dir = tempfile.TemporaryDirectory(prefix="oeh-relay-")
            preflight = {k: v for k, v in self.teacher_prestart.items() if k not in ("prestarted", "start_seconds")}
            self.teacher_adopt = {**prestarted.hand_over(str(Path(self.teacher_relay_dir.name) / "codex.sock")),
                                  "preflight": preflight}
        # Read where make_client reads them for the python channel: a bad list fails the start the same way.
        self.teacher_accounts = read_accounts(self.cfg["review"]["teacher"]["codex_accounts_file"])
        self.teacher_bridge = TeacherBridge(self)

    async def _brain_teacher_preflight(self):
        """The lane's preflight, reported as the python channel reports its client's."""
        from .teacher import TeacherUnavailable
        try:
            checked = await self.teacher_bridge.preflight()
        except (TeacherUnavailable, OSError, ValueError) as exc:
            raise self._teacher_preflight_failed(exc) from None
        if self.teacher_prestart is not None:
            checked = {**checked, "prestarted": True, "start_seconds": self.teacher_prestart.get("start_seconds")}
        self.emit("teacher_preflight", checked)

    async def repeat_reset(self):
        """The official loop's second reset before the episode's first observation (see Model.reset): repeat the
        backend reset, as the direct condition's OpenWAM Model receives it, and keep this episode."""
        if self.observation is not None or self.calls or self.gate.stopped:
            raise RuntimeError("one episode per policy server; start a fresh server for the next attempt")
        await asyncio.to_thread(self.backend.reset)
        self.resets_repeated += 1
        self.emit("policy_reset_repeated", {"count": self.resets_repeated, "native_step": self.gate.step})
        return self.identity

    def check_llm_budget(self):
        budget=getattr(self,"llm_budget",None)
        if budget is None:return
        from .llm_budget import LLMTokenBudgetExceeded
        try:budget.check()
        except LLMTokenBudgetExceeded:
            if not self.gate.stopped:
                self.emit("llm_token_budget_stop",budget.status())
                self.gate.stop("llm_token_budget_exceeded")
                _report_refusal("episode stopped at LLM token budget",{"code":"llm_token_budget_exceeded"})
            raise RuntimeError("episode terminated: LLM token budget exceeded")

    async def update_obs(self, obs):
        Runtime.check_llm_budget(self)
        if self.gate.stopped:
            raise RuntimeError("episode stopped")
        projected = self.contract.project_observation(obs)
        if self.stmem is not None:
            # Explicit optional calibration boundary. Never forward native depth,
            # segmentation, object state or reward diagnostics to either model.
            from .stmem_calibration import project_calibration
            camera = self.cfg["stmem"]["camera"]
            self.stmem_camera_metadata = {}  # never reuse geometry from a previous frame
            fresh_metadata={}
            for view_name in ('cam_head','cam_left_wrist','cam_right_wrist'):
                view=obs.get('vision',{}).get(view_name,{})
                if view.get('stmem_calibration') is not None:
                    fresh_metadata[view_name]=project_calibration(view)
            if camera not in fresh_metadata and (self.stmem.calibration[camera].get('source')=='observation' or camera!='cam_head'):
                raise ValueError(f'review-stmem update_obs: missing same-observation calibration for {camera}')
            self.stmem_camera_metadata=fresh_metadata
        self.observation = projected
        if self.stmem is not None:
            # Every native observation, including frames within an admitted WAM
            # chunk, reaches perception/memory independently of the decision owner.
            self.latest_spatial_context=await self.stmem.observe(self._packet_sim_time())
            Runtime.check_llm_budget(self)
            if self.latest_spatial_context.get("status")=="stale_worker_unavailable":
                self.gate.stop("per_frame_perception_unavailable")
                raise RuntimeError("per-frame spatial perception unavailable")
            self.spatial_observation_sequence=getattr(self,"spatial_observation_sequence",0)+1
            self.emit("spatial_frame_committed",{"sequence":self.spatial_observation_sequence,
                "native_step":self.gate.step,"time":self._packet_sim_time(),"trigger":"native_update_obs",
                "entity_count":len(self.latest_spatial_context.get("entities",[]))})
        self.observed_step = self.gate.step
        self.method.seed(projected["state"])
        # Preserve the official adapter's update cadence, including mid-chunk calls.
        model_input = self.contract.model_observation(self, projected)
        await asyncio.to_thread(self.backend.update_obs, copy.deepcopy(model_input))
        self.backend_input_sha256 = observation_digest(model_input)
        self.emit("policy_input", {"sha256": observation_digest(projected), "native_step": self.gate.step})
        ledger = self.contract.ledger_payload(self, projected)
        if ledger is not None:
            await self.client.call("event/emit", {"kind": self.contract.ledger_kind, "payload": ledger})
        if self.contract.benchmark != "robodojo" and not self.sim_ready_sent:
            await self.client.call("event/emit", {"kind": "obs/sim_policy_ready", "payload": {
                "benchmark": self.contract.benchmark, "episode_id": self.episode_id,
                "observation_sha256": self.backend_input_sha256,
                "native_reset_observed": True, "policy_connected": self.backend_connected}})
            self.sim_ready_sent = True

    def camera_payload(self):
        return self.contract.camera_payload(self)

    def _robodojo_camera_payload(self):
        images = []
        for name in CAMERAS:
            image = Image.fromarray(self.observation["vision"][name]["color"])
            image.thumbnail((320, 240), Image.Resampling.BOX)
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            images.append({"camera": name, "data_url": "data:image/png;base64," +
                           base64.b64encode(stream.getvalue()).decode()})
        return {"observation_id": self.observation_id, "observation_seq": self.gate.step, "epoch": 0,
                "task_instruction": self.observation.get("instruction", self.observation.get("task_instruction")),
                "robot_state": {**plain(self.observation["state"]),
                    **{arm + "_gripper_command": [value] for arm, value in self.method.gripper_commands.items()}},
                "phase": self.method.phase, "target": self.method.target, "images": images,
                "previous_receipt": self.previous_receipt}

    async def get_action(self):
        Runtime.check_llm_budget(self)
        if self.agent_termination is not None:
            return {"oeh_episode_control": {"kind": "agent_termination", **copy.deepcopy(self.agent_termination)}}
        if self.observation is None or self.gate.pending is not None or self.gate.stopped:
            raise RuntimeError("observation or execution lifecycle is not ready")
        if self.turn_ended is not None:
            await asyncio.wait_for(asyncio.shield(self.turn_ended), self.cfg["decision_seconds"])
        if time.monotonic() >= self.gate.deadline or self.gate.step >= self.cfg["native_step_limit"]:
            self.gate.stop("budget_exhausted")
            raise RuntimeError("episode budget exhausted")
        if self.review is not None:
            return await self._review_get_action()
        if (self.cfg["mode"] == "slow" and self.cfg["brain"]["max_calls"] > 0
                and self.calls - self.fills >= self.cfg["brain"]["max_calls"]):
            self.gate.stop("model_call_budget")
            raise RuntimeError("model call budget exhausted")
        if self.cfg["mode"] == "slow":
            try:
                return await self._decide("slow", None)
            except SlowFailed as failure:
                # brain.on_no_call: a silent turn (no_action); brain.on_call_failure fill_once: a failed call or a
                # refused intent. Either takes the on_no_call fill.
                if failure.kind == "slow_decision_no_action":
                    return await self._slow_no_call(failure)
                return await self._slow_call_failure(failure)
        if self.cfg["mode"] != "hybrid":
            return await self._decide(self.active, None)
        reason, suppressed = self.escalation.decide(self.gate.step)
        if suppressed:
            self.emit("escalation_suppressed", {"reason": suppressed, "native_step": self.gate.step})
        denied = reason if reason and self.slow_calls >= self.cfg["brain"]["max_calls"] else None
        if denied and denied != self.denied:
            # Once per denial episode: a stall that persists across decisions is one denial (m6).
            self.emit("escalation_denied", {"reason": reason, "cause": "model_call_budget",
                                            "native_step": self.gate.step})
        self.denied = denied
        if denied:
            if reason == "gate_rejection":
                self.gate.stop("fast_rejected_without_escalation_budget")
                raise RuntimeError("fast chunk refused and escalation budget exhausted")
            reason = None
        if reason is None:
            try:
                return await self._decide("fast", None)
            except FastRefused:
                self.escalation.rejected()
                if self.turn_ended is not None:
                    await asyncio.wait_for(asyncio.shield(self.turn_ended), self.cfg["decision_seconds"])
                return await self.get_action()
        try:
            return await self._decide("slow", reason)
        except SlowFailed as failure:
            return await self._fall_back(failure)

    # -- review mode ------------------------------------------------------------------------------------------------

    async def _review_get_action(self):
        """One call of the official loop in review mode: rows of the teacher's admitted decision for the current
        packet, or one admitted hold row when the decision is not ready within the call budget, or the student's
        chunk as the teacher-less fallback. Every branch goes through kernel admission and the gate."""
        state, review = self.review, self.cfg["review"]
        call_started = time.monotonic()
        if state.packet is None:
            limit = review.get("max_decisions", 0)
            if limit and state.decision >= limit and self.segment is None:
                self.gate.stop("decision_budget")
                raise RuntimeError("review decision budget exhausted")
            if self.takeover is None:
                await self._review_packet()
            else:
                rows = await (self._on_stall_boundary() if self.segment is None else self._segment_boundary())
                if rows is not None:
                    return rows  # OpenWAM's own chunk, or a hold for an empty one
        request_id = state.packet["request_id"]
        self._review_fatal_check()
        if request_id in state.failed:
            return await self._review_harness("fallback", f"teacher failed: {state.failed[request_id]}")
        if time.monotonic() - state.started >= review["teacher_max_wait_s"] and request_id not in state.waiting:
            state.give_up(request_id, "teacher_max_wait_s")
            return await self._review_harness("fallback", "teacher exceeded teacher_max_wait_s")
        if not state.candidate and state.holds >= review["empty_candidate_holds"]:
            # No admissible student rows and still no decision: drop this packet for a fresh proposal.
            state.finished[request_id] = "abandoned"
            self.emit("review_packet_abandoned", {"request_id": request_id, "reason": "empty candidate"})
            state.packet = None
            return await self.get_action()
        loop = asyncio.get_running_loop()
        if self.executing is not None and not self.executing.done():
            # The previous action's plugin call is still finishing (its lease): never let the teacher preempt it.
            await asyncio.wait({self.executing}, timeout=5)
        async with state.lock:
            state.ready = loop.create_future()
            state.window.set()
        deadline = call_started + review["call_seconds"] - review["hold_reserve_s"]
        rows = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or request_id in state.failed:
                break
            try:
                rows = await asyncio.wait_for(asyncio.shield(state.ready), min(remaining, .5))
                break
            except asyncio.TimeoutError:
                continue
        async with state.lock:
            if rows is None and state.ready.done() and not state.ready.cancelled() and state.ready.exception() is None:
                rows = state.ready.result()
            state.window.clear()
            state.ready = None
        if rows is not None:
            self.calls += 1
            return rows
        self._review_fatal_check()
        if request_id in state.failed:
            return await self._review_harness("fallback", f"teacher failed: {state.failed[request_id]}")
        return await self._review_harness("hold", "teacher decision not ready within the call budget")

    def _review_fatal_check(self):
        Runtime.check_llm_budget(self)
        if self.review.fatal is None:
            return
        self.gate.stop("teacher_fatal")
        _report_refusal(f"review teacher refused: {self.review.fatal}", self.review.fatal_error)
        raise RuntimeError(f"review teacher refused: {self.review.fatal}")

    async def _review_packet(self):
        """Inference/decision uses memory already updated by native update_obs.

        Chunk boundaries never initiate perception or duplicate a frame sample.
        """
        Runtime.check_llm_budget(self)
        request_id,truncated=await self._review_candidate()
        await self._review_wake(request_id,truncated)

    def _packet_sim_time(self):
        """The packet's sim_time_s: native steps at the review control rate."""
        return round(self.gate.step * (1.0 / self.cfg["review"]["control_hz"]), 2)

    async def _on_stall_boundary(self):
        """A chunk boundary under OEH hybrid v4's on_stall schedule (oeh_policy.takeover). OpenWAM infers a fresh
        chunk; while it runs alone that chunk executes unchanged (rows returned), unless the teacher takes over now,
        in which case the same chunk becomes the teacher's candidate and its packet is out (None returned). While the
        teacher has control every boundary is a teacher packet, as in the review mode."""
        state, takeover = self.review, self.takeover
        if self.executing is not None and not self.executing.done():
            # The previous action's acknowledgements (and a teacher's hand-back) settle before the boundary.
            await asyncio.wait({self.executing}, timeout=5)
        if takeover.engaged and takeover.spent:
            why = takeover.hand_back(self.gate.step, "teacher_decisions_spent")
            self.emit("teacher_handback", {"request_id": None, "why": why, "native_step": self.gate.step,
                                           "teacher_decisions": takeover.decisions})
        request_id, truncated = await self._review_candidate()
        if not takeover.engaged:
            reason = takeover.decide(self.gate.step, rejected=truncated is not None)
            if reason is None:
                return await self._pass_through(request_id, "OpenWAM runs alone while it progresses")
            note = takeover.engage(self.gate.step, reason)
            self.emit("teacher_takeover", note)
        await self._review_wake(request_id, truncated)
        return None

    async def _pass_through(self, request_id, reason):
        """OpenWAM's chunk at this boundary executes whole, unreviewed, through the fast brain (source auto). Under
        review.digest per_chunk the packet this boundary would have had is measured first, for the interval digest."""
        state = self.review
        state.packet = {"request_id": request_id, "decision": None, "observation_id": state.observation_id,
                        "candidate_id": state.candidate_id, "native_step": self.gate.step, "auto": True}
        state.images, state.started = None, time.monotonic()
        state.pending_digest = self._digest_entry() if state.digest is not None or state.interval_memory is not None else None
        return await self._review_harness("auto", reason)

    def _digest_entry(self):
        """G5: what a packet at this boundary would have shown of OpenWAM's chunk (the review method's packet_state,
        its student_proposal), reduced to the chunk's row count and per-arm diagnostics."""
        state = self.review
        if not state.candidate:
            return None
        view = self.method.packet_state(state.candidate, self.observation["state"], None)["student_proposal"]
        entry = {"native_step": self.gate.step, "rows": view["rows"], "diagnostics": view["diagnostics"]}
        if state.interval_memory is not None:
            entry["measured_before"] = {arm: self.observation["state"][arm + "_ee_pose"][:3]
                                         for arm in self.method.arms}
        return plain(entry)

    async def _segment_boundary(self):
        """A chunk boundary under the segment schedule (G13, oeh_policy.takeover.Segment). OpenWAM infers a fresh
        chunk; inside a segment it executes whole (rows returned) unless a detector wakes the LLM component early; at
        decision 0 and when the segment has passed, the same chunk is the candidate of a packet (None returned). The
        call budget review.max_decisions is checked when a packet is due: a spent budget stops the attempt there."""
        state, segment = self.review, self.segment
        if self.executing is not None and not self.executing.done():
            await asyncio.wait({self.executing}, timeout=5)
        request_id, truncated = await self._review_candidate()
        reason = segment.boundary(self.gate.step, rejected=truncated is not None)
        if reason is None:
            return await self._pass_through(request_id, "OpenWAM runs alone inside the segment")
        limit = self.cfg["review"].get("max_decisions", 0)
        if limit and state.decision >= limit:
            self.gate.stop("decision_budget")
            raise RuntimeError("review decision budget exhausted")
        note = segment.engage(self.gate.step, reason)
        self.emit("segment_wake", {**note, "evidence": segment.evidence})
        await self._review_wake(request_id, truncated)
        return None

    async def _review_candidate(self):
        """OpenWAM's fresh chunk on the current observation, validated row by row and bound to it as the candidate.
        Returns the decision's request id and the truncation (None when every row passed)."""
        state = self.review
        request_id = uuid.uuid4().hex
        vla_request = "vla:" + request_id
        self.emit("model_call", {"model_id": self.cfg["model_id"], "method": "vla_inference",
                                 "request_id": vla_request, "input_sha256": self.backend_input_sha256})
        start = time.monotonic_ns()
        # The former shared client waited up to 600 s, past Model's 105 s
        # caller bound. Reserve the hold time within this call; review-stmem's
        # perception runs alongside this inference (_review_packet), not after it.
        review = self.cfg["review"]
        inference_seconds = max(.001, review["call_seconds"] - review["hold_reserve_s"])
        shared = getattr(self.backend, "set_request_timeout", None)
        if shared is not None:
            shared(inference_seconds)
        try:
            rows = await asyncio.wait_for(asyncio.to_thread(self.contract.review_chunk, self.backend),
                                          inference_seconds)
        except BaseException as exc:
            from .openwam_protocol import SharedTransportError
            if shared is not None and isinstance(exc, (TimeoutError, asyncio.CancelledError)):
                abort = getattr(self.backend, "abort_inference", None)
                if abort is not None:
                    # A blocked send can hold the client's timing lock. Release
                    # the socket before the finally block drains that timing.
                    try:
                        abort()
                    except Exception as abort_error:
                        self.emit("shared_openwam_abort_error", {"error": type(abort_error).__name__})
            if isinstance(exc, SharedTransportError) or (shared is not None and isinstance(exc, TimeoutError)):
                self.emit("shared_openwam_failure", {"request_id": vla_request,
                          "error": type(exc).__name__, "timeout_s": inference_seconds,
                          "replayed": False})
            self.emit("model_failure", {"request_id": vla_request, "error": type(exc).__name__})
            self.gate.stop("vla_failed")
            raise
        finally:
            self.emit("model_timing", {"request_id": vla_request, "duration_ms": (time.monotonic_ns() - start) / 1e6,
                                       **serving_timing(self.backend)})
        if not isinstance(rows, list) or not rows or len(rows) > self.cfg["max_chunk_steps"]:
            self.gate.stop("vla_chunk_invalid")
            raise ValueError("model chunk violates frozen horizon")
        self.emit("vla_full_proposal", {"actions": rows, "sha256": digest(rows), "request_id": vla_request})
        valid, truncated = [], None
        for index, row in enumerate(rows):
            try:
                valid.append(self.contract.validate_action(row, self.cfg["action_limits"]))
            except ValueError as exc:
                truncated = {"valid_rows": index, "of": len(rows), "reason": str(exc)}
                self.emit("candidate_truncated", {"request_id": request_id, **truncated})
                break
        state.candidate = valid
        state.candidate_id = uuid.uuid4().hex
        state.observation_id = self.observation_id = uuid.uuid4().hex
        state.holds = 0
        return request_id, truncated

    async def _review_wake(self, request_id, truncated, spatial=None):
        """Show the current candidate and last committed memory; never trigger perception here."""
        state = self.review
        valid = state.candidate
        observed = self.observation["state"]
        if self.stmem is not None and getattr(self.stmem,'latest_scene',None):
            scene=self.stmem.scene_frames.query(frame_ids=[self.stmem.latest_scene['frame_id']],k=self.cfg['review']['teacher'].get('image_downsample_k',4),limit=3)
            images,meta=scene.pop('_image_items',[]),scene['images']
        else:
            images, meta = self._review_images(state.decision)
        packet = {"request_id": request_id, "decision": state.decision, "observation_id": state.observation_id,
                  "candidate_id": state.candidate_id, "native_step": self.gate.step,
                  "max_episode_steps": self.cfg["native_step_limit"],
                  "remaining_steps": self.cfg["native_step_limit"] - self.gate.step,
                  "sim_time_s": self._packet_sim_time(),
                  "instruction": self.observation.get("instruction", self.observation.get("task_instruction")),
                  **self.method.packet_state(valid, observed, truncated),
                  "previous_result": state.previous, "counters": dict(state.counters),
                  "task_started": any(v > 0 for k,v in state.counters.items() if k.endswith("_steps") and k != "hold_steps"),
                  "images": meta}
        if self.cfg["review"].get("exploration_probe"):
            packet["exploration_probe"] = copy.deepcopy(self.cfg["review"]["exploration_probe"])
        if self.stmem is not None and getattr(self.stmem,'latest_scene',None):
            packet['scene_frame']={'frame_id':self.stmem.latest_scene['frame_id'],'native_step':self.stmem.latest_scene['native_step'],
                'sim_time':self.stmem.latest_scene['sim_time'],'images':meta}
        if state.decision == 0:
            from .teacher import task_context
            if "context" in self.cfg["review"]:
                context = task_context(self.cfg["review"].get("task"), self.contract.benchmark,
                                       context=self.cfg["review"]["context"])
            else:
                context = task_context(self.cfg["review"].get("task"), self.contract.benchmark)
            if context is not None:
                packet["task_context"] = context
            if self.notes is not None:
                # G7: the store's notes for this task (scope review.notes_store.scope), on the first packet only.
                packet["task_notes"] = self._task_notes()
        manifest = self.cfg["review"].get("context") or {}
        if (manifest.get("task") or {}).get("decision_budget"):
            # G8 (experiment-plan-round1.md §3): the LLM-call budget left before this packet counts, next to
            # remaining_steps. The budget is on_stall.max_teacher_decisions under on_stall, else review.max_decisions
            # (0: unbounded, shown as null). Monotone over the episode's packets.
            if self.takeover is not None and self.segment is None:
                budget, used = self.takeover.cfg["max_teacher_decisions"], self.takeover.decisions
            else:
                budget, used = self.cfg["review"].get("max_decisions", 0), state.decision
            packet["decisions_left"] = budget - used if budget > 0 else None
        report = None
        if self.segment is not None:
            # G13: why the LLM component is consulted now, what OpenWAM did alone since its last packet, and the
            # bound its next_look may name.
            handover = self.segment.handovers[-1]
            packet["takeover"] = {"reason": handover["reason"], "since_native_step": handover["native_step"],
                                  "openwam_alone_since_last_packet": dict(state.auto),
                                  "trigger": trigger_view(handover["reason"], handover.get("evidence")),
                                  "next_look": {"max_after_chunks": self.segment.max_chunks,
                                                "default_after_chunks": self.segment.default_chunks}}
            report = {"request_id": request_id, "decision": state.decision, "native_step": self.gate.step,
                      "wake": handover["reason"], "segment": (self.segment.segments or [None])[-1],
                      "openwam_alone": dict(state.auto), "receipts": list(state.receipts),
                      "interval_digest": list(state.digest) if state.digest is not None else None}
            state.auto.update(chunks=0, steps=0)
            self.takeover.packet()
        elif self.takeover is not None:
            # on_stall: why the teacher has control, and what OpenWAM did alone since the teacher's last packet.
            handover = self.takeover.handovers[-1]
            packet["takeover"] = {"reason": handover["reason"], "since_native_step": handover["native_step"],
                                  "teacher_decisions_left": (self.takeover.cfg["max_teacher_decisions"]
                                  - self.takeover.decisions - 1 if self.takeover.cfg["max_teacher_decisions"] else None),
                                  "openwam_alone_since_last_packet": dict(state.auto)}
            if "evidence" in handover:
                packet["takeover"]["evidence"] = handover["evidence"]
            if "digest" in self.cfg["review"]:
                # G5: the wake reason with the detector's three measurements, whatever the reason.
                packet["takeover"]["trigger"] = trigger_view(handover["reason"], handover.get("evidence"))
            state.auto.update(chunks=0, steps=0)
            self.takeover.packet()
        if state.digest is not None:
            # G5 (review.digest per_chunk): per chunk OpenWAM ran alone since the last packet, the diagnostics its
            # packet would have shown and the gripper ledger after it; [] when there was none.
            packet["interval_digest"] = list(state.digest)
            state.digest.clear()
        if state.interval_memory is not None:
            packet["interval_digest"] = state.interval_memory.drain()
            if report is not None:
                report["interval_digest"] = packet["interval_digest"]
        state.receipts.clear()
        packet = plain(packet)
        if (self.cfg["review"].get("task_protocol") or self.cfg.get("stmem", {}).get("task_protocol")) == "press_by_number-public-v1":
            from .task_specs import PRESS_TASK_SPEC
            packet["task_spec"] = copy.deepcopy(PRESS_TASK_SPEC)
        if self.stmem is not None:
            spatial = copy.deepcopy(self.latest_spatial_context or self.stmem.last_context)
            if self.cfg["stmem"].get("context_max_entities"):
                from .memory_context import spatial_context
                spatial = spatial_context(spatial, max_entities=self.cfg["stmem"]["context_max_entities"],
                    max_chars=self.cfg["stmem"]["context_max_chars"])
            # Dictionary insertion order makes the spatial snapshot precede the
            # decision context in every model-visible packet.
            packet = {"spatial_context": spatial, **packet}
        from .task_tree_runtime import ensure_tree
        ensure_tree(self, packet)
        if self.guard is not None:
            # What the guard holds the teacher to at this packet; shown to it only when the guard is enforced.
            status = self.guard.at_packet(packet)
            if self.guard.enforce:
                packet["review_guard"] = status
        # The oracle boundary (declared-factor-mode.md §4), under a declared manifest only: no forbidden key at any
        # depth and the packet's top-level keys are exactly its field set. Reads only; a refusal ends the attempt
        # before the packet is written. A lock without a manifest is untouched on this path.
        if "context" in self.cfg["review"]:
            from .contracts import check_packet, packet_field_set
            check_packet(packet, packet_field_set(self.cfg["review"]))
        directory = self.output / "teacher" / "packets"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{state.decision:04d}.json"
        path.write_bytes(encoded(packet))
        state.packet, state.images, state.started = packet, images, time.monotonic()
        state.decision += 1
        if report is not None:
            # The segment report (G13): the passed chunks' receipts and, under digest per_chunk, the very digest the
            # packet carries.
            self.emit("segment_report", report)
        self.emit("teacher_packet", {"request_id": request_id, "decision": packet["decision"],
                                     "native_step": self.gate.step, "sha256": digest(packet),
                                     "artifact": str(path.relative_to(self.output)),
                                     "candidate_rows": len(valid), "candidate_sha256": digest(valid)})
        from .teacher import TEACHER_WAKE
        await self.client.call("event/emit", {"kind": TEACHER_WAKE, "payload": {
            "request_id": request_id, "observation_id": state.observation_id, "packet_sha256": digest(packet)},
            "collapse_key": "teacher_packet"})

    def review_guard_refusal(self, packet, decision, assessment):
        """The review guard's refusal of a validated teacher decision for this packet (oeh_policy.review_guard), or
        None. A refusal is recorded either way; in shadow mode (enforce false) the decision goes on unchanged."""
        if self.guard is None:
            return None
        found = self.guard.check(packet["request_id"], decision, assessment)
        if found is None:
            return None
        rule, reason = found
        self.emit("review_guard", {"request_id": packet["request_id"], "rule": rule, "mode": decision["mode"],
                                   "steps": decision["steps"], "enforced": self.guard.enforce, "reason": reason,
                                   "native_step": self.gate.step})
        return reason if self.guard.enforce else None

    def _review_images(self, decision):
        """The contract's views as the teacher sees them (downscaled PNG, like the paper's 480 px previews), saved
        beside the packet so the model-visible input can be reconstructed from the attempt."""
        from .teacher import image_items
        directory = self.output / "teacher" / "packets"
        directory.mkdir(parents=True, exist_ok=True)
        views = self.contract.review_views(self.observation)
        if tuple(views) != tuple(self.contract.review_cameras):  # the teacher's first text names this many views
            raise RuntimeError(f"review views {list(views)} differ from the contract's "
                               f"{list(self.contract.review_cameras)}")
        teacher = self.cfg["review"]["teacher"]
        return image_items(views, teacher["image_max_edge"], directory, f"{decision:04d}",
                           downsample_k=teacher.get("image_downsample_k", 1))

    async def _review_harness(self, kind, reason):
        """A harness action proposed by the deterministic fast brain: one hold row, or the fallback student chunk."""
        state = self.review
        if self.turn_ended is not None and not self.turn_ended.done():
            await asyncio.wait_for(asyncio.shield(self.turn_ended), self.cfg["decision_seconds"])
        if kind in ("fallback", "auto") and (getattr(self, "task_tree", None) is not None or
                getattr(self, "context_recovery_pending", False)):
            kind = "hold"  # Automatic motion cannot bypass a leaf binding or pending memory recovery.
        if kind in ("fallback", "auto") and not state.candidate:
            kind = "hold"  # no admissible student rows: hold, and the next call takes a fresh proposal
            state.finished[state.packet["request_id"]] = "abandoned"
            self.emit("review_packet_abandoned", {"request_id": state.packet["request_id"], "reason": reason})
        student_cap = self.method.limits["student_max_steps"]
        params = {"observation_id": state.observation_id, "candidate_id": state.candidate_id,
                  "mode": "hold" if kind == "hold" else "student",
                  "steps": 1 if kind == "hold" else min(len(state.candidate), student_cap),
                  "reason": f"harness {kind}: {reason}"}
        self.emit("review_harness_action", {"kind": kind, "request_id": state.packet["request_id"] if state.packet
                                            else None, "reason": reason, "native_step": self.gate.step})
        loop = asyncio.get_running_loop()
        self.fast_request = params
        self.active = "fast"
        self.observation_id = state.observation_id
        self.ready = loop.create_future()
        self.turn_ended = loop.create_future()
        self.decision_budget = self.cfg["decision_seconds"]
        self.decision_deadline = min(time.monotonic() + self.decision_budget, self.gate.deadline)
        await self.client.call("event/emit", {"kind": FAST_TICK, "payload": {
            "observation_id": state.observation_id, "native_step": self.gate.step}, "collapse_key": "policy_tick"})
        try:
            rows = await asyncio.wait_for(asyncio.shield(self.ready), max(0, self.decision_deadline - time.monotonic()))
        except BaseException:
            self.gate.stop("harness_action_failed")
            raise
        finally:
            self.fast_request = None
        if state.packet is not None and state.packet["request_id"] in state.finished:
            state.packet = None  # an abandoned packet: the next call infers a fresh proposal
        self.calls += 1
        return rows

    async def _fall_back(self, failure):
        """A slow decision ended before anything reached the gate (reviewer M1): record why, then decide fast on
        the same observation. It counted against max_calls; the gate keeps running."""
        self.slow_outcome = "no_action" if failure.kind == "slow_decision_no_action" else "failed"
        self.emit(failure.kind, {**failure.evidence, "native_step": self.gate.step, "slow_calls": self.slow_calls})
        decision = self.cfg["decision_seconds"]
        # The slow turn, and any intent it already made, must be over before the fast path decides.
        try:
            await asyncio.wait_for(asyncio.shield(self.turn_ended), decision)
        except asyncio.TimeoutError:
            self.gate.stop("slow_turn_did_not_end")
            raise RuntimeError("slow brain turn did not end after a failed decision") from None
        if self.executing is not None and (await asyncio.wait({self.executing}, timeout=decision))[1]:
            self.gate.stop("slow_intent_did_not_end")
            raise RuntimeError("slow intent still executing after a failed decision")
        if self.gate.stopped or self.gate.pending is not None:
            raise RuntimeError("observation or execution lifecycle is not ready")
        if time.monotonic() >= self.gate.deadline:
            self.gate.stop("budget_exhausted")
            raise RuntimeError("episode budget exhausted")
        # The failed decision leaves nothing behind: a subgoal its `vla` intent set is void, exactly as right
        # after the escalation began (the contract's `escalated` hook; LIBERO keeps subgoals across decisions),
        # and the backend decides on the observation the fast path would have used.
        self.contract.escalated(self)
        model_input = self.contract.model_observation(self, self.observation)
        if self.backend_input_sha256 != observation_digest(model_input):
            await asyncio.to_thread(self.backend.update_obs, copy.deepcopy(model_input))
            self.backend_input_sha256 = observation_digest(model_input)
        try:
            return await self._decide("fast", None)
        except FastRefused:
            # One escalation per native decision keeps the call bounded: a refused chunk after a failed
            # escalation ends the episode, as a refused chunk without budget does.
            self.gate.stop("fast_rejected_after_failed_escalation")
            raise RuntimeError("fast chunk refused after a failed escalation") from None

    def _slow_persona(self):
        if self.slow_settings.get("pose_control"):
            from .slow_reference import reference_persona
            prompt = reference_persona()
            if self.notes is not None:
                from .llm_contract import reference_section
                prompt += "\n" + reference_section("q2-experience") + "\n"
            return prompt
        return self.contract.persona_path.read_text(encoding="utf-8")

    def _slow_fills(self) -> bool:
        """Slow mode with brain.on_no_call fast_chunk or hold: the fast brain fills a silent turn."""
        return self.cfg["mode"] == "slow" and self.slow_settings.get("on_no_call") in ("fast_chunk", "hold")

    def remember(self, params, result, source):
        """Slow mode: one executed decision as the next payloads recall it (G6 route (a): compact re-injected state)."""
        entry = {"decision": len(self.decisions_made), "mode": params["mode"], "phase": params["phase"],
                 "target": params["target"], "execute_steps": params["execute_steps"],
                 **{field: copy.deepcopy(params[field]) for field in mode_fields(
                     pose_control=self.slow_settings.get("pose_control", False)).get(params["mode"], ())
                    if field in params},
                 "receipt": {"executed_steps": result["executed_steps"], "source": source}}
        failed = getattr(self, "slow_failed_reason", None)
        if failed and source.startswith("on_call_failure:"):
            entry["failed_decision"] = failed
        self.slow_failed_reason = None
        self.decisions_made.append(plain(entry))
        if self.slow_settings.get("execution_feedback"):
            self.slow_feedback_pending = {"request_observation_id": params["observation_id"],
                "completion_native_step": self.gate.step, "mode": params["mode"],
                "requested_steps": params["execute_steps"], "executed_steps": result["executed_steps"],
                "source": source, "targets": copy.deepcopy(params.get("targets"))}

    def _slow_payload(self, payload):
        """The slow payload with the slow role's additions (G6 recent_decisions, G8 decisions_left), checked against
        the oracle boundary's forbidden keys when brain.scan_payload is on (plan §3 addition 1)."""
        settings = self.slow_settings
        if settings.get("memory_last_n"):
            payload["recent_decisions"] = copy.deepcopy(self.decisions_made[-settings["memory_last_n"]:])
        if settings.get("show_decisions_left"):
            # The LLM-call budget left before this call counts (this call is self.calls - self.fills).
            cap = self.cfg["brain"]["max_calls"]
            payload["decisions_left"] = cap - (self.calls - self.fills - 1) if cap > 0 else None
        if self.slow_context is not None:
            payload["public_context"] = copy.deepcopy(self.slow_context)
        if self.notes is not None:
            # Slow requests may not retain earlier turns: supply the same public snapshot each time.
            payload["task_notes"] = self._task_notes()
        if settings.get("pose_control"):
            payload["control_contract"] = {"frame": "env_relative_world", "quaternion": "wxyz",
                "position_units": "metres", "reference_point": "native_end_effector",
                "phase_policy": settings.get("phase_policy", "checkpoints"),
                "gripper_state_semantics": self.cfg["gripper_state_semantics"],
                "max_episode_steps": self.cfg["native_step_limit"],
                "remaining_steps": self.cfg["native_step_limit"] - self.gate.step,
                "fallback": settings.get("on_call_failure", "stop")}
            payload["history_coverage"] = {"decisions_total": len(self.decisions_made),
                "decisions_shown": len(payload.get("recent_decisions", [])),
                "older_decisions_omitted": max(0, len(self.decisions_made) - len(payload.get("recent_decisions", [])))}
        if settings.get("execution_feedback"):
            pending = self.slow_feedback_pending
            payload["execution_feedback"] = None
            if pending is not None:
                feedback = {key: copy.deepcopy(value) for key, value in pending.items() if key != "targets"}
                fresh = self.observed_step is not None and self.observed_step >= pending["completion_native_step"]
                feedback["measurement_native_step"] = self.observed_step if fresh else None
                feedback["target_error"] = None
                if fresh and pending["targets"] is not None:
                    from .q1_actions import pose_feedback
                    feedback["target_error"] = pose_feedback(pending["targets"], self.observation["state"])
                payload["execution_feedback"] = feedback
        if settings.get("agent_termination"):
            payload["termination_control"] = {"tool": "robot_oeh_finish",
                "statuses": ["completed", "unable_to_continue"], "claim_source": "agent",
                "benchmark_outcome": "owned_by_evaluator"}
        if settings.get("scan_payload"):
            from .contracts import _forbidden_keys
            found = _forbidden_keys(payload)
            if found:
                self.emit("payload_forbidden_keys", {"keys": found, "native_step": self.gate.step,
                                                     "observation_id": self.observation_id})
                self.gate.stop("payload_forbidden_keys")
                raise ValueError(f"slow payload carries forbidden keys (the oracle boundary): {found}")
        return payload

    async def _slow_no_call(self, failure):
        """Slow mode, brain.on_no_call: the LLM component's turn ended without an intent. fast_chunk: one whole OpenWAM
        chunk through the fast path; hold: one hold row (a one-step wait); stop: the attempt stops. The silent call
        counted against max_calls; the fill does not."""
        kind = self.slow_settings["on_no_call"]
        self.emit("slow_no_call", {**failure.evidence, "on_no_call": kind, "native_step": self.gate.step,
                                   "calls": self.calls - self.fills})
        return await self._slow_fill(kind, "slow_no_call", "on_no_call")

    async def _slow_call_failure(self, failure):
        """Slow mode, brain.on_call_failure fill_once: the LLM component's turn ended after a failed model call or
        with an intent refused by admission. The failed decision counted against max_calls; the on_no_call fill
        (fast_chunk or hold) runs once in its place, with receipt source ``on_call_failure:<fill>``."""
        kind = self.slow_settings["on_no_call"]
        self.emit("slow_decision_failed", {**failure.evidence, "on_call_failure": kind, "native_step": self.gate.step,
                                           "calls": self.calls - self.fills})
        # The fill's decision record carries why the LLM's own decision did not run, so the next payload's
        # recent_decisions shows the refusal or failure instead of only the fill's receipt source.
        self.slow_failed_reason = str(failure.evidence.get("reason", ""))[:300]
        return await self._slow_fill(kind, "slow_decision_failed", "on_call_failure")

    async def _slow_fill(self, kind, event_kind, source):
        try:
            await asyncio.wait_for(asyncio.shield(self.turn_ended), self.cfg["decision_seconds"])
        except asyncio.TimeoutError:
            self.gate.stop("slow_turn_did_not_end")
            raise RuntimeError(f"slow brain turn did not end after {event_kind}") from None
        if kind == "stop":
            self.gate.stop(event_kind)
            raise RuntimeError(f"{event_kind}: the slow brain answered with no intent and brain.on_no_call is stop")
        if self.gate.stopped or self.gate.pending is not None:
            raise RuntimeError("observation or execution lifecycle is not ready")
        model_input = self.contract.model_observation(self, self.observation)
        if self.backend_input_sha256 != observation_digest(model_input):
            await asyncio.to_thread(self.backend.update_obs, copy.deepcopy(model_input))
            self.backend_input_sha256 = observation_digest(model_input)
        self.fills += 1
        self.fill_kind, self.fill_source = kind, source
        try:
            # The fast brain's turn stays the active one until the next slow decision (its turn_end ends it).
            return await self._decide("fast", None)
        finally:
            self.fill_kind = self.fill_source = None

    async def _decide(self, active, reason):
        if self.cfg["mode"] == "hybrid" and active == "fast" and self.active == "slow":
            self.emit("return_to_fast", {"native_step": self.gate.step, "after": self.slow_outcome})
        self.active = active
        self.calls += 1
        if active == "slow" and self.cfg["mode"] == "hybrid":
            self.slow_calls += 1
            self.slow_outcome = self.slow_error = self.slow_refusal = None
            self.escalation.escalated(self.gate.step)
            self.emit("escalation", {"reason": reason, "native_step": self.gate.step,
                                     "sim_time_s": self.escalation.sim_time(self.gate.step),
                                     "slow_calls": self.slow_calls})
        self.observation_id = uuid.uuid4().hex
        self.emit("decision_observation", {"observation_id": self.observation_id,
                  "sha256": observation_digest(self.observation), "native_step": self.gate.step,
                  "brain": active})
        self.consumed = False
        self.ready = asyncio.get_running_loop().create_future()
        self.turn_ended = asyncio.get_running_loop().create_future()
        # The episode's first decision pays the model's warm-up (optional frozen first_decision_seconds).
        self.decision_budget = (self.cfg.get("first_decision_seconds", self.cfg["decision_seconds"])
                                if self.calls == 1 else self.cfg["decision_seconds"])
        self.decision_deadline = min(time.monotonic() + self.decision_budget, self.gate.deadline)
        if self.cfg["mode"] in ("hybrid", "slow") and active == "fast":
            await self.client.call("event/emit", {"kind": FAST_TICK, "payload": {
                "observation_id": self.observation_id, "native_step": self.gate.step},
                "collapse_key": "policy_tick"})
        else:
            payload = self.camera_payload()
            if self.cfg["mode"] == "slow" and self.slow_settings:
                payload = self._slow_payload(payload)
            if reason is not None:
                payload["escalation"] = {"reason": reason,
                                         "sim_time_s": self.escalation.sim_time(self.gate.step)}
                self.contract.escalated(self)
            await self.client.call("event/emit", {"kind": "obs/robot_rgb", "payload": payload,
                                                 "collapse_key": "policy_camera"})
        slow_hybrid = self.cfg["mode"] == "hybrid" and active == "slow"
        # Slow mode with brain.on_call_failure fill_once: a decision that reaches its deadline with no admitted intent
        # (no answer in time, or an intent that expired before its rows were offered) is a failed call, so it takes
        # the fill once like hybrid's fallback instead of stopping the attempt (P15; round-1 run, 2026-10-20).
        slow_fill = (self.cfg["mode"] == "slow" and active == "slow"
                     and self.slow_settings.get("on_call_failure") == "fill_once")
        try:
            rows = await asyncio.wait_for(asyncio.shield(self.ready),
                                          max(0, self.decision_deadline - time.monotonic()))
        except (FastRefused, SlowFailed):
            raise
        except asyncio.TimeoutError:
            if not (slow_hybrid or slow_fill) or self.gate.stopped:
                self.gate.stop("decision_failed")
                raise
            if not (self.ready.done() and not self.ready.cancelled() and self.ready.exception() is None):
                if self.gate.pending is not None:
                    self.gate.stop("decision_failed")
                    raise
                raise SlowFailed("no admitted intent before the decision deadline",
                                 model_error=self.slow_error) from None
            rows = self.ready.result()  # accepted at the deadline itself
        except BaseException:
            self.gate.stop("decision_failed")
            raise
        if slow_hybrid:
            self.slow_outcome = "executed"
        return rows

    async def next_action(self):
        """The next native action for loops that submit one action per official call.

        LIBERO's predict and RoboTwin's eval callback ask for exactly one action per
        observation. Rows of a chunk admitted by one decision are served in order across
        later calls; a following row is computed from the newest observation and
        admitted just before its submission. A new decision starts only when the
        chunk is exhausted.
        """
        pending = self.gate.pending
        if pending is None:
            return (await self.get_action())[0]
        if self.gate.stopped or self.gate.inflight is not None or self.observation is None:
            raise RuntimeError("observation or execution lifecycle is not ready")
        if pending["position"] == len(pending["actions"]):
            if pending["follow"] is None or self.observed_step != self.gate.step:
                raise RuntimeError("a following row needs the post-step observation")
            row = pending["follow"](self.observation["state"])
            self.gate.admit(self.contract.validate_action(row, self.cfg["action_limits"]))
        return copy.deepcopy(pending["actions"][pending["position"]])

    async def before_action(self, action):
        Runtime.check_llm_budget(self)
        if self.plugin.disposed.is_set() or self.kernel.returncode is not None:
            self.gate.stop("kernel_or_plugin_disconnected")
            raise RuntimeError("OEH execution authority unavailable")
        # Ask the live kernel on every command, not only when the chunk was made.
        status = await self.client.call("robot/state")
        if status.get("fsm") != "ACTING":
            self.gate.stop("kernel_not_ready")
            raise RuntimeError("kernel is not ready for motion")
        return self.gate.before(action)

    async def after_action(self, ticket, action, state, terminal):
        safe_state = self.contract.project_state(state)
        self.gate.after(ticket, action, terminal=terminal, observed_state_sha256=digest(safe_state))
        if terminal:
            self.environment_terminal_seen = True
        self.method.acknowledged(action)
        if self.review is not None and self.review.interval_memory is not None:
            self.last_action_measurement = {"native_step": self.gate.step,
                "positions": {arm: plain(safe_state[arm + "_ee_pose"][:3]) for arm in self.method.arms}}
        if self.escalation is not None:
            self.escalation.observe(self.gate.step, *self.contract.motion_features(safe_state, action))
        if self.takeover is not None:
            # Measured arm positions and the raw commanded openings (turn-backs need the raw values).
            self.takeover.observe(self.gate.step, [safe_state[arm + "_ee_pose"][:3] for arm in ("left", "right")],
                                  [float(action[arm + "_ee_joint_state"][0]) for arm in ("left", "right")])

    async def drain_events(self):
        values = copy.deepcopy(self.events[self.drain_cursor:])
        self.drain_cursor = len(self.events)
        return values

    async def close(self, reason="episode_closed"):
        if self.closed:
            return
        self.closed = True
        active_seconds = time.monotonic() - self.started
        if self.review is not None:
            self.review.shutdown = True  # the teacher proposes nothing more
            if self.teacher is not None:
                self.teacher.close()  # a call in flight neither retries nor records after this
                summary = {**self.teacher.summary(), "counters": dict(self.review.counters), "fatal": self.review.fatal}
                if self.takeover is not None:
                    summary["takeover"] = self.takeover.summary()
                if self.guard is not None:
                    summary["guard"] = self.guard.summary()
                self.emit("teacher_summary", summary)
            elif getattr(self.teacher_model_client, "close", None) is not None:
                self.teacher_model_client.close()  # a Codex app-server started by the preflight alone
            elif self.teacher_start is not None:
                from .codex_teacher import discard
                discard(self.teacher_start)  # a start this runtime stopped waiting for (its call timed out)
        # Stop accepting actions first; closure never waits for another model judgment.
        try:
            from .semmem_lifecycle import close_episode
            close_episode(self, reason)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.cleanup_errors.append({"stage": "semmem_closeout", "error": type(exc).__name__})
            self.emit("cleanup_failure", self.cleanup_errors[-1])
        if self.gate.pending is None and self.turn_ended and not self.turn_ended.done():
            try:
                await asyncio.wait_for(asyncio.shield(self.turn_ended), 2)
            except asyncio.TimeoutError:
                pass
        self.gate.stop(reason)
        if self.executing is not None:
            self.executing.cancel()
            await asyncio.gather(self.executing, return_exceptions=True)
        if self.review is not None and self.review.interval_memory is not None:
            self.review.interval_memory.reset()
            self.review.pending_digest = None
            self.last_action_measurement = None
        async def cleanup(label, operation):
            try:
                await asyncio.wait_for(operation, 2)
            except Exception as exc:
                self.cleanup_errors.append({"stage": label, "error": type(exc).__name__})
                self.emit("cleanup_failure", self.cleanup_errors[-1])
        if self.stmem is not None:
            await cleanup("stmem_close", self.stmem.close())
        if self.fast:
            await cleanup("brain_detach", self.fast.detach())
        if self.teacher:
            await cleanup("teacher_detach", self.teacher.detach())
        if self.plugin:
            await cleanup("plugin_unregister", self.plugin.unregister())
        for client in (self.brain_client, self.teacher_client, self.client, self.watcher):
            if client:
                await cleanup("client_close", client.close())
        if self.kernel and self.kernel.returncode is None:
            try:
                self.kernel.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.kernel.wait(), 5)
            except asyncio.TimeoutError:
                self.kernel.kill()
                await self.kernel.wait()
        if self.kernel_log:
            self.kernel_log.close()
        if self.short_socket:
            self.short_socket.cleanup()
        if self.teacher_relayed is not None:
            # The prestarted app-server the lane adopted: the lane's close (above) already ended its relay, which
            # stops it; this is for a kernel that never took it. Not before the kernel stops, so that a call still
            # in flight sees the client closed first, as on the python channel, not the app-server gone.
            self.teacher_relayed.close()
            self.teacher_relay_dir.cleanup()
        if getattr(self,"llm_budget",None) is not None:
            self.emit("llm_token_budget_summary",self.llm_budget.status())
        if self.slow_settings.get("pose_control"):
            from .slow_reference import episode_summary
            summary = episode_summary(self.events, decisions=self.calls - self.fills, fills=self.fills,
                native_steps=self.gate.step, active_seconds=active_seconds,
                wall_seconds=time.monotonic() - self.started, close_reason=reason,
                agent_claim=self.agent_termination, environment_terminal=self.environment_terminal_seen)
            self.emit("slow_summary", summary)
            (self.output / "slow-summary.json").write_bytes(encoded(summary))
        self.emit("attempt_closed", {"reason": reason, "native_steps": self.gate.step,
                                    "wall_seconds": time.monotonic() - self.started,
                                    "native_outcome": "owned_by_evaluator"})
        self.sealed = True
        inventory = {str(p.relative_to(self.output)): file_digest(p)
                     for p in sorted(self.output.rglob("*")) if p.is_file() and p.name != "seal.json"}
        (self.output / "seal.json").write_bytes(encoded({"files": inventory, "sha256": digest(inventory)}))
