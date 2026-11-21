"""The Python half of the brain-channel review teacher (``review.teacher.channel: brain``).

With the brain channel the teacher's model loop runs inside the kernel, in oeh-brain's ``brain-teacher`` lane
(crates/oeh-brain/src/teacher/lane.rs): it keeps the episode's Codex thread, makes every model call, writes the call
artifacts and proposes each decision through admission as brain ``teacher``. What the loop needs from this process,
it asks for at the moment ``TeacherBrain`` would have read it, and this module answers with ``TeacherBrain``'s own code:

  begin     the packet to answer (``state.packet``, unless it is empty, OpenWAM's own ``auto`` packet, or closed),
            its text exactly as ``TeacherBrain._decide`` sends it and the paths of its camera views
  state     ``state.closed(request_id)`` and ``runtime.closed``
  validate  ``validate_teacher_response`` / ``intent_from_decision`` (a refusal records ``teacher_invalid``), then the
            review guard's refusal when ``review.guard`` is set (``Runtime.review_guard_refusal``); a valid decision
            is booked in ``state.assessments`` and ``state.waiting`` as ``_propose`` books it
  window    ``state.wait_window``; moot once the packet closed, else ``teacher_decision`` is recorded and the lane
            proposes the intent (``_propose``'s loop: window closed, preempted or cancelled means wait and propose again)
  unwait    ``state.waiting.discard``
  emit      a record ``TeacherBrain`` makes (model_call, teacher_call_error, model_failure, model_timing, usage)
  fatal     ``teacher_fatal`` and the attempt's stop (``state.fatal``, ``state.fatal_error``, ``give_up``)
  failed    ``teacher_failed`` and ``give_up``

The lane writes ``brain/teacher_bridge`` world-log events (this process reads them through the runtime's ``log/tail``
watcher, in order, in ``Runtime._event``) and reads ``obs/teacher_bridge`` events this process emits. Every message
carries the lane's counters; ``summary()`` is ``TeacherBrain.summary()`` over the latest of them.
"""
from __future__ import annotations

import asyncio
import json
import os

from .contracts import encoded
from .review import intent_from_decision, tool_spec, validate_teacher_response
from .teacher import TEACHER_METHODS, prompt_text

FROM_LANE = "brain/teacher_bridge"
TO_LANE = "obs/teacher_bridge"
WAKE_POLICY = {"default": "ignore", "rules": {"obs/teacher_packet": "wake", "safety/estop": "interrupt"},
               "budget": {"max_consecutive_wakes": 100000, "min_interval_ms": 0}}  # TeacherBrain's own


class LanePreflightFailed(RuntimeError):
    """The lane's preflight raised something ``Runtime.start`` does not report as a teacher preflight failure."""


class TeacherBridge:
    """Stands where ``TeacherBrain`` stands in the runtime (``summary``, ``close``, ``detach``)."""

    def __init__(self, runtime):
        self.runtime = runtime
        self.cfg = runtime.cfg["review"]["teacher"]
        self.prompt = prompt_text(runtime.cfg["review"].get("schedule"))
        self.directory = runtime.output / "teacher"
        self.directory.mkdir(exist_ok=True)
        (self.directory / "prompt.md").write_text(self.prompt, encoding="utf-8")
        (self.directory / "tool.json").write_bytes(encoded(tool_spec()))
        self.packets = {}  # request_id -> the packet being answered (TeacherBrain.on_turn's `packet`)
        self.decisions = {}  # request_id -> (intent, trigger) of its latest valid decision
        self.counters = {"calls": 0, "failed_calls": 0, "invalid_or_refused": 0,
                         "tokens": {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0}, "cost": 0.0}
        self.preflight_result: asyncio.Future | None = None
        self.tasks: set[asyncio.Task] = set()

    # -- the kernel row -------------------------------------------------------------------------------------------
    def row(self, accounts: list[str]) -> dict:
        from . import codex_teacher
        from .contracts import CAPABILITY
        from .review import response_schema
        runtime = self.runtime
        return {"id": "brain-teacher", "config": {
            "brain_id": "teacher", "wake": WAKE_POLICY, "output": str(runtime.output),
            "episode_id": runtime.episode_id, "teacher": self.cfg, "accounts": accounts,
            "prompt_file": str(self.directory / "prompt.md"), "output_schema": response_schema(),
            "capability": CAPABILITY, "turn_seconds": runtime.cfg["review"]["turn_seconds"],
            "method": TEACHER_METHODS[self.cfg["api"]],
            # The client's start policy, read at run time like CodexClient reads it (tests change it there).
            "launch": {"attempts": codex_teacher.LAUNCH_ATTEMPTS, "timeout_s": codex_teacher.LAUNCH_TIMEOUT_S,
                       "backoff_s": list(codex_teacher.LAUNCH_BACKOFF_S)},
            # CodexClient starts the app-server from this process: in its working directory, with its os.environ.
            # The kernel runs with that environment plus the profile `oeh run` names (OEH_PROFILE, unless set).
            "process_cwd": os.getcwd(),
            "env_remove": [] if "OEH_PROFILE" in os.environ else ["OEH_PROFILE"],
            # The prestarted app-server the lane carries on with (Runtime._brain_teacher_prestart), if any.
            **({"adopt": runtime.teacher_adopt} if runtime.teacher_adopt is not None else {})}}

    # -- TeacherBrain's interface ---------------------------------------------------------------------------------
    def summary(self) -> dict:
        c = self.counters
        return {"model": self.cfg["model"], "calls": c["calls"], "failed_calls": c["failed_calls"],
                "invalid_or_refused": c["invalid_or_refused"], "tokens": dict(c["tokens"]),
                "cost_usd_estimate": round(c["cost"], 4), "context_prunes": 0}

    def close(self):
        """TeacherBrain.close(): a call in flight neither retries nor records after this; the app-server stops."""
        self._send({"op": "close"})

    async def detach(self):
        """The lane is the kernel's: it goes with the kernel's own shutdown."""

    async def preflight(self) -> dict:
        """The lane's preflight (started when the kernel applied its row), as ``client.preflight()`` returns it."""
        from .teacher import TeacherFatal, TeacherUnavailable
        self.preflight_result = asyncio.get_running_loop().create_future()
        self._send({"op": "preflight"})
        message = await self.preflight_result
        if message.get("ok"):
            return message["result"]
        error = message.get("error") or {}
        if error.get("class") == "TeacherFatal":
            raise TeacherFatal(error.get("message", ""), error.get("errors") or [])
        if error.get("class") == "TeacherUnavailable":
            raise TeacherUnavailable(error.get("message", ""), error.get("errors") or [])
        raise LanePreflightFailed(error.get("message", "teacher preflight failed"))

    # -- messages from the lane -----------------------------------------------------------------------------------
    def on_event(self, payload: dict):
        """One ``brain/teacher_bridge`` event, in world-log order (runs inline in the watcher's read loop)."""
        if isinstance(payload.get("counters"), dict):
            self.counters = payload["counters"]
        op = payload.get("op")
        runtime, state = self.runtime, self.runtime.review
        if op == "emit":
            runtime.emit(payload["kind"], payload["payload"])
        elif op == "fatal":  # TeacherBrain.on_turn, `except TeacherFatal`
            request_id = payload.get("request_id")
            runtime.emit("teacher_fatal", {"request_id": request_id, "error": payload["error"]})
            state.fatal = payload["error"]
            state.fatal_error = payload.get("fatal_error")  # the provider's own status, type and code
            state.give_up(request_id, "fatal")
        elif op == "failed":  # TeacherBrain.on_turn, `except Exception`
            request_id = payload.get("request_id")
            runtime.emit("teacher_failed", {"request_id": request_id, "error": payload["error"]})
            state.give_up(request_id, payload.get("reason"))
        elif op == "unwait":
            state.waiting.discard(payload.get("request_id"))
        elif op == "preflight":
            if self.preflight_result is not None and not self.preflight_result.done():
                self.preflight_result.set_result(payload)
        elif op == "begin":
            self._reply(payload, self._begin())
        elif op == "state":
            self._reply(payload, {"closed": state.closed(payload.get("request_id")),
                                  "runtime_closed": bool(runtime.closed)})
        elif op == "validate":
            self._reply(payload, self._validate(payload.get("request_id"), payload.get("arguments")))
        elif op == "window":
            self._spawn(self._window(payload))

    def _begin(self) -> dict:
        """TeacherBrain.on_turn: the packet it would answer, as ``_decide`` sends it."""
        state = self.runtime.review
        packet = state.packet
        # An on_stall boundary where OpenWAM runs alone holds an `auto` packet: never the teacher's to answer.
        if packet is None or packet.get("auto") or state.closed(packet["request_id"]):
            return {"request_id": None}
        self.packets = {packet["request_id"]: packet}
        # state.images are these files' bytes (runtime._review_images saves each view beside the packet).
        views = [str(self.directory / "packets" / meta["artifact"]) for meta in packet.get("images") or []]
        return {"request_id": packet["request_id"], "decision": packet["decision"],
                "text": json.dumps(packet, separators=(",", ":")), "images": views}

    def _validate(self, request_id, arguments) -> dict:
        """TeacherBrain._decide's validation, then the bookkeeping at the top of ``_propose``."""
        runtime, state = self.runtime, self.runtime.review
        packet = self.packets.get(request_id)
        try:
            try:
                parsed = json.loads(arguments or "{}")
                decision = validate_teacher_response(parsed, packet, runtime.method.limits,
                                                     final_home_steps=runtime.final_home_steps)
            except (ValueError, TypeError) as exc:
                runtime.emit("teacher_invalid", {"request_id": request_id, "error": str(exc)[:800]})
                return {"valid": False, "error": str(exc)}
            # The review guard (oeh_policy.review_guard), when set, refuses before admission as in TeacherBrain._decide.
            # The lane hands the refusal back to the model with this error_type, as TeacherBrain._decide does.
            refusal = runtime.review_guard_refusal(packet, decision, parsed.get("assessment"))
            if refusal is not None:
                return {"valid": False, "error": refusal, "error_type": "review_guard"}
            intent = intent_from_decision(decision, packet["observation_id"], packet["candidate_id"])
            state.assessments[request_id] = {"assessment": parsed.get("assessment"), "trigger": decision["trigger"]}
            state.waiting.add(request_id)
            self.decisions[request_id] = (intent, decision["trigger"])
            return {"valid": True, "intent": intent}
        except Exception as exc:  # noqa: BLE001 - TeacherBrain.on_turn would have reported it as teacher_failed
            return {"exception": {"type": type(exc).__name__, "message": str(exc)}}

    async def _window(self, payload: dict):
        """TeacherBrain._propose up to the proposal: wait for the window, then record the decision."""
        runtime, state = self.runtime, self.runtime.review
        request_id = payload.get("request_id")
        await state.wait_window(request_id)
        if state.closed(request_id):
            self._reply(payload, {"moot": True})
            return
        intent, trigger = self.decisions[request_id]
        runtime.emit("teacher_decision", {"request_id": request_id, "intent": intent, "trigger": trigger})
        self._reply(payload, {"moot": False})

    # -- messages to the lane -------------------------------------------------------------------------------------
    def _reply(self, request: dict, fields: dict):
        self._send({"op": "reply", "id": request.get("id"), **fields})

    def _send(self, payload: dict):
        client = self.runtime.client
        if client is None or client.closed.is_set():
            return
        self._spawn(client.call("event/emit", {"kind": TO_LANE, "payload": payload}))

    def _spawn(self, coroutine):
        async def run():
            try:
                await coroutine
            except Exception:  # noqa: BLE001 - the kernel is going away; the lane stops with it
                pass
        task = asyncio.get_running_loop().create_task(run())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
