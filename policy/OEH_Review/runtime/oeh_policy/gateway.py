"""Two-phase native submission gate. A returned chunk is not an execution receipt."""
from __future__ import annotations

import asyncio
import copy
import time
import uuid

from .contracts import digest


class FollowingRows(list):
    """Leading rows of a chunk whose later rows depend on the achieved state.

    ``total`` rows are authorized by the one kernel admission; each later row comes
    from ``follow(state)`` with the post-step observation and is admitted and
    submitted alone (single-arm primitives through LIBERO's delta controller).
    """

    def __init__(self, rows, total, follow):
        super().__init__(rows)
        self.total, self.follow = total, follow


class ExecutionGate:
    def __init__(self, emit, *, native_step_limit, deadline):
        self.emit = emit
        self.native_step_limit = native_step_limit
        self.deadline = deadline
        self.step = 0
        self.pending = None
        self.inflight = None
        self.stopped = False
        self.sequence = 0

    def offer(self, actions, ctx, decision_deadline):
        if self.stopped or self.pending is not None:
            raise RuntimeError("execution gateway unavailable")
        if not actions or self.step >= self.native_step_limit:
            raise ValueError("no remaining native action budget")
        total = getattr(actions, "total", len(actions))
        if type(total) is not int or total < len(actions):
            raise ValueError("invalid chunk length")
        done = asyncio.get_running_loop().create_future()
        self.pending = {"actions": copy.deepcopy(list(actions)), "position": 0, "ctx": ctx,
                        "deadline": min(decision_deadline, self.deadline), "done": done,
                        "offer_id": uuid.uuid4().hex, "total": total,
                        "follow": getattr(actions, "follow", None)}
        for offset, action in enumerate(actions):
            self._admitted(self.sequence + offset, action, ctx)
        self.pending["start_index"] = self.sequence
        self.sequence += len(actions)
        return done

    def _admitted(self, index, action, ctx):
        trace = f"{ctx.trace_id}:{ctx.exec_id}:{index}"
        self.emit("action_proposed", {"index": index, "action": action, "sha256": digest(action)})
        self.emit("oeh_admission", {"trace_id": trace, "action_index": index,
                  "kernel_trace_id": ctx.trace_id, "kernel_exec_id": ctx.exec_id,
                  "verdict": "allow", "proposed_action_sha256": digest(action),
                  "scope": "kernel capability admission plus numeric action validation"})

    def admit(self, action):
        """Admit the next row of a following chunk just before its native submission."""
        p = self.pending
        if (self.stopped or p is None or self.inflight is not None or p["follow"] is None
                or p["position"] != len(p["actions"]) or len(p["actions"]) >= p["total"]):
            raise RuntimeError("no following row may be admitted now")
        index = p["start_index"] + len(p["actions"])
        p["actions"].append(copy.deepcopy(action))
        self._admitted(index, action, p["ctx"])
        self.sequence = index + 1

    def before(self, action):
        p = self.pending
        if self.stopped or p is None or self.inflight is not None:
            raise RuntimeError("no unconsumed native action authorization")
        if p["ctx"].cancelled.is_set() or time.monotonic() >= p["deadline"]:
            self.stop("cancelled_or_expired")
            raise RuntimeError("native authorization cancelled or expired")
        # The step budget bounds new chunks (offer). An admitted chunk's rows run as the official loop submits them,
        # as in the direct condition: RoboDojo's loop executes whole chunks and checks the end between chunks (its
        # steps past the limit are no-ops); LIBERO's and RoboTwin's loops never submit past theirs.
        expected = p["actions"][p["position"]]
        if digest(action) != digest(expected):
            self.stop("action_mismatch")
            raise ValueError("native submission differs from authorized action")
        index = p["start_index"] + p["position"]
        ticket = {"id": uuid.uuid4().hex, "offer_id": p["offer_id"], "step": self.step,
                  "action_sha256": digest(action),
                  "trace_id": f"{p['ctx'].trace_id}:{p['ctx'].exec_id}:{index}"}
        self.inflight = ticket
        self.emit("native_action_authorized", ticket)
        return copy.deepcopy(ticket)

    def after(self, ticket, action, *, terminal, observed_state_sha256):
        if type(terminal) is not bool or self.inflight != ticket or self.pending is None:
            raise ValueError("native receipt has no matching authorization")
        if digest(action) != ticket["action_sha256"]:
            self.stop("receipt_action_mismatch")
            raise ValueError("native receipt action mismatch")
        p = self.pending
        trace = ticket["trace_id"]
        self.emit("oeh_receipt", {"trace_id": trace, "executed_action_sha256": digest(action),
                  "native_step": self.step, "observation_sha256": observed_state_sha256,
                  "evidence": "native take_action returned; EE command, not achieved motion"})
        self.emit("action_executed", {"trace_id": trace, "action": action, "sha256": digest(action)})
        self.inflight = None
        self.step += 1
        p["position"] += 1
        if terminal or p["position"] == p["total"]:
            self.pending = None
            if terminal:
                self.stopped = True
            if not p["done"].done():
                p["done"].set_result({"executed_steps": p["position"]})

    def stop(self, reason):
        self.stopped = True
        p, self.pending = self.pending, None
        self.emit("execution_stopped", {"reason": reason, "step": self.step,
                  "unacknowledged_submission": self.inflight is not None})
        self.inflight = None
        if p is not None and not p["done"].done():
            p["done"].set_exception(RuntimeError(reason))
