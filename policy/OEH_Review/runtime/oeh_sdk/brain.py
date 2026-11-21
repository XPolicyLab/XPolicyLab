"""Brains: `brain/attach`, and the turn loop the kernel drives.

A cross-process brain is one connection plus one `brain/attach`. From then on the
connection *owns* that brain: only it gets the turns, only it can propose inside them,
and dropping the socket detaches it. The kernel drives; this SDK does not poll.

Three notifications arrive, and they are not variations of each other:

* **`brain/turn`** — you have been woken. A fresh `turn` number, a `trace_id` that ties
  everything you do now to the world log, and an `observation`. Run
  :meth:`Brain.on_turn` and finish with `turn/done`.
* **`brain/steer`** — new information *for the turn already running*. No new turn, no
  new handle; the running `on_turn` keeps its own.
* **`brain/interrupt`** — drop what you are doing. The kernel has already finished that
  turn on its side; a new `brain/turn` may follow immediately.

A directed `turn/steer` from a console lands as whichever of these fits: on an idle
brain it *wakes* it, so you receive `brain/turn` with `wake_reason.kind ==
"brain/steer"`; on a running brain you receive `brain/steer`. Handle both.

Every turn runs in its own task, so a slow `on_turn` cannot stall the connection's read
loop and therefore cannot stall its own `turn/propose` replies.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from typing import Any, Optional

from . import protocol as P
from .rpc import Client, RpcError

log = logging.getLogger("oeh_sdk.brain")

#: How long to wait before the single `brain/attach` retry. Long enough for an
#: in-flight detach on the same connection to land, short enough to be invisible.
ATTACH_RETRY_S = 0.05

#: The two REFUSED reasons that mean "an attach and a detach crossed on this
#: connection", not "this brain belongs to someone else". Both are transient by
#: construction (see `oeh_rpc::handlers::brain`), so the SDK retries once.
_RETRYABLE_ATTACH = ("attach in progress", "detached during attach")


class TurnCtx:
    """One turn. Everything a brain does happens through one of these."""

    def __init__(self, client: Client, brain_id: str, turn: int, trace_id: str, observation: Any):
        self.brain_id = brain_id
        self.turn = turn
        #: Ties this turn's `brain/*` and `act/*` lines together in the world log.
        self.trace_id = trace_id
        #: `{t_kernel, wake_reason: {kind, seq, action}, state, events,
        #: source_seq_range, truncated}`. `state["fsm"]` is SCREAMING_SNAKE_CASE.
        self.observation = observation
        #: The id of the most recent proposal, for matching against `act/*` log lines.
        self.last_proposal_id: Optional[str] = None
        self._client = client

    async def propose(
        self, capability: str, params: Optional[dict] = None, timeout: Optional[float] = None
    ) -> dict:
        """Propose an action and wait for how it ended.

        Returns the `Outcome`, internally tagged: `{"outcome": "executed", "result":
        ...}`, or `refused` / `failed` / `cancelled` / `preempted` with a `reason`. A
        refusal is a normal answer, not an error — the rule chain saying no is the
        system working.

        This waits for the whole admission round trip: rules, approval if the chain
        asks for one, and the execution itself.
        """
        return await self._propose(
            P.TURN_PROPOSE,
            {
                "brain_id": self.brain_id,
                "turn": self.turn,
                "capability": capability,
                "params": {} if params is None else params,
            },
            timeout,
        )

    async def say(self, text: str, timeout: Optional[float] = None) -> dict:
        """Speak. Not a shortcut: `say` is a proposal like any other and goes through
        the same rule chain (it is a built-in capability, class `speech`, risk `none`)."""
        return await self._propose(
            P.TURN_SAY,
            {"brain_id": self.brain_id, "turn": self.turn, "text": text},
            timeout,
        )

    async def ask_human(self, question: str, timeout: Optional[float] = None) -> dict:
        """Ask a person. Also a proposal; with no approval channel attached the outcome
        is `refused`, which is the fail-closed answer, not an error."""
        return await self._propose(
            P.TURN_ASK_HUMAN,
            {"brain_id": self.brain_id, "turn": self.turn, "question": question},
            timeout,
        )

    async def done(self, reason: str = "ok") -> None:
        """End the turn. Calling it twice is NOT_ATTACHED — that is how you can tell
        "I just closed it" from "it was already gone"."""
        await self._client.call(
            P.TURN_DONE,
            {"brain_id": self.brain_id, "turn": self.turn, "reason": reason},
        )

    async def _propose(self, method: str, params: dict, timeout: Optional[float]) -> dict:
        raw = await self._client.call(method, params, timeout=timeout)
        r = P.TurnProposeResult.from_dict(raw)
        self.last_proposal_id = r.proposal_id
        return r.outcome if isinstance(r.outcome, dict) else {"outcome": r.outcome}


class Brain:
    """Subclass this, set `brain_id` and `wake_policy`, override :meth:`on_turn`."""

    brain_id: str = ""
    #: `{"default": <action>, "rules": {<kind>: <action>}, "budget": {...}}` where an
    #: action is `ignore` / `accumulate` / `wake` / `steer` / `interrupt`. The kernel
    #: rejects an unknown key rather than ignoring it, so a typo is loud.
    wake_policy: dict = {}

    def __init__(self, brain_id: Optional[str] = None, wake_policy: Optional[dict] = None):
        if brain_id is not None:
            self.brain_id = brain_id
        # Always a fresh dict, even when falling back to the class attribute: a
        # class-level default is one object shared by every instance, and a brain that
        # adjusts its own policy would otherwise adjust everyone's. Deep, because the
        # interesting part of a policy is the nested `rules` and `budget` maps — a
        # shallow copy leaves exactly those shared.
        self.wake_policy = copy.deepcopy(
            self.wake_policy if wake_policy is None else wake_policy
        )
        self._client: Optional[Client] = None
        self._turns: dict[int, TurnCtx] = {}
        self._attached = False
        self._wired: Optional[Client] = None

    # -- to override -------------------------------------------------------------------

    async def on_turn(self, turn: TurnCtx) -> None:
        """The brain. The default ends the turn immediately, which is a valid brain:
        it stays attached, is woken, and decides to do nothing.

        End every turn — with `turn.done()`, or by raising, which this class turns into
        `turn/done(reason="error: <Type>")`. A turn nobody ends is held open until the
        kernel's `turn_timeout_ms` (30 s by default), and for that whole time the brain
        is `running` and gets no new turns.
        """
        await turn.done()

    async def on_steer(self, observation: dict) -> None:
        """New information for the turn already running. Default: ignore it."""

    async def on_interrupt(self) -> None:
        """The running turn has been cut short by the kernel. Default: ignore it."""

    # -- lifecycle ---------------------------------------------------------------------

    async def attach(self, client: Client, timeout: Optional[float] = None) -> int:
        """Attach to the kernel; returns the brain's log cursor.

        Retries exactly once, after a beat, on the two REFUSED reasons that mean an
        attach crossed a detach on this same connection (2026-09-21 ruling). Both are
        states the kernel is already leaving; a second, permanent-looking failure is a
        real conflict and is raised.

        The `cursor` is read back live rather than pinned at the moment of attach, so
        it is *not earlier* than the mount point — it can already have moved past it if
        the kernel woke the brain in between. Safe as a "where do I start reading"
        answer; not an exact mount point.

        Re-attaching (after a `detach`, or after a supervisor reconnects) does **not**
        subscribe a second time. `Client.on_notification` appends, so a second
        subscription would run `on_turn` twice per wake: two `turn/propose` calls for
        one intent — the robot does the thing twice — and the loser's `turn/done` comes
        back NOT_ATTACHED, which this class treats as ordinary. The duplication would
        be entirely silent.
        """
        self._client = client
        if self._wired is not client:
            client.on_notification(P.BRAIN_TURN, self._on_turn_notice)
            client.on_notification(P.BRAIN_STEER, self._on_steer_notice)
            client.on_notification(P.BRAIN_INTERRUPT, self._on_interrupt_notice)
            self._wired = client
        params = {"brain_id": self.brain_id, "wake_policy": self.wake_policy or {}}
        try:
            raw = await client.call(P.BRAIN_ATTACH, params, timeout=timeout)
        except RpcError as e:
            if e.code != P.code.REFUSED or not any(m in e.message for m in _RETRYABLE_ATTACH):
                raise
            log.debug("brain/attach refused (%s); retrying once", e.message)
            await asyncio.sleep(ATTACH_RETRY_S)
            raw = await client.call(P.BRAIN_ATTACH, params, timeout=timeout)
        self._attached = True
        return P.BrainAttachResult.from_dict(raw).cursor

    async def detach(self) -> None:
        """Give the brain back. Dropping the connection does the same thing."""
        client = self._client
        self._attached = False
        self._turns.clear()
        if client is None or client.closed.is_set():
            return
        await client.call(P.BRAIN_DETACH, {"brain_id": self.brain_id})

    # -- notification plumbing ---------------------------------------------------------

    async def _on_turn_notice(self, params: dict) -> None:
        n = P.BrainTurn.from_dict(params)
        if n.brain_id != self.brain_id or not self._attached:
            return
        assert self._client is not None
        ctx = TurnCtx(self._client, n.brain_id, n.turn, n.trace_id, n.observation)
        # A turn supersedes the previous one: the kernel only ever runs one at a time,
        # and a turn that ended by interrupt or timeout never saw `turn/done`.
        self._turns = {n.turn: ctx}
        try:
            await self.on_turn(ctx)
        except RpcError as e:
            # NOT_ATTACHED here means the turn is already over (interrupted, or the
            # kernel's turn sweep collected it). That is ordinary, not a brain bug,
            # and there is no turn left to close.
            if e.code == P.code.NOT_ATTACHED:
                log.debug("turn %d of `%s` was already over: %s", n.turn, n.brain_id, e.message)
            else:
                log.exception("on_turn for `%s` failed", n.brain_id)
                await self._abandon(ctx, e)
        except Exception as e:  # noqa: BLE001
            log.exception("on_turn for `%s` raised", n.brain_id)
            await self._abandon(ctx, e)

    async def _abandon(self, ctx: TurnCtx, exc: BaseException) -> None:
        """Close a turn whose `on_turn` blew up.

        Not tidiness: a turn nobody ends is ended by the kernel's `turn_timeout_ms`,
        **30 seconds** by default. For those 30 s the brain reports `running`, no new
        turn is delivered, and the log gets a `brain/error{reason: "turn_timeout"}`
        that says nothing about the exception. Ending it here costs one call and names
        the cause in the session log.

        The reason carries the exception *type* and not its message: `reason` lands in
        the world log, and an exception message can contain anything the failure
        happened to be holding — a path, a payload, a credential.
        """
        with contextlib.suppress(Exception):
            await ctx.done(f"error: {type(exc).__name__}")

    async def _on_steer_notice(self, params: dict) -> None:
        n = P.BrainSteerNotice.from_dict(params)
        if n.brain_id != self.brain_id or not self._attached:
            return
        try:
            await self.on_steer(n.observation if isinstance(n.observation, dict) else {})
        except Exception:  # noqa: BLE001
            log.exception("on_steer for `%s` raised", n.brain_id)

    async def _on_interrupt_notice(self, params: dict) -> None:
        if params.get("brain_id") != self.brain_id or not self._attached:
            return
        try:
            await self.on_interrupt()
        except Exception:  # noqa: BLE001
            log.exception("on_interrupt for `%s` raised", self.brain_id)
