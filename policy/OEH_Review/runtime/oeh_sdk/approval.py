"""The approval channel: answering `approval/request` for the whole robot.

When the rule chain returns `Ask`, admission asks whoever holds the approval channel,
and there is exactly one holder per kernel. `approval/attach` is **preemptive**: a
later attach displaces the earlier one, which learns about it through an
`approval/detached {"reason": "replaced"}` notification. That is deliberate — an
operator arriving at a console must not be locked out by a stale connection — but it
means an approver that cares should watch for the notification rather than assume it is
still in charge.

Everything here is fail-closed. If the approver is unreachable, slow, or answers with
something unparseable, the kernel denies: an approval channel that cannot be reached is
worth less than none at all, because the only thing it could do wrong is say yes.

**There is no cancel for an abandoned Ask.** When the proposal behind a pending
`approval/request` goes away — the turn is interrupted, the lease expires, admission's
own `ask_timeout_ms` fires — the kernel does not tell the approver. The request simply
never gets an answer that matters, and a UI built on this must time its own prompts out
rather than wait for a cancellation that is not part of the protocol. (A decision sent
late is not an error either; it is dropped on the kernel side.)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from . import protocol as P
from .rpc import Client

log = logging.getLogger("oeh_sdk.approval")


class Approver:
    """Subclass this and implement :meth:`decide`."""

    #: The default `by` on decisions: who is answering when the decision itself does
    #: not name a person. It shows up in the brain's `Outcome` and in the world log.
    name: str = "oeh-sdk"

    def __init__(self, name: Optional[str] = None) -> None:
        if name is not None:
            self.name = name
        self._client: Optional[Client] = None
        self._wired: Optional[Client] = None

    async def decide(self, request: dict) -> tuple[bool, str]:
        """Answer one `approval/request`.

        `request` is `{ask_id, proposal, spec, reason, trace}`: `reason` is why the
        chain wants a human, and `trace` is each rule's incoming and outgoing verdict —
        which is what makes a console able to show *why* it is being asked rather than
        just *that* it is.

        Returns `(allow, reason)`. The default denies everything, because a subclass
        that forgets to override this must not become a rubber stamp.
        """
        return (False, "no approver implemented")

    async def attach(self, client: Client, timeout: Optional[float] = None) -> None:
        """Take over the approval channel for this kernel.

        The handler is installed before the call: the kernel may route an Ask the
        instant the attach lands, and a `approval/request` arriving before the handler
        exists would come back METHOD_NOT_FOUND — which admission reads as a broken
        channel and turns into a denial.

        Re-attaching on the same connection subscribes once, not twice:
        `on_notification` appends, so a second `approval/detached` subscription would
        fire :meth:`on_detached` twice for one displacement. (`on_request` replaces, so
        that half was already idempotent.)
        """
        self._client = client
        client.on_request(P.APPROVAL_REQUEST, self._on_request)
        if self._wired is not client:
            client.on_notification(P.APPROVAL_DETACHED, self._on_detached)
            self._wired = client
        await client.call(P.APPROVAL_ATTACH, {"name": self.name}, timeout=timeout)

    async def detach(self) -> None:
        """Give the channel back. Dropping the connection does the same thing."""
        client = self._client
        if client is None or client.closed.is_set():
            return
        await client.call(P.APPROVAL_DETACH, {})

    async def on_detached(self, reason: str) -> None:
        """Another connection took the channel (`reason == "replaced"`). Default: log it."""
        log.warning("approval channel detached: %s", reason)

    # -- plumbing ----------------------------------------------------------------------

    async def _on_request(self, params: dict) -> dict:
        req = P.ApprovalRequest.from_dict(params)
        allow, reason = await self.decide(params)
        log.debug("ask %s -> %s (%s)", req.ask_id, "allow" if allow else "deny", reason)
        return P.ApprovalDecision(
            decision="allow" if allow else "deny",
            by=self.name,
            reason=reason,
        ).to_dict()

    def _on_detached(self, params: dict) -> Any:
        return self.on_detached(str(params.get("reason", "")))
