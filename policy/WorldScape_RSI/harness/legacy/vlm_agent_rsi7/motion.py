"""The motion envelope: what one turn looks like on the wire.

Every turn produces exactly one envelope, and its ``kind`` says what the
simulation client does with it:

``sequence``
    Up to a few discrete commands (see :mod:`commands`), executed one after
    another by the simulation client, each as one whole motion resolved from
    the live pose at that moment (:mod:`main_route`).

``stop``
    The policy is done or has given up, and the episode loop should end.

The simulation client dispatches on ``kind`` rather than on its own copy of the
configuration: the two processes read the same ``deploy.yml``, but if they ever
disagree, a missing branch fails loudly instead of quietly executing a motion
the policy did not intend.

Only the standard library is imported here: this module is loaded by the policy
server, by the simulation client, and by the offline tests.
"""

from __future__ import annotations

from typing import Any

KIND_SEQUENCE = "sequence"
KIND_STOP = "stop"
KINDS = (KIND_SEQUENCE, KIND_STOP)


def sequence(commands: list[dict], plan: dict, options: dict | None = None) -> dict:
    """Envelope for a turn of discrete commands the simulation client executes in order.

    ``commands`` are the wire forms of :class:`commands.Command`; ``options``
    carries the execution knobs the client needs (hold and gripper step counts,
    the fingertip offset, the planner step caps).
    """
    return {"kind": KIND_SEQUENCE, "commands": list(commands), "plan": plan, "options": dict(options or {})}


def stop(reason: str | None = None) -> dict:
    """Envelope that ends the episode loop."""
    return {"kind": KIND_STOP, "reason": reason or "policy_stop"}


def normalize(result: Any) -> dict:
    """Coerce whatever ``get_action`` returned into an envelope; anything else stops the episode."""
    if not isinstance(result, dict):
        return stop(f"policy returned {type(result).__name__}, expected a motion envelope")
    kind = result.get("kind")
    if kind not in KINDS:
        return stop(f"policy returned unknown motion kind {kind!r}")
    envelope = dict(result)
    if kind == KIND_SEQUENCE:
        if not isinstance(envelope.get("commands"), list):
            return stop("sequence envelope carries no command list")
        envelope["commands"] = list(envelope["commands"])
        envelope["options"] = dict(envelope.get("options") or {})
        envelope["plan"] = dict(envelope.get("plan") or {})
    return envelope
