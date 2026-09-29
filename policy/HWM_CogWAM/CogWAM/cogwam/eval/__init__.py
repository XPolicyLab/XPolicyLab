"""RoboDojo evaluation client for CogWAM checkpoints.

The simulator side of the stack: it owns every per-environment fact (action
chunk cache, semantic memory, cached subtask, step counter) and talks to one or
more stateless CogWAM policy servers over WebSocket.
"""

from __future__ import annotations

__all__ = ["DIMENSIONS"]


def __getattr__(name: str):
    # ``robodojo_policy`` imports XPolicyLab, which only exists inside the
    # simulator checkout, so it must not be pulled in by ``import cogwam.eval``.
    if name == "DIMENSIONS":
        from cogwam.eval.summarize import DIMENSIONS

        return DIMENSIONS
    raise AttributeError(name)
