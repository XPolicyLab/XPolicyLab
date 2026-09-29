"""CogWAM inference serving: a stateless WebSocket policy server.

All per-environment state (semantic memory, cached subtask, action-chunk
scheduling) lives in the evaluation client, never here.

``protocol`` is importable on its own: the RoboDojo client runs in the Isaac
environment, which has neither the training stack nor ``cogwam.data``, and it
only needs the WebSocket client. Everything heavier is resolved lazily.
"""

from __future__ import annotations

from cogwam.serve.protocol import WebsocketClientPolicy, WebsocketPolicyServer

__all__ = [
    "PolicyNormProcessor",
    "PolicyServerWrapper",
    "WebsocketClientPolicy",
    "WebsocketPolicyServer",
    "validate_artifact_directory",
]

_LAZY = {"PolicyNormProcessor", "PolicyServerWrapper", "validate_artifact_directory"}


def __getattr__(name: str):
    if name in _LAZY:
        from cogwam.serve import policy_wrapper

        return getattr(policy_wrapper, name)
    raise AttributeError(name)
