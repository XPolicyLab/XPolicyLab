"""XPolicyLab adapter for CogWAM (policy name: HWM_CogWAM).

This is deliberately a one-line re-export. All inference logic — checkpoint
loading, tri-view camera compositing, state normalization, flow-matching
action generation, chunk/replan bookkeeping and event-memory semantic-state
tracking — lives in CogWAM's own ``cogwam.eval.robodojo_policy.Model``,
imported here unmodified. That class already implements the exact
``ModelTemplate`` contract XPolicyLab requires (it is written against
``XPolicyLab.model_template.ModelTemplate`` directly): see
``cogwam/eval/robodojo_policy.py`` in the CogWAM checkout for the
implementation and for the chunk/replan and event-memory protocol.

``Model`` does not load any weights itself — it is a WebSocket client to the
GPU-heavy ``cogwam.serve.policy_server`` process that
``setup_eval_policy_server.sh`` starts before this policy server comes up.
See this policy's README.md for the two-hop architecture.

CogWAM must be importable in the policy environment; ``install.sh`` does that
with ``pip install -e``.
"""

from __future__ import annotations

from cogwam.eval.robodojo_policy import Model

__all__ = ["Model"]
