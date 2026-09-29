"""XPolicyLab adapter for CogWAM (policy name: HWM_CogWAM).

This is deliberately a one-line re-export. All inference logic — checkpoint
loading, tri-view camera compositing, state normalization, flow-matching
action generation, chunk/replan bookkeeping and event-memory semantic-state
tracking — lives in CogWAM's own ``cogwam.eval.robodojo_policy.Model``,
imported here unmodified. That class already implements the exact
``ModelTemplate`` contract XPolicyLab requires (it is written against
``XPolicyLab.model_template.ModelTemplate`` directly): see
``CogWAM/cogwam/eval/robodojo_policy.py`` in this directory for the
implementation and for the chunk/replan and event-memory protocol.

``Model`` does not load any weights itself — it is a WebSocket client to the
GPU-heavy ``cogwam.serve.policy_server`` process that
``setup_eval_policy_server.sh`` starts before this policy server comes up.
See this policy's README.md for the two-hop architecture.

CogWAM is vendored under ``policy/HWM_CogWAM/CogWAM/`` and installed by
``install.sh``. The ``sys.path`` fallback below only fires when it has not been
installed into the active environment, so the adapter still imports from a bare
checkout (same approach as ``policy/FastWAM/model.py``).
"""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent
COGWAM_ROOT = POLICY_DIR / "CogWAM"

if find_spec("cogwam") is None and (COGWAM_ROOT / "cogwam" / "__init__.py").is_file():
    sys.path.insert(0, str(COGWAM_ROOT))

from cogwam.eval.robodojo_policy import Model  # noqa: E402

__all__ = ["Model"]
