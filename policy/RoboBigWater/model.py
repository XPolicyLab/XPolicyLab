"""XPolicyLab adapter of RoboBigWater: the policy is a coding agent behind the robo interface.

The server, the agent container and the task tools live in this directory, so it is put on sys.path here and
`roboshell` imports resolve wherever the policy server was started from.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from XPolicyLab.model_template import ModelTemplate  # noqa: E402
from roboshell.server.bridge import Model as _BridgeModel  # noqa: E402


class Model(_BridgeModel, ModelTemplate):
    """Bridge executor with the XPolicyLab template hooks (prepare_case, on_trial_end) inherited."""
