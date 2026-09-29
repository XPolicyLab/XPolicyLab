"""XPolicyLab import shim for the repository-owned RoboDojo adapter.

Import the adapter through its installed package path rather than by walking
up from this shim: the repository root is already first on ``sys.path`` (see
``cogwam/eval/launch_bridge.py``), and ``cogwam.eval.robodojo_policy`` has no
namespace collision with anything the simulator image ships.
"""

from cogwam.eval.robodojo_policy import Model

__all__ = ["Model"]
