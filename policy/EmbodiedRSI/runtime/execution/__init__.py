"""The single simulator boundary used by EmbodiedRSI.

Every task is a Gymnasium environment.  The action is always one complete
Python source string; the simulator-specific code only decides how that
source is executed and how the task's observation/reward are produced.
"""

from XPolicyLab.policy.EmbodiedRSI.runtime.execution.gym_env import CODE_CHARSET, MAX_CODE_LENGTH, Environment, StepResult, execute

__all__ = [
    "CODE_CHARSET",
    "MAX_CODE_LENGTH",
    "Environment",
    "StepResult",
    "execute",
]
