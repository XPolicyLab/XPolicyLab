"""The single simulator boundary used by EmbodiedEvo.

Every task is a Gymnasium environment.  The action is always one complete
Python source string; the simulator-specific code only decides how that
source is executed and how the task's observation/reward are produced.
"""

from .base import (
    CODE_CHARSET,
    MAX_CODE_LENGTH,
    Environment,
    StepResult,
    execute,
)

__all__ = [
    "CODE_CHARSET",
    "MAX_CODE_LENGTH",
    "Environment",
    "StepResult",
    "execute",
]
