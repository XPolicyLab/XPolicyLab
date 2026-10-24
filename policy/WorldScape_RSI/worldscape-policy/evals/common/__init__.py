"""Runner-independent evaluation contracts.

Rollout classes are resolved lazily because the platform-neutral runner imports
the protocols in this package. Eagerly importing it here would create a package
initialization cycle.
"""

from evals.common.artifacts import EvaluationArtifactWriter
from evals.common.protocols import RolloutInput


def __getattr__(name: str):
    if name in {"RolloutConfig", "RolloutResult", "RolloutRunner"}:
        from worldscape_policy.rollout import runner

        return getattr(runner, name)
    raise AttributeError(name)

__all__ = [
    "EvaluationArtifactWriter",
    "RolloutConfig",
    "RolloutInput",
    "RolloutResult",
    "RolloutRunner",
]
