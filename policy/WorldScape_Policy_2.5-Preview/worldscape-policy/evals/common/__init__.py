"""Evaluation contracts with lazy rollout imports."""

from evals.common.protocols import RolloutInput


def __getattr__(name: str):
    if name == "EvaluationArtifactWriter":
        from evals.common.artifacts import EvaluationArtifactWriter
        return EvaluationArtifactWriter
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
