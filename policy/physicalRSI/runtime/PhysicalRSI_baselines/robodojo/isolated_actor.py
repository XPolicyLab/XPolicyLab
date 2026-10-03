"""Neutral public name for one isolated code-policy actor."""

from .code_policy_actor import CodePolicyActor


class IsolatedActor(CodePolicyActor):
    """Compatibility-preserving actor name used by the skill_selection runtime."""
