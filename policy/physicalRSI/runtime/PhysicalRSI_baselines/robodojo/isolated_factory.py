"""Neutral factory name for isolated code-policy execution."""

from .code_policy_factory import code_policy_factory, verify_spec


def isolated_factory(spec, *, pool=None):
    return code_policy_factory(spec, pool=pool)
