"""Expose the hyphenated policy package through a valid Agent factory name."""

from importlib import import_module


def make_agent(case_dir, *, config):
    """Use the same package module and agent classes as the policy server."""
    module = import_module("XPolicyLab.policy.SenseNova-RoboRSI.hybrid_agent")
    return module.make_agent(case_dir, config=config)
