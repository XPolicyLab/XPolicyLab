"""Reviewed primitive-only code-policy example shipped with the adapter."""

# Keep the source in the public baseline module so its digest and validation
# are shared by the local demo and the installed XPolicyLab adapter.
from PhysicalRSI_baselines.robodojo.code_policy_demo import SOURCE, describe

__all__ = ["SOURCE", "describe"]
