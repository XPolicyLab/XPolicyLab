# setup_policy_server.py imports `XPolicyLab.policy.HWM_CogWAM.model.Model`
# directly, so this file only needs to mark the directory as a package.
# Importing .model or .deploy here would either swallow real ImportError bugs
# (e.g. CogWAM not installed) or pull heavy upstream deps into any code that
# just walks XPolicyLab.policy.*, which we want to avoid. Same rationale as
# policy/X_WAM/__init__.py.
