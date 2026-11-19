# XPolicyLab.setup_policy_server imports `XPolicyLab.policy.FlexPi.model.Model`
# directly, so this file only needs to mark the directory as a package.
# Importing .model or .deploy here would either swallow real ImportError bugs
# (the demo_policy try/except: pass pattern) or pull the heavy vendored deps
# into any code that merely walks XPolicyLab.policy.*.
