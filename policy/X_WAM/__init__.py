# XPolicyLab.setup_policy_server imports `XPolicyLab.policy.X_WAM.model.Model`
# directly, so this file only needs to mark the directory as a package.
# Importing .model or .deploy here would either swallow real ImportError bugs
# or pull heavy upstream deps into any code that just walks XPolicyLab.policy.*,
# which we want to avoid.
