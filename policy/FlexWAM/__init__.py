# XPolicyLab.setup_policy_server imports `XPolicyLab.policy.FlexWAM.model.Model`
# directly, so this file only needs to mark the directory as a package.
# Keep it import-free: pulling .model here would drag torch/FlexWAM into any
# code that merely walks XPolicyLab.policy.*.
