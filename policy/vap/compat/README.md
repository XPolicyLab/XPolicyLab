# RoboDojo client transport configuration

**Diagnostic only; do not apply for XPolicyLab-only reproduction.** The current VAP environment-side adapter reads its own `deploy.yml` and configures request/connect timeouts on the existing official client before its policy RPCs. The standard combined entry therefore needs no parent patch. This historical diagnostic patch is retained only to reproduce the older parent-forwarding experiment.

For that historical diagnostic experiment only, from the parent RoboDojo directory:

```bash
git apply --check XPolicyLab/policy/vap/compat/robodojo_client_timeout.patch
git apply XPolicyLab/policy/vap/compat/robodojo_client_timeout.patch
```

For an already patched checkout, verify instead:

```bash
git apply --reverse --check XPolicyLab/policy/vap/compat/robodojo_client_timeout.patch
```

The patch forwards request_timeout_s, max_connect_seconds and ping settings from deploy.yml to the existing WsModelClient. It changes no policy math, observations, actions, IK, scene selection, exception handling or score calculation. The shared XPolicyLab protocol remains unmodified. The validated defaults are 16 steps, shift=1, seed=0, five additional acceleration flags disabled, M16 disabled, and the existing KV cache/CUDA Graph/decode RoPE retained.

This compatibility patch belongs to the parent benchmark, not to the model server. If a newer parent has native forwarding, verify the effective constructor values instead of applying the patch blindly. Batch scheduling and worker watchdogs remain external to the submitted adapter.
