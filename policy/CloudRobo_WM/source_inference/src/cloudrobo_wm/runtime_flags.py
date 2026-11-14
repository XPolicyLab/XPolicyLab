"""Process-local inference setup; simulator processes keep their own settings."""
from __future__ import annotations
import os

POLICY_ENVIRONMENT={
    'PYTHONHASHSEED':'42',
    'CUBLAS_WORKSPACE_CONFIG':':4096:8',
    'OMP_NUM_THREADS':'4',
    'OPENBLAS_NUM_THREADS':'1',
}
_POLICY_CONFIGURED=False

def require_policy_environment():
    mismatches=[name for name,value in POLICY_ENVIRONMENT.items() if os.environ.get(name)!=value]
    if mismatches:
        raise RuntimeError('Start the policy process with the required inference environment: '+', '.join(mismatches))

def configure_policy_runtime():
    global _POLICY_CONFIGURED
    require_policy_environment()
    import torch
    if _POLICY_CONFIGURED:
        if torch.get_num_threads()!=4 or torch.initial_seed()!=42 or not torch.backends.cuda.cudnn_sdp_enabled():
            raise RuntimeError('Policy inference settings changed after initialization')
        return
    if torch.cuda.is_initialized():
        raise RuntimeError('Policy inference settings must be applied before device initialization')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.enable_cudnn_sdp(True)
    _POLICY_CONFIGURED=True
    # Other backend switches retain their existing values; no unverified
    # algorithm, precision, or simulator setting is silently substituted.
