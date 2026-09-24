#!/bin/bash
set -euo pipefail

# HWM_CogWAM has no separate package metadata; install XPolicyLab so imports
# like `XPolicyLab.policy.HWM_CogWAM.model` and `client_server.ws` resolve in
# the policy environment, then install CogWAM itself (untouched, from its own
# checkout) so `cogwam.eval.robodojo_policy` (the adapter class this policy
# re-exports) and `cogwam.serve.policy_server` (the backend process
# setup_eval_policy_server.sh launches) resolve too.
#
# One conda env hosts both halves of the two-hop server described in
# README.md: the lightweight XPolicyLab-facing bridge and the GPU-heavy
# CogWAM backend. CogWAM's own requirements.txt covers the CUDA/torch/
# flash-attn pinning this script intentionally does not second-guess.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COGWAM_ROOT="${COGWAM_ROOT:?export COGWAM_ROOT=/path/to/CogWAM checkout before running install.sh}"

pip install -r "${COGWAM_ROOT}/requirements.txt"
python -m pip install -e "${COGWAM_ROOT}"
python -m pip install -e "${XPL_ROOT}"
