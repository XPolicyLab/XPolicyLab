#!/bin/bash
set -euo pipefail

# HWM_CogWAM vendors CogWAM under policy/HWM_CogWAM/CogWAM/ (same pattern as
# policy/FastWAM and policy/AHA_WAM), so this script needs no external checkout
# and no network access to a source host.
#
# Installs, in order:
#   1. CogWAM's own pinned requirements (CUDA/torch/flash-attn pinning that this
#      script intentionally does not second-guess),
#   2. CogWAM itself, so `cogwam.eval.robodojo_policy` (the adapter class
#      model.py re-exports) and `cogwam.serve.policy_server` (the GPU backend
#      setup_eval_policy_server.sh launches) resolve,
#   3. XPolicyLab, so `XPolicyLab.policy.HWM_CogWAM.model` and `client_server.ws`
#      resolve.
#
# One conda env hosts both halves of the two-hop server described in README.md:
# the lightweight XPolicyLab-facing bridge and the GPU-heavy CogWAM backend.
#
# Set COGWAM_ROOT to use a different CogWAM checkout instead of the vendored one.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COGWAM_ROOT="${COGWAM_ROOT:-${SCRIPT_DIR}/CogWAM}"

if [[ ! -f "${COGWAM_ROOT}/pyproject.toml" ]]; then
    echo "[install][ERROR] no CogWAM source at ${COGWAM_ROOT}" >&2
    echo "[install][ERROR] expected the vendored copy at ${SCRIPT_DIR}/CogWAM," >&2
    echo "[install][ERROR] or export COGWAM_ROOT=/path/to/your/CogWAM checkout." >&2
    exit 1
fi

echo "[install] CogWAM source: ${COGWAM_ROOT}"

pip install -r "${COGWAM_ROOT}/requirements.txt"
python -m pip install -e "${COGWAM_ROOT}"
python -m pip install -e "${XPL_ROOT}"
