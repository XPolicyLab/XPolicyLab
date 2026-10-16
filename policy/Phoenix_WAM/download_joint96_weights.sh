#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${XPL_ROOT}/..:${PYTHONPATH:-}"
WEIGHTS_DIR="${1:-${SCRIPT_DIR}/checkpoints/phoenix-wam-joint96-step40000}"
if [[ $# -gt 0 ]]; then shift; fi
exec "${PYTHON:-python}" -m XPolicyLab.policy.Phoenix_WAM.joint96_assets \
  "${WEIGHTS_DIR}" --download "$@"
