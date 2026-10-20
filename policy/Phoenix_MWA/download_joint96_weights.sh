#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${XPL_ROOT}/..:${PYTHONPATH:-}"
WEIGHTS_DIR="${1:-${SCRIPT_DIR}/checkpoints/phoenix-wam-h5-nocrop-init256013-step40000-20261019}"
if [[ $# -gt 0 ]]; then shift; fi
exec "${PYTHON:-python}" -m XPolicyLab.policy.Phoenix_MWA.joint96_assets \
  "${WEIGHTS_DIR}" --download "$@"
