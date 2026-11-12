#!/usr/bin/env bash
# Separate RLinf v0.4 native PyTorch SFT entry; train.sh remains official OpenPI.
set -euo pipefail
if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id> [--dry-run|--compute-norm-stats] [key=value ...]" >&2
  exit 1
fi
XPL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
POLICY_DIR="${XPL_ROOT}/policy/Pi_05"
WORKSPACE_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
export RLINF_PATH="${XPL_ROOT}/infra/RLinf/RLinf"
export REPO_PATH="${RLINF_PATH}"
export PYTHONPATH="${WORKSPACE_ROOT}:${RLINF_PATH}:${POLICY_DIR}/openpi/src:${POLICY_DIR}/openpi/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
# OpenPI is used for CPU data transforms only; it must not reserve CUDA memory.
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES="$6"
# Use the shared training dimension helper (no simulator dependencies).
export PI05_SFT_ACTION_DIM="$(bash "${XPL_ROOT}/utils/get_action_dim.sh" "${WORKSPACE_ROOT}" "$3")"
exec "${RL_PYTHON:-python}" -m XPolicyLab.policy.Pi_05.rlinf.sft "$@"
