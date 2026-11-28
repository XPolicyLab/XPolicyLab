#!/usr/bin/env bash
# Launch RoboTwin/RoboDojo RL training or evaluation with the XPolicyLab adapter and the vendored RLinf checkout.
# Required environment variables when the selected config needs them:
#   ROBOTWIN_ASSETS_PATH  RoboTwin embodiment assets
#   POLICY_MODEL_PATH     policy checkpoint
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XPL_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
WORKSPACE_ROOT=$(cd "${XPL_ROOT}/.." && pwd)

export RLINF_PATH="${XPL_ROOT}/infra/RLinf/RLinf"
export REPO_PATH="${RLINF_PATH}"
export EMBODIED_PATH="${RLINF_PATH}/examples/embodiment"
export RLINF_EXT_MODULE=XPolicyLab.infra.RLinf.register
export PYTHONPATH="${WORKSPACE_ROOT}:${RLINF_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# A separate simulator checkout may vendor its own XPolicyLab. Expose only
# this package before the runtime so its envs and robotwin resolve together.
if [[ -n "${ROBOTWIN_RUNTIME_PATH:-}" && "${ROBOTWIN_RUNTIME_PATH}" != "${WORKSPACE_ROOT}" ]]; then
  shim="${XPL_ROOT}/.import_shim"
  mkdir -p "$shim"
  ln -sfn "${XPL_ROOT}" "$shim/XPolicyLab"
  export PYTHONPATH="${shim}:${ROBOTWIN_RUNTIME_PATH}:${PYTHONPATH}"
fi
cd "${SCRIPT_DIR}/.."
python -c 'from XPolicyLab.infra.RLinf.core.validation import checked_rlinf_revision; checked_rlinf_revision()'

CONFIG_NAME=robotwin_adjust_bottle_grpo_xpolicylab_pi05
if [[ $# -gt 0 && "$1" != -* ]]; then
  CONFIG_NAME="$1"
  shift
fi

# Hydra config/help inspection must not start a GPU/Ray job.
for arg in "$@"; do
  if [[ "$arg" == "--cfg" || "$arg" == "--help" || "$arg" == "-h" ]]; then
    exec python "${SCRIPT_DIR}/check_config.py" --config-name "${CONFIG_NAME}" "$@"
  fi
done

if [[ -z "${RAY_ADDRESS:-}" ]] && ! ray status >/dev/null 2>&1; then
  ray start --head --port="${RAY_PORT:-6379}" --disable-usage-stats
fi

exec python "${SCRIPT_DIR}/run.py" --config-name "${CONFIG_NAME}" "$@"
