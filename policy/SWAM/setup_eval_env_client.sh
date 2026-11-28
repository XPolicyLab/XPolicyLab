#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?seed}
env_gpu_id=${7:?env_gpu_id}
eval_env_conda_env=${8:?eval_env_conda_env}
additional_info=${9:-}
policy_server_port=${10:?policy_server_port}
policy_server_ip=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"


task_config="${ROBOTWIN_TASK_CONFIG:-demo_clean}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${env_gpu_id}"
export PYTHONPATH="${XPL_ROOT}/..:${ROBOTWIN_ROOT:-}:${XPL_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"

if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
  PYTHON_BIN="${eval_env_conda_env}/bin/python"
elif [[ -x "${eval_env_conda_env}" && ! -d "${eval_env_conda_env}" ]]; then
  PYTHON_BIN="${eval_env_conda_env}"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${eval_env_conda_env}"
  PYTHON_BIN="$(command -v python)"
fi

if [[ "${EVAL_ENV_TYPE:-sim}" == debug ]]; then
  debug_args=()
  if [[ -n "${SWAM_EVALUATION_ID:-}" ]]; then
    debug_args+=(--evaluation_id "${SWAM_EVALUATION_ID}")
  fi
  exec "${PYTHON_BIN}" -u "${XPL_ROOT}/utils/debug_env_client.py" \
    --bench_name "${bench_name}" --task_name "${task_name}" \
    --policy_name SWAM --env_cfg_type "${env_cfg_type}" \
    --action_type "${action_type}" --protocol ws \
    --host "${policy_server_ip}" --port "${policy_server_port}" \
    --eval_batch false "${debug_args[@]}"
fi
: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to a RoboTwin 2.0 checkout with scripts/eval_policy.sh}"
[[ -f "${ROBOTWIN_ROOT}/scripts/eval_policy_xpolicylab.py" ]] || {
  echo "This adapter requires the current XPolicyLab-native RoboTwin evaluation entry." >&2
  exit 1
}
# Use the selected interpreter through the official shell entry.
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export XPOLICYLAB_ROOT="${XPL_ROOT}"
additional_info="${additional_info:+${additional_info},}action_type=${action_type},xpolicylab_root=${XPL_ROOT}"
if [[ -n "${SWAM_EVALUATION_ID:-}" ]]; then
  additional_info+=",evaluation_id=${SWAM_EVALUATION_ID}"
fi
exec bash "${ROBOTWIN_ROOT}/scripts/eval_policy.sh" \
  --bench_name "${bench_name}" --task_name "${task_name}" \
  --policy_name SWAM --env_cfg_type "${env_cfg_type}" \
  --host "${policy_server_ip}" --port "${policy_server_port}" \
  --protocol ws --eval_batch false --root_dir "${ROBOTWIN_ROOT}" \
  --device_id 0 --seed "${seed}" --task_config "${task_config}" \
  --test_num "${ROBOTWIN_TEST_NUM:-100}" --additional_info "${additional_info}"
