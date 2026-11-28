#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?seed}
policy_gpu_id=${7:?policy_gpu_id}
policy_conda_env=${8:?policy_conda_env}
policy_server_port=${9:?policy_server_port}
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
YAML_FILE="${SCRIPT_DIR}/deploy.yml"

# Model resolves source/checkpoint paths from deploy.yml, then environment variables.

if [[ -x "${policy_conda_env}/bin/python" ]]; then
  PYTHON_BIN="${policy_conda_env}/bin/python"
elif [[ -x "${policy_conda_env}" && ! -d "${policy_conda_env}" ]]; then
  PYTHON_BIN="${policy_conda_env}"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${policy_conda_env}"
  PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
export PYTHONWARNINGS="ignore::UserWarning"
export SWAM_DIST_INIT_FILE="/tmp/swam_xpolicylab_dist_${$}"

exec "${PYTHON_BIN}" -u "${XPL_ROOT}/setup_policy_server.py" \
  --config_path "${YAML_FILE}" \
  --overrides \
    port="${policy_server_port}" \
    host="${policy_server_host}" \
    bench_name="${bench_name}" \
    task_name="${task_name}" \
    ckpt_name="${ckpt_name}" \
    env_cfg_type="${env_cfg_type}" \
    action_type="${action_type}" \
    seed="${seed}" \
    policy_name="SWAM"
