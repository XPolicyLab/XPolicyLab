#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <policy_conda_env> <port> [host]" >&2
  exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_conda_env=$9
policy_server_port=${10}
policy_server_host=${11:-${POLICY_SERVER_HOST:-localhost}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT_DIR="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"
INNER_DIR="${SCRIPT_DIR}/giga_world_policy"
policy_name="$(basename "${SCRIPT_DIR}")"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${XPL_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, host=${policy_server_host}, port=${policy_server_port}, action_dim=${action_dim}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export PYTHONPATH="${INNER_DIR}/src:${INNER_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"

exec env \
  PYTHONWARNINGS=ignore::UserWarning \
  CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
  PYTHONPATH="${PYTHONPATH}" \
  python "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
      host="${policy_server_host}" \
      port="${policy_server_port}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      expert_data_num="${expert_data_num}" \
      seed="${seed}" \
      policy_name="${policy_name}" \
      action_type="${action_type}" \
      action_dim="${action_dim}"
