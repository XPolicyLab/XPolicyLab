#!/bin/bash
set -euo pipefail

if [[ $# -lt 11 || $# -gt 12 ]]; then
    echo "Usage: bash setup_eval_env_client.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <env_gpu_id> <eval_env_conda_env> <additional_info> <policy_server_port> [policy_server_host]"
    exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
env_gpu_id=$8
eval_env_conda_env=$9
additional_info=${10}
policy_server_port=${11}
policy_server_host=${12:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, expert_data_num=${expert_data_num}, server=${policy_server_host}:${policy_server_port}"

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${ROOT_DIR}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_host}"
