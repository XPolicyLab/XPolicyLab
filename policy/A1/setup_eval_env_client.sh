#!/bin/bash
set -e

dataset_name=$1
task_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
env_gpu_id=$6
eval_env_conda_env=$7
additional_info=$8
policy_server_port=$9
policy_server_ip=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[CLIENT] policy=A1, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${dataset_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "A1" \
    "${additional_info}" \
    "${ROOT_DIR}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
