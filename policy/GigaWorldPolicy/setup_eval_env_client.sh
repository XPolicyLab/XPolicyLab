#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 10 ]]; then
  echo "Usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <env_gpu_id> <eval_env_conda_env> <additional_info> <port> [policy_server_ip]" >&2
  exit 1
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9
policy_server_port=${10}
policy_server_ip=${11:-${POLICY_SERVER_IP:-localhost}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROOT_DIR="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"
policy_name="$(basename "${SCRIPT_DIR}")"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

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
  "${policy_server_ip}"
