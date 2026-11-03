#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -x "${policy_conda_env}/bin/python" ]]; then
  policy_python="${policy_conda_env}/bin/python"
else
  policy_python="${XPL_ROOT}/../.conda/envs/${policy_conda_env}/bin/python"
fi
[[ -x "${policy_python}" ]] || {
  echo "[error] policy Python is not executable: ${policy_python}" >&2
  exit 1
}

exec env CUDA_VISIBLE_DEVICES="${policy_gpu_id}" PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}:${PYTHONPATH:-}" \
  "${policy_python}" "${XPL_ROOT}/setup_policy_server.py" \
    --config_path "${DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}" \
    --overrides \
      port="${policy_server_port}" \
      host="${policy_server_host}" \
      bench_name="${bench_name}" \
      task_name="${task_name}" \
      ckpt_name="${ckpt_name}" \
      env_cfg_type="${env_cfg_type}" \
      seed="${seed}" \
      policy_name=go3_agent_preview \
      action_type="${action_type}"
