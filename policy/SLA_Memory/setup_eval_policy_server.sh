#!/bin/bash
set -euo pipefail

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env=${8:-uv}
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

# `uv` selects the project directory set by policy_uv_env_path in deploy.yml; anything else is a project path.
if [[ "${policy_uv_env}" == "uv" ]]; then
    policy_uv_env="$(awk -F': *' '$1 == "policy_uv_env_path" {print $2}' "${yaml_file}")"
fi
[[ "${policy_uv_env}" == /* ]] || policy_uv_env="${SCRIPT_DIR}/${policy_uv_env}"
PYTHON_BIN="${policy_uv_env}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[SERVER][ERROR] uv environment not found: ${policy_uv_env}/.venv. Run: bash ${SCRIPT_DIR}/install.sh" >&2
    exit 1
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, python=${PYTHON_BIN}"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${BENCH_ROOT}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
