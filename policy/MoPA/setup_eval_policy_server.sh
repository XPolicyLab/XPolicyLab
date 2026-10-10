#!/bin/bash
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

source "${SCRIPT_DIR}/resolve_python.sh"
PY="$(resolve_python "${policy_conda_env}")"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}"
echo "[SERVER] python=${PY}"

overrides=(
    "port=${policy_server_port}"
    "host=${policy_server_host}"
    "bench_name=${bench_name}"
    "task_name=${task_name}"
    "ckpt_name=${ckpt_name}"
    "env_cfg_type=${env_cfg_type}"
    "seed=${seed}"
    "policy_name=${policy_name}"
    "action_type=${action_type}"
)
if [[ -n "${MOPA_CHECKPOINT_PATH:-}" ]]; then
    overrides+=("checkpoint_path=${MOPA_CHECKPOINT_PATH}")
fi
if [[ -n "${MOBILE_ACTION_CONTRACT:-}" ]]; then
    overrides+=("mobile_action_contract=${MOBILE_ACTION_CONTRACT}")
fi
if [[ -n "${MOPA_ACTION_KEY_STYLE:-}" ]]; then
    overrides+=("action_key_style=${MOPA_ACTION_KEY_STYLE}")
fi
if [[ -n "${MOPA_EXECUTE_STEPS:-}" ]]; then
    overrides+=("execute_steps=${MOPA_EXECUTE_STEPS}")
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PY}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides "${overrides[@]}"
