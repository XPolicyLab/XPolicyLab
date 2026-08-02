#!/usr/bin/env bash
set -euo pipefail

bench_name=$1
task_name=$2
_submission_ckpt=$3
env_cfg_type=$4
_submission_action_type=$5
seed=$6
policy_gpu_id=$7
_submission_policy_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROUTER_PYTHON="${CSU_ROUTER_PYTHON:-}"
if [[ -z "${ROUTER_PYTHON}" ]]; then
    ROUTER_PYTHON="$(command -v python3 || command -v python)"
fi
[[ -x "${ROUTER_PYTHON}" ]] || {
    echo "[CSU-AI-0][ERROR] router python is not executable: ${ROUTER_PYTHON}" >&2
    exit 1
}

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$(cd "${XPL_ROOT}/.." && pwd):${XPL_ROOT}:${PYTHONPATH:-}" \
    "${ROUTER_PYTHON}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${SCRIPT_DIR}/deploy.yml" \
        --overrides \
            protocol=ws \
            host="${policy_server_host}" \
            port="${policy_server_port}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name=QRouter \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name=CSU-AI-0 \
            action_type=auto \
            gpu_id="${policy_gpu_id}"
