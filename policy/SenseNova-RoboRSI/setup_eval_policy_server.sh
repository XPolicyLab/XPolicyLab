#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "usage: bash setup_eval_policy_server.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_env_or_uv_path> <port> [host]" >&2
    exit 2
fi
: "${CODEX_API_KEY:?Set CODEX_API_KEY before starting the SenseNova-RoboRSI policy server}"
export CODEX_API_KEY
# Keep controller startup independent of caller Python environment settings.
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_env=$8
policy_server_port=$9
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

if [[ "${policy_env}" == "uv" ]]; then
    policy_python="${SCRIPT_DIR}/.venv/bin/python"
else
    policy_path="${policy_env}"
    if [[ "${policy_path}" != /* ]]; then
        policy_path="${SCRIPT_DIR}/${policy_path}"
    fi
    if [[ -x "${policy_path}/bin/python" ]]; then
        policy_python="${policy_path}/bin/python"
    elif [[ -x "${policy_path}/.venv/bin/python" ]]; then
        policy_python="${policy_path}/.venv/bin/python"
    elif [[ -f "${policy_path}" && -x "${policy_path}" ]]; then
        policy_python="${policy_path}"
    elif [[ "${policy_env}" == */* || "${policy_env}" == .* ]]; then
        echo "[SERVER][ERROR] Policy Python or virtualenv unavailable: ${policy_path}" >&2
        exit 1
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
        # shellcheck disable=SC1091
        source "${conda_base}/etc/profile.d/conda.sh"
        conda activate "${policy_env}"
        policy_python="$(command -v python)"
    else
        echo "[SERVER][ERROR] Could not resolve policy environment: ${policy_env}" >&2
        exit 1
    fi
fi
if [[ ! -x "${policy_python}" ]]; then
    echo "[SERVER][ERROR] Policy interpreter unavailable: ${policy_python}. Run bash install.sh first." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}"
export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
export PATH="$(dirname "${policy_python}"):${PATH}"
cd "${XPL_ROOT}"

# Preflight and server share the exact interpreter and fixed deployment config.
"${policy_python}" "${SCRIPT_DIR}/preflight.py" \
    --config "${yaml_file}" --task-name "${task_name}"
echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}"
exec "${policy_python}" "${SCRIPT_DIR}/serve.py" \
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
