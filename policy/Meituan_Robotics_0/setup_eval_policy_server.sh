#!/bin/bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: bash setup_eval_policy_server.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_conda_env> <port> [host]"
    exit 1
fi

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
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
yaml_file="${MEITUAN_ROBOTICS_DEPLOY_CONFIG:-${SCRIPT_DIR}/deploy.yml}"
if [[ "${yaml_file}" != /* ]]; then
    yaml_file="${SCRIPT_DIR}/${yaml_file}"
fi
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

if [[ -x "${policy_conda_env}/bin/python" ]]; then
    policy_python="${policy_conda_env}/bin/python"
else
    if ! command -v conda >/dev/null 2>&1; then
        echo "[SERVER][ERROR] conda is required when policy_conda_env is an environment name: ${policy_conda_env}" >&2
        exit 1
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
    policy_python="$(command -v python)"
fi

overrides=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    action_type="${action_type}"
    action_dim="${action_dim}"
    seed="${seed}"
    policy_name=Meituan_Robotics_0
)
if [[ -n "${MEITUAN_ROBOTICS_CKPT_PATH:-}" ]]; then
    overrides+=(checkpoint_path="${MEITUAN_ROBOTICS_CKPT_PATH}")
fi
if [[ -n "${STARVLA_BASE_VLM:-}" ]]; then
    overrides+=(base_vlm="${STARVLA_BASE_VLM}")
fi

unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export PYTHONPATH="${SCRIPT_DIR}/source_starvla:${PYTHONPATH:-}"
export PYTHONPATH="${XPL_ROOT}/..:${PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

exec env PYTHONWARNINGS=ignore::UserWarning CUDA_VISIBLE_DEVICES="${policy_gpu_id}" "${policy_python}" "${XPL_ROOT}/setup_policy_server.py" --config_path "${yaml_file}" --overrides "${overrides[@]}"
