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
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

CKPT_ROOT="${SCRIPT_DIR}/checkpoints"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo "[SERVER] policy=${policy_name} task=${task_name} ckpt=${ckpt_name}"
echo "[SERVER] ${policy_server_host}:${policy_server_port} gpu=${policy_gpu_id}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${SCRIPT_DIR}/src" \
    DIFFSYNTH_MODEL_BASE_PATH="${CKPT_ROOT}" \
    COSMOS3_NANO_ROOT="${CKPT_ROOT}/Cosmos3-Nano" \
    python -u "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            "port=${policy_server_port}" \
            "host=${policy_server_host}" \
            "bench_name=${bench_name}" \
            "task_name=${task_name}" \
            "ckpt_name=${ckpt_name}" \
            "env_cfg_type=${env_cfg_type}" \
            "seed=${seed}" \
            "policy_name=${policy_name}" \
            "action_type=${action_type}"
