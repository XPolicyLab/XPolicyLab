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
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

extra_overrides=()
if [[ -n "${WALLX_CHECKPOINT:-}" ]]; then
    extra_overrides+=(checkpoint_path="${WALLX_CHECKPOINT}")
fi
if [[ -n "${WALLX_TRAIN_CONFIG:-}" ]]; then
    extra_overrides+=(train_config_path="${WALLX_TRAIN_CONFIG}")
fi

# wall-x must be importable in the policy env; point WALLX_REPO at the checkout
# (default: a sibling of the XPolicyLab checkout, i.e. <bench root>/wall-x).
WALLX_REPO="${WALLX_REPO:-${BENCH_ROOT}/wall-x}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${WALLX_REPO}:${PYTHONPATH:-}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
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
            action_type="${action_type}" \
            "${extra_overrides[@]}"
