#!/bin/bash
set -e

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_conda_env=$9
policy_server_port=${10}
policy_server_host=${11:-"localhost"}
config_name=${12:-robotwin30_train}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${CURRENT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

if [[ "${ckpt_name}" = /* ]]; then
    CHECKPOINT_PATH="${ckpt_name}"
else
    CHECKPOINT_PATH="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/checkpoints/${ckpt_name}"
fi

BASE_MODEL_PATH=$(python - <<PY
import sys
import yaml
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
base = cfg.get("base_model_path")
if not base:
    sys.exit("base_model_path must be set in deploy.yml")
print(base)
PY
)

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] checkpoint_path=${CHECKPOINT_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${policy_server_port}"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

# Upstream LingBot VA server (launch_wan_va_server.sh) address.
# Override via env vars VA_SERVER_HOST / VA_SERVER_PORT; otherwise fall back
# to deploy.yml's va_server_host / va_server_port.
OVERRIDE_LIST=(
    port="${policy_server_port}"
    host="${policy_server_host}"
    bench_name="${bench_name}"
    task_name="${task_name}"
    ckpt_name="${ckpt_name}"
    env_cfg_type="${env_cfg_type}"
    env_cfg="${env_cfg_type}"
    expert_data_num="${expert_data_num}"
    seed="${seed}"
    policy_name="${policy_name}"
    action_type="${action_type}"
    action_dim="${action_dim}"
    checkpoint_path="${CHECKPOINT_PATH}"
    base_model_path="${BASE_MODEL_PATH}"
    config_name="${config_name}"
)

if [[ -n "${VA_SERVER_HOST:-}" ]]; then
    OVERRIDE_LIST+=("va_server_host=${VA_SERVER_HOST}")
    echo "[SERVER] override va_server_host=${VA_SERVER_HOST}"
fi
if [[ -n "${VA_SERVER_PORT:-}" ]]; then
    OVERRIDE_LIST+=("va_server_port=${VA_SERVER_PORT}")
    echo "[SERVER] override va_server_port=${VA_SERVER_PORT}"
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            "${OVERRIDE_LIST[@]}"
