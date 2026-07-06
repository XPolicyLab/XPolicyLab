#!/bin/bash
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
policy_server_host=${10:-"localhost"}
config_name=${11:-robotwin30_train}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

if [[ "${ckpt_name}" = /* ]]; then
    CHECKPOINT_PATH="${ckpt_name}"
else
    CHECKPOINT_PATH="${XPL_ROOT}/policy/${policy_name}/checkpoints/${ckpt_name}"
fi

BASE_MODEL_PATH="${LINGBOT_VA_BASE_MODEL_PATH:-}"
if [[ -z "${BASE_MODEL_PATH}" ]]; then
    BASE_MODEL_PATH=$(python - <<PY
import yaml
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
print(cfg.get("base_model_path") or "")
PY
)
fi
if [[ -z "${BASE_MODEL_PATH}" ]]; then
    echo "[SERVER][ERROR] base model path not set. Set LINGBOT_VA_BASE_MODEL_PATH env or base_model_path in deploy.yml." >&2
    exit 1
fi

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] checkpoint_path=${CHECKPOINT_PATH}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

forward_master_port="${LINGBOT_VA_FORWARD_MASTER_PORT:-$(bash "${UTILS_DIR}/get_free_port.sh")}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${forward_master_port}"
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
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            "${OVERRIDE_LIST[@]}"
