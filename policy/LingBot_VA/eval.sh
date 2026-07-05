#!/bin/bash
set -e

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Current Dir
ROOT_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

SERVER_SCRIPT="${CURRENT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${CURRENT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

CONFIG_NAME="${LINGBOT_VA_CONFIG_NAME:-robotwin30_train}"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

# ---------------------------------------------------------------------------
# Resolve the finetuned checkpoint directory for the wan_va backend server.
# Training writes weights to
#   <run_root>/checkpoints/checkpoint_step_<N>/transformer/
# where <run_root> = policy/LingBot_VA/checkpoints/<ckpt_name> (or an absolute
# ckpt_name). We hand <run_root> to launch_wan_va_server.sh ->
# prepare_merged_ckpt.py, which auto-discovers the latest checkpoint_step_<N>.
# Overrides:
#   LINGBOT_VA_CHECKPOINT_PATH  point straight at a ckpt dir (skips the above)
#   LINGBOT_VA_STEP             pin a specific step (consumed by prepare_merged_ckpt.py)
# ---------------------------------------------------------------------------
if [[ -n "${LINGBOT_VA_CHECKPOINT_PATH:-}" ]]; then
    VA_CHECKPOINT_PATH="${LINGBOT_VA_CHECKPOINT_PATH}"
elif [[ "${ckpt_name}" = /* ]]; then
    VA_CHECKPOINT_PATH="${ckpt_name}"
else
    VA_CHECKPOINT_PATH="${CURRENT_DIR}/checkpoints/${ckpt_name}"
fi

# wan_va backend endpoint. Reuse an already-running server by exporting
# LINGBOT_VA_VA_HOST / LINGBOT_VA_VA_PORT; otherwise launch one here.
va_server_host="${LINGBOT_VA_VA_HOST:-127.0.0.1}"
if [[ -n "${LINGBOT_VA_VA_PORT:-}" ]]; then
    va_server_port="${LINGBOT_VA_VA_PORT}"
    LAUNCH_VA_SERVER=0
else
    va_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
    va_master_port=$(bash "${UTILS_DIR}/get_free_port.sh")
    LAUNCH_VA_SERVER=1
fi

VA_SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill forward server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
    if [[ -n "${VA_SERVER_PID:-}" ]]; then
        echo "[MAIN] kill wan_va server group ${VA_SERVER_PID}"
        # VA_SERVER_PID is the setsid process-group leader; kill the whole group
        # so torch.distributed.run and its worker exit too.
        kill -TERM "-${VA_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# 1) wan_va backend server (holds the real weights). Launched in its own
#    process group (setsid) so cleanup() can reap the whole torchrun tree.
# ---------------------------------------------------------------------------
if [[ "${LAUNCH_VA_SERVER}" == "1" ]]; then
    echo "[MAIN] launch wan_va backend, endpoint=${va_server_host}:${va_server_port}"
    echo "[MAIN]   checkpoint_path=${VA_CHECKPOINT_PATH}"
    echo "[MAIN]   config_name=${CONFIG_NAME}, gpu=${policy_gpu_id}, master_port=${va_master_port}"

    setsid env \
        LVA_CURRENT_DIR="${CURRENT_DIR}" \
        LVA_CONDA_ENV="${policy_conda_env}" \
        LVA_CHECKPOINT_PATH="${VA_CHECKPOINT_PATH}" \
        LVA_CONFIG_NAME="${CONFIG_NAME}" \
        LVA_MASTER_PORT="${va_master_port}" \
        LVA_GPU_ID="${policy_gpu_id}" \
        LVA_PORT="${va_server_port}" \
        bash -c '
            set -eo pipefail
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate "${LVA_CONDA_ENV}"
            base_model_path=$(python - "${LVA_CURRENT_DIR}/deploy.yml" <<PY
import sys, yaml
print(yaml.safe_load(open(sys.argv[1], encoding="utf-8")).get("base_model_path", "") or "")
PY
)
            export CHECKPOINT_PATH="${LVA_CHECKPOINT_PATH}"
            [[ -n "${base_model_path}" ]] && export BASE_MODEL_PATH="${base_model_path}"
            export CONFIG_NAME="${LVA_CONFIG_NAME}"
            export MASTER_PORT="${LVA_MASTER_PORT}"
            cd "${LVA_CURRENT_DIR}"
            exec bash launch_wan_va_server.sh "${LVA_GPU_ID}" "${LVA_PORT}"
        ' &
    VA_SERVER_PID=$!

    bash "${UTILS_DIR}/wait_for_policy_server.sh" \
        "${va_server_host}" \
        "${va_server_port}" \
        "${VA_SERVER_PID}" \
        "wan_va server" \
        "${LINGBOT_VA_VA_TIMEOUT:-1800}"
else
    echo "[MAIN] using external wan_va server at ${va_server_host}:${va_server_port}"
fi

# The forward policy server (model.py) connects to the backend via these.
export VA_SERVER_HOST="${va_server_host}"
export VA_SERVER_PORT="${va_server_port}"

# ---------------------------------------------------------------------------
# 2) forward policy server (model.py bridge -> wan_va backend).
# ---------------------------------------------------------------------------
echo "[MAIN] start forward server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" \
    "${policy_server_ip}" \
    "${CONFIG_NAME}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 600

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo "[MAIN] eval finished"
