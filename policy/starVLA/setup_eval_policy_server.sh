#!/bin/bash
set -e

dataset_name=$1
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
checkpoint_path=${12:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
processed_name="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
default_result_ckpt="${SCRIPT_DIR}/results/Checkpoints/${processed_name}-${seed}/final_model/pytorch_model.pt"
default_legacy_ckpt="${SCRIPT_DIR}/checkpoints/${processed_name}-${seed}"
if [[ -z "${checkpoint_path}" ]]; then
    if [[ -f "${default_result_ckpt}" ]]; then
        checkpoint_path="${default_result_ckpt}"
    else
        checkpoint_path="${default_legacy_ckpt}"
    fi
fi
if [[ ! -f "${checkpoint_path}" ]]; then
    echo "[SERVER][ERROR] checkpoint file does not exist: ${checkpoint_path}" >&2
    echo "[SERVER][ERROR] pass checkpoint as eval.sh arg 12, e.g. results/Checkpoints/<run>/final_model/pytorch_model.pt" >&2
    exit 1
fi
checkpoint_path="$(realpath "${checkpoint_path}")"
starvla_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
starvla_server_host="127.0.0.1"

cleanup() {
    if [[ -n "${STARVLA_SERVER_PID:-}" ]]; then
        echo "[SERVER] kill StarVLA websocket server ${STARVLA_SERVER_PID}"
        kill "${STARVLA_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, starvla_port=${starvla_server_port}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

(
    cd "${STARVLA_ROOT}"
    PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${checkpoint_path}" \
        --port "${starvla_server_port}" \
        --use_bf16
) &
STARVLA_SERVER_PID=$!

sleep 6

PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${policy_server_port}" \
        host="${policy_server_host}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        ckpt_name="${ckpt_name}" \
        checkpoint_path="${checkpoint_path}" \
        env_cfg_type="${env_cfg_type}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        starvla_root="${STARVLA_ROOT}" \
        starvla_server_host="${starvla_server_host}" \
        starvla_server_port="${starvla_server_port}"
