#!/bin/bash
set -euo pipefail

if [[ $# -lt 10 || $# -gt 11 ]]; then
    echo "Usage: bash setup_eval_policy_server.sh <dataset_name> <task_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <policy_gpu_id> <policy_conda_env> <policy_server_port> [policy_server_host]"
    exit 1
fi

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
EVENTVLA_ROOT="${SCRIPT_DIR}/source_eventvla"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

read_yaml_value() {
    local key=$1
    awk -F': ' -v lookup_key="${key}" '
        $1 == lookup_key {
            value=$2
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            gsub(/^"|"$/, "", value)
            if (value == "null") value=""
            print value
            exit
        }
    ' "${yaml_file}"
}

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
processed_name="${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}"
result_run_dir="${SCRIPT_DIR}/results/Checkpoints/${processed_name}-${seed}"
local_run_dir="${SCRIPT_DIR}/checkpoints/${processed_name}-${seed}"

resolve_eventvla_checkpoint() {
    local run_dir=$1
    local candidates=()

    if [[ ! -d "${run_dir}" ]]; then
        return 1
    fi

    shopt -s nullglob
    candidates=("${run_dir}"/checkpoints/*.pt "${run_dir}"/checkpoints/*.safetensors)
    shopt -u nullglob

    if (( ${#candidates[@]} == 1 )); then
        echo "${candidates[0]}"
        return 0
    fi
    if (( ${#candidates[@]} > 1 )); then
        echo "[SERVER][ERROR] multiple checkpoints found under ${run_dir}/checkpoints:" >&2
        printf '[SERVER][ERROR]   %s\n' "${candidates[@]}" >&2
        echo "[SERVER][ERROR] keep only one checkpoint file or set EVENTVLA_CKPT_PATH explicitly." >&2
        exit 1
    fi

    return 1
}

checkpoint_path="${EVENTVLA_CKPT_PATH:-}"
deploy_checkpoint_path="$(read_yaml_value checkpoint_path)"
if [[ -n "${checkpoint_path}" ]]; then
    :
elif [[ -n "${deploy_checkpoint_path}" && -f "${deploy_checkpoint_path}" ]]; then
    checkpoint_path="${deploy_checkpoint_path}"
elif checkpoint_path=$(resolve_eventvla_checkpoint "${result_run_dir}"); then
    :
elif checkpoint_path=$(resolve_eventvla_checkpoint "${local_run_dir}"); then
    :
else
    checkpoint_path="${local_run_dir}/checkpoints/<checkpoint>.pt"
fi

if [[ ! -f "${checkpoint_path}" ]]; then
    echo "[SERVER][ERROR] checkpoint file does not exist: ${checkpoint_path}" >&2
    echo "[SERVER][ERROR] set EVENTVLA_CKPT_PATH=/path/to/pytorch_model.pt to override checkpoint lookup" >&2
    echo "[SERVER][ERROR] expected exactly one .pt or .safetensors file under one of:" >&2
    echo "[SERVER][ERROR]   ${result_run_dir}/checkpoints/" >&2
    echo "[SERVER][ERROR]   ${local_run_dir}/checkpoints/" >&2
    exit 1
fi

checkpoint_path="$(realpath "${checkpoint_path}")"
run_dir="$(cd "$(dirname "${checkpoint_path}")/.." && pwd)"
if [[ ! -f "${run_dir}/config.yaml" || ! -f "${run_dir}/dataset_statistics.json" ]]; then
    echo "[SERVER][ERROR] EventVLA checkpoint run dir must contain config.yaml and dataset_statistics.json: ${run_dir}" >&2
    exit 1
fi

eventvla_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
eventvla_server_host="127.0.0.1"

cleanup() {
    if [[ -n "${EVENTVLA_SERVER_PID:-}" ]]; then
        echo "[SERVER] kill EventVLA websocket server ${EVENTVLA_SERVER_PID}"
        kill "${EVENTVLA_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[SERVER] resolved EventVLA checkpoint: ${checkpoint_path}"
echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, eventvla_port=${eventvla_server_port}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

(
    cd "${EVENTVLA_ROOT}"
    PYTHONPATH="${EVENTVLA_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    DEBUG="" \
    python "${EVENTVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "${checkpoint_path}" \
        --port "${eventvla_server_port}" \
        --use_bf16
) &
EVENTVLA_SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" \
    "${eventvla_server_host}" \
    "${eventvla_server_port}" \
    "${EVENTVLA_SERVER_PID}" \
    "EventVLA websocket server" \
    "${EVENTVLA_SERVER_READY_TIMEOUT:-360}"

PYTHONPATH="${EVENTVLA_ROOT}:${PYTHONPATH:-}" \
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
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        eventvla_root="${EVENTVLA_ROOT}" \
        eventvla_server_host="${eventvla_server_host}" \
        eventvla_server_port="${eventvla_server_port}" \
        unnorm_key="new_embodiment" \
        action_mode="abs"
