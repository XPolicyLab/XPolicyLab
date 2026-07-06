#!/bin/bash
set -euo pipefail
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
checkpoint_path=${11:-""}
config_path=${12:-""}
lang_embedding_path=${13:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

resolve_checkpoint_path() {
    local explicit_path="$1"
    local default_dir="$2"

    if [[ -n "${explicit_path}" ]]; then
        echo "${explicit_path}"
        return
    fi

    if [[ -f "${default_dir}/pytorch_model.bin" || -f "${default_dir}/model.safetensors" || -f "${default_dir}/config.json" ]]; then
        echo "${default_dir}"
        return
    fi

    if [[ ! -d "${default_dir}" ]]; then
        echo "${default_dir}"
        return
    fi

    local matches=()
    shopt -s nullglob
    matches=("${default_dir}"/checkpoint-*)
    shopt -u nullglob

    if (( ${#matches[@]} == 1 )); then
        echo "${matches[0]}"
        return
    fi

    if (( ${#matches[@]} == 0 )); then
        echo "[ERROR] No checkpoint-* found under ${default_dir}" >&2
    else
        echo "[ERROR] Multiple checkpoint-* directories found under ${default_dir}; pass checkpoint_path explicitly." >&2
    fi
    exit 1
}

# ckpt_name is the full run directory name under checkpoints/.
checkpoint_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"
checkpoint_path="$(resolve_checkpoint_path "${checkpoint_path}" "${checkpoint_dir}")"
lang_embedding_path="${lang_embedding_path:-${SCRIPT_DIR}/H_RDT/datasets/xpolicylab/lang_embeddings/${task_name}.pt}"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"
echo "[MAIN] checkpoint_path=${checkpoint_path}"

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
    "${checkpoint_path}" \
    "${config_path}" \
    "${lang_embedding_path}" &

SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

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