#!/bin/bash
set -e

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
checkpoint_path=${11:-""}
config_path=${12:-""}
lang_embedding_path=${13:-""}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

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

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
# ckpt_name is the full run directory name under checkpoints/.
checkpoint_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"
checkpoint_path="$(resolve_checkpoint_path "${checkpoint_path}" "${checkpoint_dir}")"
# Prefer the config copied into the checkpoint dir by train.sh; fall back to data/;
# pass config_path explicitly if neither matches.
if [[ -z "${config_path}" ]]; then
    if [[ -f "${checkpoint_dir}/hrdt_finetune_xpolicy.yaml" ]]; then
        config_path="${checkpoint_dir}/hrdt_finetune_xpolicy.yaml"
    else
        config_path="${SCRIPT_DIR}/data/${ckpt_name}/hrdt_finetune_xpolicy.yaml"
    fi
fi
lang_embedding_path="${lang_embedding_path:-${SCRIPT_DIR}/H_RDT/datasets/xpolicylab/lang_embeddings/${task_name}.pt}"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] checkpoint_path=${checkpoint_path}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            checkpoint_path="${checkpoint_path}" \
            config_path="${config_path}" \
            lang_embedding_path="${lang_embedding_path}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"