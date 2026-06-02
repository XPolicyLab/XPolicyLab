#!/bin/bash
set -e

# Launched by eval.sh. Activates the GalaxeaVLA uv env, resolves the checkpoint
# directory from ckpt_name, then starts the XPolicyLab policy server.
dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
expert_data_num=$5
action_type=$6
seed=$7
policy_gpu_id=$8
policy_uv_env_path=$9          # uv project dir holding .venv (conda-env slot)
policy_server_port=${10}

export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
echo -e "\033[33m[SERVER] GPU=${policy_gpu_id} port=${policy_server_port}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"

_resolve_run_root() {
    local root="$1"
    if [[ -f "${root}/dataset_stats.json" || -d "${root}/checkpoints" ]]; then
        echo "${root}"
        return 0
    fi
    local latest=""
    local run_dir
    for run_dir in "${root}"/*/; do
        [[ -d "${run_dir}" ]] || continue
        if [[ -f "${run_dir}/dataset_stats.json" || -d "${run_dir}/checkpoints" ]]; then
            latest="${run_dir%/}"
        fi
    done
    if [[ -n "${latest}" ]]; then
        echo "${latest}"
    else
        echo "${root}"
    fi
}

# XPolicyLab 6-tuple (see XPolicyLab/README.md); ckpt_name alone is NOT the directory key.
ckpt_run_id="${GALAXEA_CKPT_RUN_ID:-${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}}"

# ---- resolve eval args -> ckpt_path ----
if [[ -d "${ckpt_name}" && "${ckpt_name}" == */* ]]; then
    # absolute path or path with slashes passed as ckpt_name slot
    ckpt_path="${ckpt_name}"
elif [[ -d "${SCRIPT_DIR}/checkpoints/${ckpt_run_id}" ]]; then
    ckpt_path="${SCRIPT_DIR}/checkpoints/${ckpt_run_id}"
elif [[ -d "${SCRIPT_DIR}/checkpoints/${ckpt_name}" ]]; then
    echo -e "\033[33m[SERVER] fallback: checkpoints/${ckpt_name} (legacy layout)\033[0m"
    ckpt_path="${SCRIPT_DIR}/checkpoints/${ckpt_name}"
else
    echo -e "\033[31m[SERVER] ckpt not found: checkpoints/${ckpt_run_id}\033[0m" >&2
    echo -e "\033[31m[SERVER] (eval args: dataset=${dataset_name} ckpt_name=${ckpt_name} env=${env_cfg_type} num=${expert_data_num} action=${action_type} seed=${seed})\033[0m" >&2
    exit 1
fi
ckpt_path="$(cd "${ckpt_path}" && pwd)"
ckpt_path="$(_resolve_run_root "${ckpt_path}")"
ckpt_path="$(cd "${ckpt_path}" && pwd)"
echo -e "\033[33m[SERVER] ckpt_run_id=${ckpt_run_id}\033[0m"
echo -e "\033[33m[SERVER] ckpt_path=${ckpt_path}\033[0m"

# ---- action_type -> upstream Hydra task config (must match training) ----
case "${action_type}" in
    ee)    task_config_name="real/g0plus_xpolicylab_ee_finetune" ;;
    joint) task_config_name="real/g0plus_xpolicylab_finetune" ;;
    *)
        echo -e "\033[31m[SERVER] unknown action_type '${action_type}' (expected ee|joint)\033[0m"
        exit 1
        ;;
esac
echo -e "\033[33m[SERVER] task_config_name=${task_config_name}\033[0m"

# Local backbone dir (PaliGemma / SmolVLM2). deploy.yml may also set this.
paligemma_path="${GALAXEA_PALIGEMMA_PATH:-${SCRIPT_DIR}/weights/paligemma-3b-pt-224}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

# ---- activate uv env ----
if [[ -z "${policy_uv_env_path}" || "${policy_uv_env_path}" == "null" ]]; then
    policy_uv_env_path="${SCRIPT_DIR}/GalaxeaVLA"
fi
echo -e "\033[32m[SERVER] activating uv env: ${policy_uv_env_path}/.venv\033[0m"
source "${policy_uv_env_path}/.venv/bin/activate"

PYTHONWARNINGS=ignore::UserWarning \
PYTHONPATH="${ROOT_DIR}:${policy_uv_env_path}/src:${PYTHONPATH}" \
python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${policy_server_port}" \
        host="localhost" \
        policy_name="GalaxeaVLA" \
        task_name="${task_name}" \
        dataset_name="${dataset_name}" \
        env_cfg_type="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        ckpt_path="${ckpt_path}" \
        task_config_name="${task_config_name}" \
        paligemma_path="${paligemma_path}"
