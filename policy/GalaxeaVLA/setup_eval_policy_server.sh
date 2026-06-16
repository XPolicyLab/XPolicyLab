#!/bin/bash
set -euo pipefail

dataset_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env_path=$8
policy_server_port=${9}
policy_server_host=${10:-localhost}

export CUDA_VISIBLE_DEVICES="${policy_gpu_id}"
echo -e "\033[33m[SERVER] GPU=${policy_gpu_id} host=${policy_server_host} port=${policy_server_port}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${SCRIPT_DIR}/deploy.yml"

ADAPTER_DIR="${SCRIPT_DIR}/GalaxeaVLA/xpolicylab_adapter"

source "${ADAPTER_DIR}/_artifact_paths.sh"

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

ckpt_run_id="${GALAXEA_CKPT_RUN_ID:-$(xpolicylab_ckpt_run_id "${dataset_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")}"

if [[ -d "${ckpt_name}" && "${ckpt_name}" == */* ]]; then
    ckpt_path="${ckpt_name}"
else
    ckpt_path="$(xpolicylab_resolve_ckpt_dir "${SCRIPT_DIR}" "${dataset_name}" "${ckpt_name}" \
        "${env_cfg_type}" "${action_type}" "${seed}")"
    if [[ ! -d "${ckpt_path}" ]]; then
        echo -e "\033[31m[SERVER] ckpt not found: checkpoints/${ckpt_run_id}\033[0m" >&2
        echo -e "\033[31m[SERVER] (eval args: dataset=${dataset_name} ckpt_name=${ckpt_name} env=${env_cfg_type} action=${action_type} seed=${seed})\033[0m" >&2
        exit 1
    fi
fi
ckpt_path="$(cd "${ckpt_path}" && pwd)"
ckpt_path="$(_resolve_run_root "${ckpt_path}")"
ckpt_path="$(cd "${ckpt_path}" && pwd)"
echo -e "\033[33m[SERVER] ckpt_run_id=${ckpt_run_id}\033[0m"
echo -e "\033[33m[SERVER] ckpt_path=${ckpt_path}\033[0m"

if [[ "${action_type}" != "joint" ]]; then
    echo -e "\033[31m[SERVER] GalaxeaVLA only supports action_type=joint, got '${action_type}'\033[0m" >&2
    exit 1
fi
task_config_name="real/g0plus_xpolicylab_finetune"
echo -e "\033[33m[SERVER] task_config_name=${task_config_name}\033[0m"

paligemma_path="${GALAXEA_PALIGEMMA_PATH:-${SCRIPT_DIR}/weights/paligemma-3b-pt-224}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

if [[ -z "${policy_uv_env_path}" || "${policy_uv_env_path}" == "null" ]]; then
    policy_uv_env_path="${SCRIPT_DIR}/GalaxeaVLA"
fi
policy_uv_env_path="$(cd "${policy_uv_env_path}" && pwd)"
VENV_PYTHON="${policy_uv_env_path}/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo -e "\033[31m[SERVER] missing uv venv python: ${VENV_PYTHON}\033[0m" >&2
    echo -e "\033[31m[SERVER] Run: cd ${SCRIPT_DIR} && bash install.sh\033[0m" >&2
    exit 1
fi
echo -e "\033[32m[SERVER] using uv venv: ${VENV_PYTHON}\033[0m"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${ROOT_DIR}:${policy_uv_env_path}/src:${PYTHONPATH:-}" \
    "${VENV_PYTHON}" -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            policy_name="GalaxeaVLA" \
            task_name="${task_name}" \
            dataset_name="${dataset_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            ckpt_path="${ckpt_path}" \
            task_config_name="${task_config_name}" \
            paligemma_path="${paligemma_path}"
