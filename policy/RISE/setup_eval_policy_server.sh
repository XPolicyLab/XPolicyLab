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
policy_server_host=${10:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
ADAPTER_DIR="${SCRIPT_DIR}/xpolicylab_adapter"
OFFLINE_DIR="${SCRIPT_DIR}/RISE/policy_and_value/policy_offline_and_value"

source "${ADAPTER_DIR}/_artifact_paths.sh"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

expert_data_num="${RISE_EXPERT_DATA_NUM:-}"
ckpt_run_id="${RISE_CKPT_RUN_ID:-$(xpolicylab_ckpt_run_id "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" "${seed}")}"
ckpt_root="$(xpolicylab_resolve_ckpt_dir "${SCRIPT_DIR}" "${bench_name}" "${ckpt_name}" \
    "${env_cfg_type}" "${action_type}" "${seed}" "${expert_data_num}")"
ckpt_root_rel="${ckpt_root#${SCRIPT_DIR}/}"
policy_root_rel="${ckpt_root_rel}/Policy_offline_release/Policy_offline_release"
policy_root="${SCRIPT_DIR}/${policy_root_rel}"
config_name="${RISE_CONFIG_NAME:-Policy_offline_release}"
default_prompt="${RISE_DEFAULT_PROMPT:-stack the bowls}"
asset_id="${RISE_ASSET_ID:-}"
model_action_dim="${RISE_MODEL_ACTION_DIM:-}"
checkpoint_step="${RISE_CHECKPOINT_STEP:-}"

is_valid_checkpoint_dir() {
    local dir="$1"
    [[ -f "${dir}/model.safetensors" || -f "${dir}/model.pt" || -d "${dir}/params" ]]
}

checkpoint_path="${RISE_CHECKPOINT_PATH:-}"
checkpoint_path_abs=""

if [[ -n "${checkpoint_path}" && "${checkpoint_path}" != "null" ]]; then
    checkpoint_path_abs="${checkpoint_path}"
    [[ "${checkpoint_path_abs}" = /* ]] || checkpoint_path_abs="${SCRIPT_DIR}/${checkpoint_path_abs}"
elif is_valid_checkpoint_dir "${ckpt_root}"; then
    checkpoint_path="${ckpt_root_rel}"
    checkpoint_path_abs="${ckpt_root}"
else
    if [[ -n "${checkpoint_step}" ]]; then
        step_dir="${policy_root}/${checkpoint_step}"
    else
        latest_step=""
        for step_dir in "${policy_root}"/*; do
            [[ -d "${step_dir}" ]] || continue
            step="$(basename "${step_dir}")"
            [[ "${step}" =~ ^[0-9]+$ ]] || continue
            is_valid_checkpoint_dir "${step_dir}" || continue
            if [[ -z "${latest_step}" || "${step}" -gt "${latest_step}" ]]; then
                latest_step="${step}"
            fi
        done
        step_dir="${policy_root}/${latest_step}"
    fi

    if [[ -n "${checkpoint_step:-${latest_step:-}}" && -d "${step_dir}" ]] && is_valid_checkpoint_dir "${step_dir}"; then
        step="$(basename "${step_dir}")"
        checkpoint_path="${policy_root_rel}/${step}"
        checkpoint_path_abs="${step_dir}"
    fi
fi

if [[ -z "${checkpoint_path}" ]]; then
    echo -e "\033[31m[SERVER] checkpoint not found for ckpt_name='${ckpt_name}'\033[0m" >&2
    echo -e "\033[31m[SERVER] expected model.safetensors, model.pt, or params/ in the checkpoint directory.\033[0m" >&2
    echo -e "\033[31m[SERVER] tried: RISE_CHECKPOINT_PATH, ${ckpt_root_rel}\033[0m" >&2
    exit 1
fi

if ! is_valid_checkpoint_dir "${checkpoint_path_abs}"; then
    echo -e "\033[31m[SERVER] invalid checkpoint directory: ${checkpoint_path}\033[0m" >&2
    echo -e "\033[31m[SERVER] expected model.safetensors, model.pt, or params/; found assets-only or incomplete checkpoint.\033[0m" >&2
    exit 1
fi

if [[ -z "${asset_id}" ]]; then
    asset_id=$(
        python3 - "${checkpoint_path_abs}" <<'PY'
import pathlib
import sys

assets_dir = pathlib.Path(sys.argv[1]) / "assets"
matches = sorted(assets_dir.glob("*/norm_stats.json"))
if len(matches) == 1:
    print(matches[0].parent.name)
PY
    )
fi

echo -e "\033[33m[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] ckpt_run_id=${ckpt_run_id}\033[0m"
echo -e "\033[33m[SERVER] checkpoint_path=${checkpoint_path}\033[0m"
echo -e "\033[33m[SERVER] config_name=${config_name}\033[0m"
echo -e "\033[33m[SERVER] asset_id=${asset_id:-<config default>}\033[0m"
echo -e "\033[33m[SERVER] policy_server_host=${policy_server_host} policy_server_port=${policy_server_port}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] env_action_dim=${action_dim}, model_action_dim=${model_action_dim:-<config default>}\033[0m"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${OFFLINE_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}" \
    python3 -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
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
            action_dim="${action_dim}" \
            model_action_dim="${model_action_dim}" \
            gpu_id="${policy_gpu_id}" \
            config_name="${config_name}" \
            checkpoint_path="${checkpoint_path}" \
            default_prompt="${default_prompt}" \
            asset_id="${asset_id}"
