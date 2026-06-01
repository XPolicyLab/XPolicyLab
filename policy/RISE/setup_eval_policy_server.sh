#!/bin/bash
set -euo pipefail

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
policy_server_host=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
OFFLINE_DIR="${SCRIPT_DIR}/RISE/policy_and_value/policy_offline_and_value"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

CKPT_ROOT="${OFFLINE_DIR}/checkpoints/Policy_offline_release/Policy_offline_release"
STANDARD_CKPT_DIR="${SCRIPT_DIR}/checkpoints/${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
STANDARD_POLICY_ROOT="${STANDARD_CKPT_DIR}/Policy_offline_release/Policy_offline_release"
config_name="${RISE_CONFIG_NAME:-Policy_offline_release}"
default_prompt="${RISE_DEFAULT_PROMPT:-stack the bowls}"
debug_zero_action="${RISE_DEBUG_ZERO_ACTION:-false}"
asset_id="${RISE_ASSET_ID:-}"
model_action_dim="${RISE_MODEL_ACTION_DIM:-}"
checkpoint_step="${RISE_CHECKPOINT_STEP:-}"

is_valid_checkpoint_dir() {
    local dir="$1"
    [[ -f "${dir}/model.safetensors" || -f "${dir}/model.pt" || -d "${dir}/params" ]]
}

latest_valid_step_dir() {
    local root="$1"
    python - "${root}" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(1)

steps = sorted(
    int(path.name)
    for path in root.iterdir()
    if path.is_dir()
    and path.name.isdigit()
    and (
        (path / "model.safetensors").is_file()
        or (path / "model.pt").is_file()
        or (path / "params").is_dir()
    )
)
if not steps:
    raise SystemExit(1)
print(root / str(steps[-1]))
PY
}

resolve_checkpoint_path() {
    local name="$1"

    if [[ -n "${RISE_CHECKPOINT_PATH:-}" && "${RISE_CHECKPOINT_PATH}" != "null" ]]; then
        echo "${RISE_CHECKPOINT_PATH}"
        return 0
    fi

    if [[ -d "${name}" ]]; then
        echo "$(cd "${name}" && pwd)"
        return 0
    fi

    if [[ -d "${SCRIPT_DIR}/checkpoints/${name}" ]]; then
        echo "$(cd "${SCRIPT_DIR}/checkpoints/${name}" && pwd)"
        return 0
    fi

    # XPolicyLab README convention:
    # policy/RISE/checkpoints/<dataset>-<ckpt>-<env_cfg>-<expert_num>-<action_type>-<seed>/
    if is_valid_checkpoint_dir "${STANDARD_CKPT_DIR}"; then
        echo "${STANDARD_CKPT_DIR}"
        return 0
    fi

    if [[ -n "${checkpoint_step}" && -d "${STANDARD_POLICY_ROOT}/${checkpoint_step}" ]] && is_valid_checkpoint_dir "${STANDARD_POLICY_ROOT}/${checkpoint_step}"; then
        echo "${STANDARD_POLICY_ROOT}/${checkpoint_step}"
        return 0
    fi

    if [[ -d "${STANDARD_POLICY_ROOT}" ]]; then
        local latest_standard
        if latest_standard="$(latest_valid_step_dir "${STANDARD_POLICY_ROOT}")"; then
            echo "${latest_standard}"
            return 0
        fi
    fi

    # Backward-compatible legacy paths from upstream RISE.
    if [[ -d "${CKPT_ROOT}/${name}" ]] && is_valid_checkpoint_dir "${CKPT_ROOT}/${name}"; then
        echo "${CKPT_ROOT}/${name}"
        return 0
    fi

    if [[ "${name}" == "latest" && -d "${CKPT_ROOT}" ]]; then
        local latest_step
        latest_step="$(latest_valid_step_dir "${CKPT_ROOT}")"
        echo "${latest_step}"
        return 0
    fi

    return 1
}

if ! checkpoint_path="$(resolve_checkpoint_path "${ckpt_name}")"; then
    echo -e "\033[31m[SERVER] checkpoint not found for ckpt_name='${ckpt_name}'\033[0m" >&2
    echo -e "\033[31m[SERVER] expected model.safetensors, model.pt, or params/ in the checkpoint directory.\033[0m" >&2
    echo -e "\033[31m[SERVER] tried: RISE_CHECKPOINT_PATH, abs dir, ${STANDARD_CKPT_DIR}, ${CKPT_ROOT}/<step>\033[0m" >&2
    exit 1
fi

if ! is_valid_checkpoint_dir "${checkpoint_path}"; then
    echo -e "\033[31m[SERVER] invalid checkpoint directory: ${checkpoint_path}\033[0m" >&2
    echo -e "\033[31m[SERVER] expected model.safetensors, model.pt, or params/; found assets-only or incomplete checkpoint.\033[0m" >&2
    exit 1
fi

if [[ -z "${asset_id}" ]]; then
    asset_id=$(
        python - "${checkpoint_path}" <<'PY'
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
echo -e "\033[33m[SERVER] standard_ckpt_dir=${STANDARD_CKPT_DIR}\033[0m"
echo -e "\033[33m[SERVER] checkpoint_path=${checkpoint_path}\033[0m"
echo -e "\033[33m[SERVER] config_name=${config_name}, debug_zero_action=${debug_zero_action}\033[0m"
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
    python -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            dataset_name="${dataset_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            model_action_dim="${model_action_dim}" \
            gpu_id="${policy_gpu_id}" \
            config_name="${config_name}" \
            checkpoint_path="${checkpoint_path}" \
            default_prompt="${default_prompt}" \
            asset_id="${asset_id}" \
            debug_zero_action="${debug_zero_action}"
