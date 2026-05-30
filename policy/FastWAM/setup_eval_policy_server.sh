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

policy_name="$(basename "${SCRIPT_DIR}")"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/${policy_name}"
FASTWAM_DIR="${POLICY_DIR}/FastWAM"
yaml_file="${POLICY_DIR}/deploy.yml"

# Checkpoint is selected by ckpt_name (not task_name). task_name is only the
# simulator task forwarded to the env client; ckpt_name resolves the trained
# checkpoint, so the same `cotrain` checkpoint can be evaluated across tasks.
ckpt_setting="${FASTWAM_CKPT_SETTING:-${dataset_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}}"
ckpt_dir="${FASTWAM_CKPT_ROOT:-${POLICY_DIR}/checkpoints}/${ckpt_setting}"
weights_dir="${ckpt_dir}/checkpoints/weights"
dataset_stats_path="${FASTWAM_DATASET_STATS_PATH:-${ckpt_dir}/dataset_stats.json}"

allow_dummy_policy="${FASTWAM_ALLOW_DUMMY_POLICY:-false}"
checkpoint_path="${FASTWAM_CHECKPOINT_PATH:-}"
if [[ -z "${checkpoint_path}" && "${allow_dummy_policy}" != "true" ]]; then
    if [[ -d "${weights_dir}" ]]; then
        checkpoint_path=$(find "${weights_dir}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)
    fi
    if [[ -z "${checkpoint_path}" ]]; then
        checkpoint_path="${weights_dir}/step_latest.pt"
    fi
fi
if [[ -n "${checkpoint_path}" && -d "${checkpoint_path}" ]]; then
    indexed_checkpoint=$(find "${checkpoint_path}" -maxdepth 1 -type f -name 'step_*.pt' | sort -V | tail -n 1)
    if [[ -n "${indexed_checkpoint}" ]]; then
        checkpoint_path="${indexed_checkpoint}"
    fi
fi

echo -e "\033[33m[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] checkpoint_path: ${checkpoint_path:-<dummy-policy>}\033[0m"
echo -e "\033[33m[SERVER] dataset_stats_path: ${dataset_stats_path}\033[0m"
echo -e "\033[33m[SERVER] policy_server_host=${policy_server_host} policy_server_port=${policy_server_port}\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[SERVER] action_dim=${action_dim}\033[0m"

# Scope env vars to this server process only; never export them in the
# orchestrator, otherwise they leak into the env client.
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${ROOT_DIR}:${FASTWAM_DIR}:${FASTWAM_DIR}/src:${PYTHONPATH:-}" \
    DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_DIR}/checkpoints" \
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
            checkpoint_path="${checkpoint_path}" \
            dataset_stats_path="${dataset_stats_path}" \
            allow_dummy_policy="${allow_dummy_policy}"
