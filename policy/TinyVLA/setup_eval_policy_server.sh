#!/bin/bash
set -e

bench_name=$1
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

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}"
ckpt_root="${SCRIPT_DIR}/checkpoints/${ckpt_setting}"

# train.sh writes a series of `checkpoint-<step>` directories under ckpt_root;
# always evaluate the most recent one.
model_path="$(ls -d "${ckpt_root}"/checkpoint-* | sort -V | tail -n1)"
model_base="${ckpt_root}/pretrained_vlm"
stats_path="${ckpt_root}/dataset_stats.pkl"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] model_path=${model_path}"

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
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            model_path="${model_path}" \
            model_base="${model_base}" \
            stats_path="${stats_path}"
