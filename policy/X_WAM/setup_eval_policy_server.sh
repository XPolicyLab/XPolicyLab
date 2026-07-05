#!/bin/bash
set -euo pipefail

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
policy_server_host=${11:-localhost}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
POLICY_DIR="${ROOT_DIR}/XPolicyLab/policy/${policy_name}"
XWAM_DIR="${POLICY_DIR}/X-WAM"
yaml_file="${POLICY_DIR}/deploy.yml"

# X-WAM checkpoints are experiment directories (config.yaml + checkpoints/<steps>.ckpt/...).
# ckpt_name selects the experiment; the same checkpoint can be evaluated across tasks.
exp_setting="${XWAM_EXP_SETTING:-${bench_name}-${ckpt_name}-${env_cfg_type}-${expert_data_num}-${action_type}-${seed}}"
exp_path="${XWAM_EXP_PATH:-${XWAM_CKPT_ROOT:-${POLICY_DIR}/checkpoints}/${exp_setting}}"
steps="${XWAM_STEPS:-last}"

# Wan2.2-TI2V-5B base weights (T5 + VAE + DiT). Falls back to config.yaml value if unset.
wan_checkpoint_dir="${XWAM_WAN_CHECKPOINT_DIR:-}"

allow_dummy_policy="${XWAM_ALLOW_DUMMY_POLICY:-false}"

echo -e "\033[33m[SERVER] policy=${policy_name}, task=${task_name}, ckpt=${ckpt_name}\033[0m"
echo -e "\033[33m[SERVER] exp_path: ${exp_path} (steps=${steps})\033[0m"
echo -e "\033[33m[SERVER] wan_checkpoint_dir: ${wan_checkpoint_dir:-<from config.yaml>}\033[0m"
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
    TOKENIZERS_PARALLELISM=false \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    PYTHONPATH="${ROOT_DIR}:${XWAM_DIR}:${XWAM_DIR}/evaluation:${PYTHONPATH:-}" \
    python -u "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            expert_data_num="${expert_data_num}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}" \
            exp_path="${exp_path}" \
            steps="${steps}" \
            wan_checkpoint_dir="${wan_checkpoint_dir}" \
            allow_dummy_policy="${allow_dummy_policy}"
