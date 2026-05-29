#!/bin/bash
set -e

# Mem_0 evaluation: start the policy model server in the policy conda env, then
# run the XPolicyLab debug client (single-environment) in eval_env_conda_env.
#
# Usage:
#   bash eval.sh <dataset_name> <task_name> <env_cfg_type> <expert_data_num> \
#                <action_type> <gpu_id> <seed> <policy_conda_env> <eval_env_conda_env>
# Example (debug-client wiring check, dual-arm joint):
#   bash eval.sh RoboDojo arx_x5_task arx_x5 50 joint 0 0 mem0 XPolicyLab
#
# Optional env overrides (real rollouts):
#   EXECUTION_CKPT=checkpoints/<run>/model.pt \
#   STATE_STATS_PATH=Mem_0/assets/<task>/norm_stats.json \
#   GLOBAL_TASK="..." VLLM_URL="http://host:8000" bash eval.sh ...

policy_name=Mem_0
dataset_name=${1}
task_name=${2}
env_cfg_type=${3}
expert_data_num=${4}
action_type=${5}
gpu_id=${6}
seed=${7}
policy_conda_env=${8}
eval_env_conda_env=${9}

# Single-environment debug client (set true only after wiring batch methods).
eval_batch=false
additional_info=""

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
yaml_file="${ROOT_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}")
echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
FREE_PORT=$(bash "${UTILS_DIR}/get_free_port.sh")

cleanup(){ [[ -n "${SERVER_PID:-}" ]] && echo -e "\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m" && kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

# ==================== model server (policy env) ====================
echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# Optional checkpoint / stats / planner overrides (only forwarded when set).
extra_overrides=()
[[ -n "${EXECUTION_CKPT:-}" ]]   && extra_overrides+=( execution_ckpt="${EXECUTION_CKPT}" )
[[ -n "${STATE_STATS_PATH:-}" ]] && extra_overrides+=( state_stats_path="${STATE_STATS_PATH}" )
[[ -n "${GLOBAL_TASK:-}" ]]      && extra_overrides+=( global_task="${GLOBAL_TASK}" )
[[ -n "${VLLM_URL:-}" ]]         && extra_overrides+=( vllm_url="${VLLM_URL}" )

echo -e "\033[32m[SERVER] Launching policy model server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
CUDA_VISIBLE_DEVICES="${gpu_id}" python "${ROOT_DIR}/XPolicyLab/setup_policy_server.py" \
    --config_path "${yaml_file}" \
    --overrides \
        port="${FREE_PORT}" \
        dataset_name="${dataset_name}" \
        task_name="${task_name}" \
        env_cfg_type="${env_cfg_type}" \
        expert_data_num="${expert_data_num}" \
        seed="${seed}" \
        policy_name="${policy_name}" \
        action_type="${action_type}" \
        action_dim="${action_dim}" \
        "${extra_overrides[@]}" \
    &
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

# ==================== debug client (eval env) ====================
CUDA_VISIBLE_DEVICES="${gpu_id}" bash "${UTILS_DIR}/run_debug_env_client.sh" \
    "${eval_batch}" "${eval_env_conda_env}" "${FREE_PORT}" \
    "${dataset_name}" "${task_name}" "${env_cfg_type}" "${policy_name}" \
    "${additional_info}" "${ROOT_DIR}" "${seed}" "${gpu_id}"
echo -e "\033[33m[MAIN] debug client finished; cleaning up server.\033[0m"
