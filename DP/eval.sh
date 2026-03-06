#!/bin/bash
set -e  # 出错即退出

# ==================== 参数定义 ====================
policy_name=DP
task_name=${1}
env_cfg=${2}
expert_data_num=${3}
action_type=${4}
gpu_id=${5}
seed=${6}
policy_conda_env=${7} # Conda
sim_conda_env=${8} # Conda

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_path="${SCRIPT_DIR}/data/${task_name}-${env_cfg}-${expert_data_num}-${action_type}.zarr"

ZARR="${data_path}/data/action/.zarray"
action_dim=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["chunks"][1])' "$ZARR")

cd ../..

yaml_file="XPolicyLab/${policy_name}/deploy.yml"
echo -e "\033[33m[INFO] Using config file: ${yaml_file}\033[0m"

# ==================== 动态端口分配 ====================
FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33m[INFO] Using socket port: ${FREE_PORT}\033[0m"

# ==================== 启动 server ====================
echo -e "\033[32m[SERVER] Activating Conda environment: ${policy_conda_env}\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

echo -e "\033[32m[SERVER] Launching policy_model_server in background...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python XPolicyLab/setup_policy_server.py \
  --config_path "${yaml_file}" \
  --overrides \
    port="${FREE_PORT}" \
    task_name="${task_name}" \
    env_cfg="${env_cfg}" \
    expert_data_num="${expert_data_num}" \
    seed="${seed}" \
    policy_name="${policy_name}" \
    action_type="${action_type}" \
    action_dim="${action_dim}" \
  &
SERVER_PID=$!
echo -e "\033[32m[SERVER] PID=${SERVER_PID} (running in background)\033[0m"

# ==================== 清理机制 ====================
trap "echo -e '\033[31m[CLEANUP] Killing server PID=${SERVER_PID}\033[0m'; kill ${SERVER_PID} 2>/dev/null" EXIT

# # ==================== 启动 client ====================
conda deactivate
conda activate "${sim_conda_env}"
echo -e "\033[34m[CLIENT] Activating Conda environment: ${sim_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server port ${FREE_PORT}...\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python XPolicyLab/debug_policy_env.py \
    --task_name "${task_name}" \
    --policy_name "${policy_name}" \
    --env_cfg "${env_cfg}" \
    --port ${FREE_PORT}

echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"