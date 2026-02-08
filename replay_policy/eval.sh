#!/bin/bash
set -e  # 出错即退出

# ==================== 参数定义 ====================
policy_name=replay_policy
task_name=${1}
base_cfg=${2}
ckpt_setting=${3}
gpu_id=${4}
seed=${5}
policy_conda_env=${6} # Conda
sim_conda_env=${7} # Conda

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33m[INFO] GPU ID (to use): ${gpu_id}\033[0m"

cd ../..

yaml_file="policy_lab/${policy_name}/deploy.yml"
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
python policy_lab/setup_policy_server.py \
  --port "${FREE_PORT}" \
  --config_path "${yaml_file}" \
  --overrides \
    task_name="${task_name}" \
    base_cfg="${base_cfg}" \
    ckpt_setting="${ckpt_setting}" \
    seed="${seed}" \
    policy_name="${policy_name}" \
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
python pipeline/deploy.py \
    --task_name "${task_name}" \
    --policy_name "${policy_name}" \
    --base_cfg "${base_cfg}" \
    --port ${FREE_PORT}

echo -e "\033[33m[MAIN] eval_policy_client has finished; cleaning up server.\033[0m"
