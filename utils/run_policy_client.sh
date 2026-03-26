#!/bin/bash
set -e

sim_conda_env="$1"
free_port="$2"
task_name="$3"
env_cfg="$4"
policy_name="$5"
root_dir="$6"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${sim_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${sim_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server port ${free_port}...\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python "${root_dir}/XPolicyLab/debug_policy_env.py" \
    --task_name "${task_name}" \
    --env_cfg "${env_cfg}" \
    --policy_name "${policy_name}" \
    --port "${free_port}"