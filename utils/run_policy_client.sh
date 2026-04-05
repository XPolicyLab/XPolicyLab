#!/bin/bash
set -e

sim_conda_env="$1"
free_port="$2"
task_name="$3"
env_cfg_type="$4"
policy_name="$5"
root_dir="$6"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${sim_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${sim_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server port ${free_port}...\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
bash "${root_dir}/scripts/eval_policy.sh" \
    --root_dir "${root_dir}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --port "${free_port}" \
    --device_id 0