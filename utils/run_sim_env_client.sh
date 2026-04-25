#!/bin/bash
set -e

eval_batch="$1"
eval_env_conda_env="$2"
free_port="$3"
task_name="$4"
env_cfg_type="$5"
policy_name="$6"
root_dir="$7"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server port ${free_port}...\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
bash "${root_dir}/scripts/eval_policy.sh" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --port "${free_port}" \
    --eval_batch "${eval_batch}" \
    --root_dir "${root_dir}" \
    --device_id 0 # TODO