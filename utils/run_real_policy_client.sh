#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
free_port="${3}"
dataset_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"

deploy_config_path="${root_dir}/XPolicyLab/policy/${policy_name}/deploy.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

export PYTHONPATH="${root_dir}/src:${root_dir}/XPolicyLab:${root_dir}:${PYTHONPATH:-}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting real robot client to server ${policy_server_ip}:${free_port}...\033[0m"

# Real client always uses arx_x5; runtime maps to test_dual_robot in real_env_client.py.
# eval may pass dual_x5 for policy server — do not forward that here.
client_env_cfg_type="${REAL_CLIENT_ENV_CFG_TYPE:-arx_x5}"

PYTHONWARNINGS=ignore::UserWarning \
python "${root_dir}/src/task_env/real_env_client.py" \
    --dataset_name "${dataset_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${client_env_cfg_type}" \
    --policy_name "${policy_name}" \
    --host "${policy_server_ip}" \
    --port "${free_port}" \
    --seed "${seed}" \
    --eval_batch "${eval_batch}" \
    --additional_info "${additional_info}" \
    --deploy_config "${deploy_config_path}" \
    --root_dir "${root_dir}"
