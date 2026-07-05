#!/bin/bash
set -e

eval_batch="${1}"
eval_env_conda_env="${2}"
free_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-robodojo_ws}"
run_mode="${14:---run-once}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${free_port}...\033[0m"

PYTHONPATH="${root_dir}/XPolicyLab/integrations:${root_dir}/XPolicyLab${PYTHONPATH:+:${PYTHONPATH}}"

CLIENT_ARGS=(
    --bench_name "${bench_name}"
    --task_name "${task_name}"
    --env_cfg_type "${env_cfg_type}"
    --policy_name "${policy_name}"
    --protocol "${protocol}"
    --host "${policy_server_ip}"
    --port "${free_port}"
    --eval_batch "${eval_batch}"
)

if [[ "${run_mode}" == "--run-once" ]]; then
    PYTHONWARNINGS=ignore::UserWarning \
    python "${root_dir}/XPolicyLab/debug_env_client.py" "${CLIENT_ARGS[@]}"
else
    PYTHONWARNINGS=ignore::UserWarning \
    python -m eval_station.servers.env_client_server \
        "${CLIENT_ARGS[@]}" \
        --serve-host 0.0.0.0 \
        --serve-port 19200
fi
