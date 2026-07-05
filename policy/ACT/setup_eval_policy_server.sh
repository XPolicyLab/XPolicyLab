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

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_DIR="$(cd "${CURRENT_DIR}/../../.." && pwd)"

policy_name="$(basename "${CURRENT_DIR}")"
yaml_file="${XPL_DIR}/XPolicyLab/policy/${policy_name}/deploy.yml"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

action_dim=$(
    PYTHONPATH="${XPL_DIR}" python -c "
import sys
from XPolicyLab.utils.process_data import get_action_dim
print(get_action_dim(sys.argv[1]))
" "${env_cfg_type}"
)
export ACT_ACTION_DIM="${action_dim}"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_DIR}/XPolicyLab/setup_policy_server.py" \
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
            action_dim="${action_dim}"
