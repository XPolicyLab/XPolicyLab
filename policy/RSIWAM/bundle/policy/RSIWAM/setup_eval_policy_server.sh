#!/usr/bin/env bash
set -euo pipefail
bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?seed}
policy_gpu_id=${7:?policy_gpu_id}
policy_conda_env=${8:?policy_conda_env}
policy_server_port=${9:?policy_server_port}
policy_server_host=${10:-127.0.0.1}
[[ "$bench_name" == RoboDojo && "$ckpt_name" == rsiwam-original32-lora33-step400 ]]
[[ "$env_cfg_type" == arx_x5 && "$action_type" == ee ]]
# This positional seed belongs to the environment evaluation. The model's
# inference seed stays fixed at 1 in deploy.yml and is not overridden here.
[[ "$seed" =~ ^[0-9]+$ ]]
: "${RSIWAM_XPL_ROOT:?Set RSIWAM_XPL_ROOT to XPolicyLab}"
: "${RSIWAM_BASE_WEIGHTS:?Set RSIWAM_BASE_WEIGHTS to external checkpoints}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$policy_conda_env"
exec "${CONDA_PREFIX}/bin/python" -u "${SCRIPT_DIR}/launch_policy.py" \
  --task "$task_name" --gpu "$policy_gpu_id" --port "$policy_server_port" \
  --host "$policy_server_host" --xpl-root "$RSIWAM_XPL_ROOT"
