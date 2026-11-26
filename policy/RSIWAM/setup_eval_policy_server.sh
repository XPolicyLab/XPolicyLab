#!/usr/bin/env bash
set -euo pipefail
bench_name=${1:?bench_name}
task_name=${2:?task_name}
ckpt_name=${3:?ckpt_name}
env_cfg_type=${4:?env_cfg_type}
action_type=${5:?action_type}
seed=${6:?evaluation_seed}
policy_gpu_id=${7:?policy_gpu_id}
policy_conda_env=${8:?policy_conda_env}
policy_server_port=${9:?policy_server_port}
policy_server_host=${10:-localhost}
if [[ "$bench_name" != RoboDojo || "$env_cfg_type" != arx_x5 || "$action_type" != ee ]]; then
    echo 'RSIWAM supports RoboDojo / arx_x5 / ee only.' >&2
    exit 1
fi
[[ "$seed" =~ ^[0-9]+$ ]] || { echo 'Expected a non-negative integer evaluation seed.' >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$policy_conda_env"
# Standard seed is forwarded for run naming and evidence. model.py fixes the
# native inference seed at 1, independently of this environment seed.
exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="$policy_gpu_id" \
    python -u "$XPL_ROOT/setup_policy_server.py" \
    --config_path "$SCRIPT_DIR/deploy.yml" --overrides \
    port="$policy_server_port" host="$policy_server_host" \
    bench_name="$bench_name" task_name="$task_name" ckpt_name="$ckpt_name" \
    env_cfg_type="$env_cfg_type" seed="$seed" policy_name=RSIWAM action_type="$action_type"
