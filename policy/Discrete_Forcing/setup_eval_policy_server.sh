#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 9 || $# -gt 10 ]]; then
    echo "Usage: $0 bench task checkpoint env_cfg_type action_type seed policy_gpu policy_env port [host]" >&2
    exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$8"
export PYTHONPATH="${XPL_ROOT}/..:${XPL_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
unset RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset PMI_RANK PMI_SIZE OMPI_COMM_WORLD_RANK OMPI_COMM_WORLD_SIZE
exec env CUDA_VISIBLE_DEVICES="$7" python "$XPL_ROOT/setup_policy_server.py" \
    --config_path "$SCRIPT_DIR/deploy.yml" --overrides \
    bench_name="$1" task_name="$2" ckpt_name="$3" env_cfg_type="$4" \
    action_type="$5" seed="$6" gpu_id="$7" port="$9" host="${10:-localhost}" \
    policy_name=Discrete_Forcing
