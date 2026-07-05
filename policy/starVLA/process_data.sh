#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# expert_data_num: optional; empty = use all episodes

if [[ $# -lt 4 ]]; then
    echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
    echo "Example: bash process_data.sh RoboDojo stack_bowls arx_x5 joint" >&2
    echo "Example: bash process_data.sh RoboDojo stack_bowls arx_x5 joint 50" >&2
    exit 1
fi

bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
out_tag="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

echo "[process_data] ${bench_name}/${ckpt_name}/${env_cfg_type} expert_data_num=${expert_data_num:-<all>} (${action_type}) -> data/${out_tag}/"

py_args=(
    "${SCRIPT_DIR}/source_starvla/examples/XPolicyLab/train_files/convert_xpolicy_to_lerobot3.py"
    --root_dir "${ROOT_DIR}"
    --bench_name "${bench_name}"
    --ckpt_name "${ckpt_name}"
    --env_cfg_type "${env_cfg_type}"
    --action_type "${action_type}"
    --output_dir "${SCRIPT_DIR}/data/${out_tag}"
)
if [[ -n "${expert_data_num}" ]]; then
    py_args+=(--expert_data_num "${expert_data_num}")
fi

python "${py_args[@]}"
