#!/bin/bash
# Derived from XPolicyLab (Apache License 2.0); see runtime/THIRD_PARTY_NOTICES.md.
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.
set -euo pipefail

if (( $# < 9 || $# > 10 )); then
    echo "Usage: $0 BENCH TASK LABEL ENV_CFG ACTION SEED GPU CONDA_ENV PORT [HOST]" >&2
    echo "Use CONDA_ENV=- for the active Python environment." >&2
    exit 2
fi

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"127.0.0.1"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="${MARTINGALE_XPOLICYLAB_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ "$XPL_ROOT" != /* || ! -f "$XPL_ROOT/setup_policy_server.py" ]]; then
    echo "Set MARTINGALE_XPOLICYLAB_ROOT to the absolute XPolicyLab checkout containing setup_policy_server.py." >&2
    exit 2
fi

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"
if [[ ! -f "$yaml_file" || ! -f "$XPL_ROOT/policy/$policy_name/model.py" ]]; then
    echo "Copy the Martingale policy directory to $XPL_ROOT/policy/Martingale first." >&2
    exit 2
fi
if [[ "$bench_name" != RoboDojo || "$env_cfg_type" != arx_x5 || "$action_type" != joint ]]; then
    echo "This adapter supports RoboDojo with arx_x5 and joint actions." >&2
    exit 2
fi
if [[ ! "$policy_server_port" =~ ^[0-9]{1,5}$ ]] || (( 10#$policy_server_port < 1 || 10#$policy_server_port > 65535 )); then
    echo "Policy port must be an integer between 1 and 65535." >&2; exit 2
fi

# ONE inference configuration and ONE library for every task: these variables
# are deployment-wide and are never chosen from task_name, seed or layout.
export MARTINGALE_INFERENCE_CONFIG="${MARTINGALE_INFERENCE_CONFIG:-$SCRIPT_DIR/inference.json}"
: "${MARTINGALE_OUTPUT_ROOT:?absolute private output directory for episode records}"
export SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-${XDG_CACHE_HOME:-$HOME/.cache}/martingale/sam3.pt}"
for input in "$MARTINGALE_INFERENCE_CONFIG" "$SAM3_CHECKPOINT_PATH"; do
    if [[ "$input" != /* || ! -r "$input" || ! -f "$input" ]]; then
        echo "Required deployment input must be an absolute readable file: $input" >&2; exit 2
    fi
done
if [[ "$MARTINGALE_OUTPUT_ROOT" != /* ]]; then
    echo "MARTINGALE_OUTPUT_ROOT must be absolute." >&2; exit 2
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

if [[ "$policy_conda_env" != - ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${policy_conda_env}"
fi

# XPolicyLab imports policy.Martingale from its own working directory.
cd "$XPL_ROOT"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
