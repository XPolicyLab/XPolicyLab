#!/bin/bash
# Derived from XPolicyLab (Apache License 2.0); see runtime/THIRD_PARTY_NOTICES.md.
# Modifications Copyright © 2026 Martingale. All Rights Reserved.
# This file has been modified by Martingale.
set -euo pipefail

if (( $# < 10 || $# > 11 )); then
    echo "Usage: $0 BENCH TASK LABEL ENV_CFG ACTION SEED GPU CONDA_ENV INFO PORT [HOST]" >&2
    exit 2
fi

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"127.0.0.1"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="${MARTINGALE_XPOLICYLAB_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ "$XPL_ROOT" != /* || ! -f "$XPL_ROOT/utils/setup_env_client.sh" ]]; then
    echo "Set MARTINGALE_XPOLICYLAB_ROOT to the absolute XPolicyLab checkout containing utils/setup_env_client.sh." >&2
    exit 2
fi
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"
if [[ ! -f "$yaml_file" || ! -f "$XPL_ROOT/policy/$policy_name/deploy.py" ]]; then
    echo "Install the Martingale adapter in XPolicyLab/policy/Martingale first." >&2; exit 2
fi
if [[ "$bench_name" != RoboDojo || "$env_cfg_type" != arx_x5 || "$action_type" != joint ]]; then
    echo "This adapter supports RoboDojo with arx_x5 and joint actions." >&2; exit 2
fi
if [[ ! "$policy_server_port" =~ ^[0-9]{1,5}$ ]] || (( 10#$policy_server_port < 1 || 10#$policy_server_port > 65535 )); then
    echo "Policy port must be an integer between 1 and 65535." >&2; exit 2
fi

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

cd "$XPL_ROOT"
bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
