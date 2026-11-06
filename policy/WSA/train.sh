#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
    echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
    exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

if [[ "${action_type}" != "joint" ]]; then
    echo "[ERROR] WSA RoboTwin checkpoints support only XPolicyLab action_type=joint." >&2
    exit 1
fi
if [[ "${env_cfg_type}" != "aloha_agilex" ]]; then
    echo "[ERROR] WSA RoboTwin training supports only env_cfg_type=aloha_agilex." >&2
    exit 1
fi

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
WSA_DIR="${WSA_ROOT:-${POLICY_DIR}/WSA}"
UTILS_DIR="${XPL_ROOT}/utils"

variant="${WSA_VARIANT:-base}"
action_mode="${WSA_ACTION_MODE:-delta}"
dataset_id="${WSA_DATASET_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
robotwin_root="${WSA_ROBOTWIN_ROOT:-${WSA_LEROBOT_HOME:-${POLICY_DIR}/data/lerobot}}"
stats_root="${WSA_STATS_ROOT:-${POLICY_DIR}/data/norm_stats}"
stats_path="${stats_root}/aloha/${action_mode}/stats.json"
run_name="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
action_dim="$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")"

if [[ ! -d "${robotwin_root}/${dataset_id}/meta" && -z "$(find -L "${robotwin_root}" -path '*/meta/info.json' -print -quit 2>/dev/null)" ]]; then
    echo "[ERROR] No LeRobot v3.0 datasets found under ${robotwin_root}." >&2
    echo "Run process_data.sh first or set WSA_ROBOTWIN_ROOT." >&2
    exit 1
fi
if [[ ! -f "${stats_path}" ]]; then
    echo "[ERROR] Missing normalization stats: ${stats_path}" >&2
    echo "Run process_data.sh first or set WSA_STATS_ROOT." >&2
    exit 1
fi

gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PROC_PER_NODE="${WSA_PROC_PER_NODE:-${gpu_count}}"
export ROBOTWIN_ROOT="${robotwin_root}"
export ACTION_TYPE="${action_mode}"
export TRAIN_SEED="${seed}"
export BASE_OUTPUT_DIR="${POLICY_DIR}/checkpoints"
export JOB_NAME="${run_name}"
export DATASET_EXTERNAL_STATS_ROOT="${stats_root}"
export NORMALIZATION_STATS_PATH="${stats_path}"
export ACTION_DIM="${action_dim}"
export PROPRIO_DIM="${action_dim}"

case "${variant}" in
    base)
        export POLICY_INIT_PATH="${WSA_POLICY_INIT_PATH:-zaleni/WSA-Base}"
        bash "${WSA_DIR}/launch/wsa_base_finetune_robotwin.sh"
        ;;
    large)
        export POLICY_INIT_PATH="${WSA_POLICY_INIT_PATH:-zaleni/WSA-Large}"
        export SKIP_DIT_LOAD_FROM_PRETRAIN="${SKIP_DIT_LOAD_FROM_PRETRAIN:-true}"
        bash "${WSA_DIR}/launch/wsa_large_finetune_robotwin.sh"
        ;;
    *)
        echo "[ERROR] WSA_VARIANT must be base or large, got ${variant}." >&2
        exit 1
        ;;
esac

echo "[WSA] checkpoint run directory: ${POLICY_DIR}/checkpoints/${run_name}"
