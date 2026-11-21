#!/bin/bash
set -euo pipefail

# Usage: bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_ids> [policy_env]
# Runs ResWAM's two-stage RoboDojo schedule (stage 1: 0 -> 200k, stage 2: 200k -> 400k at the
# floor LR; the released checkpoint is stage-2 step 328k) on the comma-separated <gpu_ids>.
# Run directories go to checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/{robodojo,robodojo_stage2}.
# <seed> only names the run; the training seed stays at the paper value (20260907, configs/robodojo.yaml).
# RESWAM_STAGE=1|2|all selects the stage(s) (default: all).
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_ids=${6}
policy_env=${7:-${RESWAM_POLICY_ENV:-}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${RESWAM_REPO:-${SCRIPT_DIR}/ResWAM}"
if [[ "${bench_name}" != "RoboDojo" || "${env_cfg_type}" != "arx_x5" || "${action_type}" != "joint" ]]; then
    echo "ResWAM supports RoboDojo / arx_x5 / joint, got ${bench_name} / ${env_cfg_type} / ${action_type}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${gpu_ids}")}"
export RESWAM_DATA_ROOT="${RESWAM_DATA_ROOT:-${SCRIPT_DIR}/data}"
export RESWAM_OUTPUT_DIR="${SCRIPT_DIR}/checkpoints/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
[[ -n "${policy_env}" ]] && export RESWAM_PYTHON="${policy_env}/bin/python"
mkdir -p "${RESWAM_OUTPUT_DIR}"

stage=${RESWAM_STAGE:-all}
if [[ "${stage}" == "1" || "${stage}" == "all" ]]; then
    RESWAM_STAGE=1 bash "${UPSTREAM_DIR}/scripts/train/train_robodojo.sh"
fi
if [[ "${stage}" == "2" || "${stage}" == "all" ]]; then
    RESWAM_STAGE=2 bash "${UPSTREAM_DIR}/scripts/train/train_robodojo.sh"
fi
