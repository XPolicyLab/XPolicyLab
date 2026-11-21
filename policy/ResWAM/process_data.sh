#!/bin/bash
set -euo pipefail

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [policy_env]
# Builds ResWAM's dataset contract (dataset.json, clean index, codec_statistics.json) from the
# official RoboDojo LeRobot export (RoboDojo-Benchmark/RoboDojo on Hugging Face). Set
# RESWAM_DATA_ROOT to the directory holding robodojo/ (default: policy/ResWAM/data).
bench_name=${1}
ckpt_name=${2}
env_cfg_type=${3}
action_type=${4}
policy_env=${5:-${RESWAM_POLICY_ENV:-}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${RESWAM_REPO:-${SCRIPT_DIR}/ResWAM}"
PYTHON_BIN=python
[[ -n "${policy_env}" ]] && PYTHON_BIN="${policy_env}/bin/python"

if [[ "${bench_name}" != "RoboDojo" || "${env_cfg_type}" != "arx_x5" || "${action_type}" != "joint" ]]; then
    echo "ResWAM supports RoboDojo / arx_x5 / joint, got ${bench_name} / ${env_cfg_type} / ${action_type}" >&2
    exit 1
fi

export RESWAM_DATA_ROOT="${RESWAM_DATA_ROOT:-${SCRIPT_DIR}/data}"
cd "${UPSTREAM_DIR}"
"${PYTHON_BIN}" scripts/data_prep/prepare_robodojo.py \
    --dataset-root "${RESWAM_DATA_ROOT}/robodojo" \
    --output-dir "${RESWAM_DATA_ROOT}/prepared/robodojo"
echo "[process_data] ${ckpt_name}: prepared data in ${RESWAM_DATA_ROOT}/prepared/robodojo"
