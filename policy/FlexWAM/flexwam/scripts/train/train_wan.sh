#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export FLEXWAM_TMPDIR="${FLEXWAM_TMPDIR:-${HOME}/tmp}"
source "${SCRIPT_DIR}/env_tmpdir.sh"

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29521}"
export FLEXWAM_OUTPUT_BASE_DIR="${FLEXWAM_OUTPUT_BASE_DIR:-./runs}"
export FLEXWAM_MODEL_BASE_PATH="${FLEXWAM_MODEL_BASE_PATH:-/path/to/fastwam}"
export DIFFSYNTH_MODEL_BASE_PATH="${FLEXWAM_MODEL_BASE_PATH}"
export FLEXWAM_ACTION_DIT_PATH="${FLEXWAM_MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

TASK=wan/flexiv_cable

DATASET_DIR="${DATASET_DIR:-/path/to/flexiv_cable}"

python scripts/preprocess/precompute_text_embeds.py \
  "task=${TASK}" \
  "data.roots=[${DATASET_DIR}]" \
  "data.val_roots=[${DATASET_DIR}]" \
  "$@"

NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}" NCCL_ALGO="${NCCL_ALGO:-Ring}" \
bash scripts/train/train_zero1.sh 8 \
  "task=${TASK}" \
  "data.roots=[${DATASET_DIR}]" \
  "data.val_roots=[${DATASET_DIR}]" \
  "$@"
