#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export FLEXWAM_MODEL_BASE_PATH="${FLEXWAM_MODEL_BASE_PATH:-/path/to/wan-base}"
export DIFFSYNTH_MODEL_BASE_PATH="${FLEXWAM_MODEL_BASE_PATH}"

RUN_DIR="${RUN_DIR:-/path/to/run}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python -m deploy.server \
  --task wan/piper_penbox \
  --robot-type piper \
  --checkpoint-path "${CHECKPOINT_PATH:-${RUN_DIR}/checkpoints/weights/step.pt}" \
  --dataset-stats-path "${DATASET_STATS_PATH:-${RUN_DIR}/dataset_stats.json}" \
  --host 0.0.0.0 \
  --port 8006 \
  --default-prompt "Put the pens on the table into the pen box and zip it closed." \
  --action-horizon 32 \
  --num-inference-steps 10 \
  --delta \
  --cuda-graph
