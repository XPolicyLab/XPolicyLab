#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export FLEXWAM_LINGBOT_PATH="${FLEXWAM_LINGBOT_PATH:-/path/to/lingbot-video}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}"

RUN_DIR="${RUN_DIR:-/path/to/run}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
python -m deploy.server \
  --task wan/flexiv_cable \
  --robot-type flexiv \
  --checkpoint-path "${CHECKPOINT_PATH:-${RUN_DIR}/checkpoints/weights/step.pt}" \
  --dataset-stats-path "${DATASET_STATS_PATH:-${RUN_DIR}/dataset_stats.json}" \
  --host 0.0.0.0 \
  --port 8006 \
  --default-prompt "insert the Ethernet cable into the network port with precise pose alignment." \
  --action-horizon 32 \
  --num-inference-steps 10 \
  --delta \
  --cuda-graph
