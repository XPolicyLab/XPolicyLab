#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVENTVLA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ -n "${EVENTVLA_PYTHON:-}" ]]; then
  EVENTVLA_PYTHON_BIN="${EVENTVLA_PYTHON}"
else
  EVENTVLA_PYTHON_BIN="python"
fi

# Usage:
#   bash examples/RoboTwin-Mem/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port]
# Supported checkpoints:
#   - EventVLA pure_image_keyframe_memory
#   - legacy starVLA QwenOFT pure_image_keyframe_memory
your_ckpt=${1:-${POLICY_CKPT_PATH:-}}
gpu_id=${2:-${GPU_ID:-1}}
port=${3:-${PORT:-5840}}
use_bf16=${USE_BF16:-1}

if [[ -z "${your_ckpt}" ]]; then
  echo "[eval][error] Missing checkpoint path."
  echo "Usage: bash examples/RoboTwin-Mem/eval_files/run_policy_server.sh <ckpt_path> [gpu_id] [port]"
  exit 1
fi

if [[ ! -f "${your_ckpt}" ]]; then
  echo "[eval][error] Checkpoint not found: ${your_ckpt}"
  exit 1
fi

################# star Policy Server ######################

# export DEBUG=true
echo "[eval] ckpt_path=${your_ckpt}"
echo "[eval] gpu_id=${gpu_id}"
echo "[eval] port=${port}"
echo "[eval] python=${EVENTVLA_PYTHON_BIN}"

export PYTHONPATH="${EVENTVLA_ROOT}:${PYTHONPATH:-}"

cd "$EVENTVLA_ROOT"

bf16_flags=()
if [[ "${use_bf16}" == "1" ]]; then
  bf16_flags=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${EVENTVLA_PYTHON_BIN}" deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    "${bf16_flags[@]}"

# #################################
