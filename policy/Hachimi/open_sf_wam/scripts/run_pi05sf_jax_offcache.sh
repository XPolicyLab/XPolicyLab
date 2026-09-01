#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# XPolicy Pi_05 / openpi-SF aligned JAX Pi05-SF config.
CONFIG_NAME="${CONFIG_NAME:-pi05sf_jax_robodojo_v21_offcache}"
EXP_NAME="${EXP_NAME:-pi05sf_robodojo_v21_offcache_sf_0.2_jax}"

# XPolicy Pi_05 uses the RoboDojo LeRobot v2.1 dataset.
LEROBOT_HOME="${HF_LEROBOT_HOME:-${XPL_DATA_ROOT:-${ROOT_DIR}/../data}}"

# Pi05 base JAX checkpoint. This must contain params/ and assets/.
PI05_BASE_PATH="${PI05_BASE_PATH:-${ROOT_DIR}/checkpoints/pi05_base}"

# Ordinary VGGT checkpoint directory. It must contain model.pt.
VGGT_WEIGHT_PATH="${VGGT_WEIGHT_PATH:-${ROOT_DIR}/checkpoints/VGGT-1B}"

# Required: openpi-SF compatible chunked VGGT feature cache directory.
SF_CACHE_DIR="${SF_CACHE_DIR:-${ROOT_DIR}/results/sf_cache}"
SF_CACHE_SAVE_DTYPE="${SF_CACHE_SAVE_DTYPE:-bf16}"
SF_CACHE_CHUNK_SIZE="${SF_CACHE_CHUNK_SIZE:-128}"
SF_DATASET_UID="${SF_DATASET_UID:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-60000}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ -z "${SF_CACHE_DIR}" ]]; then
  cat >&2 <<'EOF'
SF_CACHE_DIR is required.
Example:
  SF_CACHE_DIR=/path/to/openpi_sf_chunked_cache scripts/run_pi05sf_jax_offcache.sh
EOF
  exit 2
fi

if [[ ! -d "${SF_CACHE_DIR}" ]]; then
  echo "SF_CACHE_DIR does not exist: ${SF_CACHE_DIR}" >&2
  exit 2
fi

if [[ ! -d "${PI05_BASE_PATH}/params" || ! -d "${PI05_BASE_PATH}/assets" ]]; then
  echo "PI05_BASE_PATH must contain params/ and assets/: ${PI05_BASE_PATH}" >&2
  exit 2
fi

if [[ ! -f "${VGGT_WEIGHT_PATH}/model.pt" ]]; then
  echo "VGGT model.pt not found under VGGT_WEIGHT_PATH: ${VGGT_WEIGHT_PATH}" >&2
  exit 2
fi

LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"

export HF_LEROBOT_HOME="${LEROBOT_HOME}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${LOCAL_CACHE_ROOT}/hf/datasets}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${LOCAL_CACHE_ROOT}/jax}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

echo "Running JAX Pi05-SF with offline VGGT feature cache"
echo "  CONFIG_NAME=${CONFIG_NAME}"
echo "  EXP_NAME=${EXP_NAME}"
echo "  LEROBOT_HOME=${LEROBOT_HOME}"
echo "  PI05_BASE_PATH=${PI05_BASE_PATH}"
echo "  VGGT_WEIGHT_PATH=${VGGT_WEIGHT_PATH}"
echo "  SF_CACHE_DIR=${SF_CACHE_DIR}"
echo "  SF_CACHE_SAVE_DTYPE=${SF_CACHE_SAVE_DTYPE}"
echo "  SF_CACHE_CHUNK_SIZE=${SF_CACHE_CHUNK_SIZE}"
echo "  SF_DATASET_UID=${SF_DATASET_UID}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  TRAIN_STEPS=${TRAIN_STEPS}"
echo "  NUM_WORKERS=${NUM_WORKERS}"

uv run --no-sync scripts/train_align.py "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --num-train-steps "${TRAIN_STEPS}" \
  --align-target-model vggt \
  --vggt-weight-path "${VGGT_WEIGHT_PATH}" \
  --use-camera-params False \
  --use-vggt-pe True \
  --use-vlm-norm True \
  --align-loss-coeff 0.2 \
  --sf-cache-enable \
  --sf-cache-mode readonly \
  --sf-cache-miss-policy error \
  --sf-cache-dir "${SF_CACHE_DIR}" \
  --sf-cache-save-dtype "${SF_CACHE_SAVE_DTYPE}" \
  --sf-cache-chunk-size "${SF_CACHE_CHUNK_SIZE}" \
  --sf-dataset-uid "${SF_DATASET_UID}" \
  "$@"
