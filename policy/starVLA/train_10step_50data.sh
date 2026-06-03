#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash train_10step_50data.sh <gpu_id>"
    echo "Example: bash train_10step_50data.sh 0,1,2,3"
    exit 1
fi

gpu_id=${1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"
STARVLA_ADAPTER="${SCRIPT_DIR}/starvla_adapter"
CONFIG_YAML="${SCRIPT_DIR}/xpolicy_oft_vla.yaml"
DATA_ROOT="${SCRIPT_DIR}/data/RoboDojo-stack_bowls-arx_x5-50-joint"
DATASET_DIR="${DATA_ROOT}/arx_x5"
num_processes=$(awk -F',' '{print NF}' <<< "${gpu_id}")

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "[starVLA] missing 50-data dataset: ${DATASET_DIR}"
    echo "[starVLA] run first: bash process_data.sh RoboDojo stack_bowls arx_x5 50 joint"
    exit 1
fi

# The dataloader now uses stats_xpolicy.json. Reuse the old 50-data stats cache
# if it already exists, so this smoke test starts quickly.
if [[ ! -f "${DATASET_DIR}/meta/stats_xpolicy.json" && -f "${DATASET_DIR}/meta/stats_gr00t.json" ]]; then
    cp "${DATASET_DIR}/meta/stats_gr00t.json" "${DATASET_DIR}/meta/stats_xpolicy.json"
fi

echo "[starVLA] config_yaml=${CONFIG_YAML}"
echo "[starVLA] data_root=${DATA_ROOT}"
echo "[starVLA] data_mix=arx_x5"
echo "[starVLA] run_id=xpolicy_oft_50data_smoke10"
echo "[starVLA] train_entry=starVLA/training/train_starvla.py"
echo "[starVLA] num_processes=${num_processes}, mixed_precision=bf16"

cd "${STARVLA_ROOT}"
STARVLA_EXTRA_DATA_REGISTRY="${STARVLA_ADAPTER}/data_registry" \
PYTHONPATH="${STARVLA_ADAPTER}:${STARVLA_ROOT}:${PYTHONPATH:-}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}" \
CUDA_VISIBLE_DEVICES="${gpu_id}" accelerate launch \
    --num_processes "${num_processes}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    starVLA/training/train_starvla.py \
    --config_yaml "${CONFIG_YAML}" \
    --run_id xpolicy_oft_50data_smoke10 \
    --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
    --datasets.vla_data.data_mix arx_x5 \
    --trainer.max_train_steps 10 \
    --trainer.num_warmup_steps 0 \
    --trainer.save_interval 10 \
    --trainer.eval_interval 999999 \
    --trainer.logging_frequency 1
