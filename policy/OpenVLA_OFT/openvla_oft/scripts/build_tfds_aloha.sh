#!/usr/bin/env bash
set -euo pipefail


DATASET_NAME=aloha_${1}
RAW_DATA_DIR=${2}
PREPROCESSED_BASE_DIR=${3}
PERCENT_VAL=${4}
GPU_ID=${5}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" # change to your path
TFDS_DATA_DIR="${TFDS_DATA_DIR:-${REPO_ROOT}/tensorflow_datasets}" # change to your path

export CUDA_VISIBLE_DEVICES=${GPU_ID}

cd "${REPO_ROOT}"

RAW_DATA_BASENAME=$(basename "${RAW_DATA_DIR%/}")
PREPROCESSED_DIR="${PREPROCESSED_BASE_DIR%/}/${RAW_DATA_BASENAME}"

python experiments/robot/aloha/preprocess_split_aloha_data.py --dataset_path "${RAW_DATA_DIR}" \
  --out_base_dir "${PREPROCESSED_BASE_DIR}" \
  --percent_val "${PERCENT_VAL}"

mkdir -p "${TFDS_DATA_DIR}"

python rlds_dataset_builder/aloha_datasets/build_aloha_dataset.py \
  --dataset_name "${DATASET_NAME}" \
  --preprocessed_dir "${PREPROCESSED_DIR}" \
  --tfds_data_dir "${TFDS_DATA_DIR}" \
  --overwrite

python scripts/register_aloha_dataset.py --dataset_name "${DATASET_NAME}" --repo_root "${REPO_ROOT}"
