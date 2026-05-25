#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <repo_id> <dataset_root> <model_weights_root> [device] [t5_folder_name] [max_episodes] [overwrite] [strip_parquet_metadata]"
    echo "Example: $0 robodojo_sim /vepfs-cnbje63de6fae220/xspark_shared/lerobot/robodojo_sim /vepfs-cnbje63de6fae220/xspark_shared/model_weights cuda t5_embedding 0 false false"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_SH="/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/etc/profile.d/conda.sh"
MOTUS_PYTHON_DEFAULT="/vepfs-cnbje63de6fae220/xspark_shared/miniconda3/envs/motus/bin/python"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate motus
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate motus
fi

PYTHON_BIN="${PYTHON_BIN:-$MOTUS_PYTHON_DEFAULT}"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(command -v python)"
fi

REPO_ID="${1}"
DATASET_ROOT="${2}"
MODEL_WEIGHTS_ROOT="${3}"
DEVICE="${4:-cuda}"
T5_FOLDER_NAME="${5:-t5_embedding}"
MAX_EPISODES="${6:-0}"
OVERWRITE_FLAG="${7:-false}"
STRIP_FLAG="${8:-false}"

CMD=(
    "$PYTHON_BIN" data/lerobot/add_t5_cache_to_lerobot_dataset.py
    --repo_id "$REPO_ID"
    --root "$DATASET_ROOT"
    --wan_path "$MODEL_WEIGHTS_ROOT"
    --device "$DEVICE"
    --t5_folder_name "$T5_FOLDER_NAME"
    --max_episodes "$MAX_EPISODES"
)

if [ "$OVERWRITE_FLAG" = "true" ]; then
    CMD+=(--overwrite)
fi

if [ "$STRIP_FLAG" = "true" ]; then
    CMD+=(--strip_parquet_metadata)
fi

echo "Preparing LeRobot T5 cache"
echo "  repo_id=$REPO_ID"
echo "  dataset_root=$DATASET_ROOT"
echo "  model_weights_root=$MODEL_WEIGHTS_ROOT"
echo "  device=$DEVICE"
echo "  t5_folder_name=$T5_FOLDER_NAME"
echo "  max_episodes=$MAX_EPISODES"

"${CMD[@]}"