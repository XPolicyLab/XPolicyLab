#!/bin/bash
set -e
set -o pipefail

usage() {
    cat <<'EOF'
Usage:
  bash train.sh <dataset_name> <ckpt_name> <env_cfg_type> <expert_data_num> <action_type> <seed> <gpu_id>

Optional environment overrides:
  LEROBOT_DATA_PATH   Use this LeRobot dataset directly.
  TASK_NAME           Optional fallback for local single-task HDF5 conversion.
  A1_TRAIN_CONFIG     Default: A1/train_config.local.yaml if present, else A1/train_config.yaml.
EOF
}

if [ "$#" -ne 7 ]; then
    usage >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${SCRIPT_DIR}/A1/xpolicylab_train.py" "$@"
