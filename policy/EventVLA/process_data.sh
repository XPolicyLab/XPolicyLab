#!/bin/bash
set -euo pipefail

# Standard XPolicyLab contract:
#   bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# EventVLA training data is prepared by the upstream EventVLA pipeline. This script
# downloads the pre-built LeRobot dataset and links it for local training/eval.
# Pass the standard 4 (+ optional expert_data_num) args for consistency. The
# optional episode cap is accepted for interface compatibility but is not applied.
# eval ckpt_name should match the upstream training output directory name.

bench_name=${1:-}
ckpt_name=${2:-}
env_cfg_type=${3:-}
action_type=${4:-}
expert_data_num=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${bench_name}" || -z "${ckpt_name}" || -z "${env_cfg_type}" || -z "${action_type}" ]]; then
  echo "Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]" >&2
  echo "  Data is fetched from the upstream EventVLA HuggingFace dataset." >&2
  exit 1
fi

if [[ -n "${expert_data_num}" ]]; then
  echo "[EventVLA] note: expert_data_num=${expert_data_num} is not applied; upstream dataset is used as a whole." >&2
fi

HF_REPO="KailunSu/niantian"
DATA_SUBDIR="RoboDojo_lerobot_v21_video"
DOWNLOAD_ROOT="${EVENTVLA_DATA_ROOT:-${SCRIPT_DIR}/data}"
DOWNLOAD_PATH="${DOWNLOAD_ROOT}"

mkdir -p "${DOWNLOAD_PATH}"

if command -v huggingface-cli >/dev/null 2>&1; then
    echo "[EventVLA] bench_name=${bench_name} ckpt_name=${ckpt_name} env_cfg_type=${env_cfg_type} action_type=${action_type}"
    echo "[EventVLA] Downloading ${DATA_SUBDIR} from ${HF_REPO} ..."
    huggingface-cli download "${HF_REPO}" \
        --repo-type dataset \
        --include "${DATA_SUBDIR}/*" \
        --local-dir "${DOWNLOAD_PATH}"
else
    cat >&2 <<'EOF'
[EventVLA] huggingface-cli is not installed.
Install it first:
  pip install -U "huggingface_hub[cli]"
Then rerun:
  bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
EOF
    exit 1
fi

if [[ -d "${DOWNLOAD_PATH}/${DATA_SUBDIR}" ]]; then
    TRAIN_DATA_DIR="${DOWNLOAD_PATH}/${DATA_SUBDIR}"
else
    TRAIN_DATA_DIR="${DOWNLOAD_PATH}"
fi

mkdir -p "${SCRIPT_DIR}/data"
ln -sfn "${TRAIN_DATA_DIR}" "${SCRIPT_DIR}/data/train_data"

echo "[EventVLA] Download completed."
echo "[EventVLA] Training data directory: ${TRAIN_DATA_DIR}"
echo "[EventVLA] Symlink updated: ${SCRIPT_DIR}/data/train_data -> ${TRAIN_DATA_DIR}"
echo "[EventVLA] eval ckpt_name should be the RUN_ID printed by train.sh."
