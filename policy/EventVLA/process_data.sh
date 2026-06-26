#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_REPO="KailunSu/niantian"
DATA_SUBDIR="RoboDojo_lerobot_v21_video"
DOWNLOAD_ROOT="${1:-${SCRIPT_DIR}/data}"
DOWNLOAD_PATH="${DOWNLOAD_ROOT}"

mkdir -p "${DOWNLOAD_PATH}"

if command -v huggingface-cli >/dev/null 2>&1; then
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
  bash process_data.sh [download_root]
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
