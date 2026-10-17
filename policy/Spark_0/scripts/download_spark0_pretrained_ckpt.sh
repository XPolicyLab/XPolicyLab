#!/usr/bin/env bash
# Download the Spark-0 pretraining checkpoint from Hugging Face.
# The repository contains the checkpoint under a subdirectory; the directory
# layout is preserved below Spark_0/checkpoints/.
#
# Usage:
#   bash scripts/download_spark0_pretrained_ckpt.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_spark0_pretrained_ckpt.sh
#   OUT_DIR=./checkpoints bash scripts/download_spark0_pretrained_ckpt.sh
#
# Requires huggingface_hub (`hf` CLI preferred, or the Python package).
# Authenticate first with `hf auth login` only if the repository requires it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(dirname "${SCRIPT_DIR}")"

REPO_ID="${HF_REPO_ID:-TianxingChen/Spark-0}"
# Published run-dir name; its "wuji" tag predates the tianji_marvin_wuji
# default and is still resolved by setup_eval_policy_server.sh's tag glob.
CHECKPOINT_NAME="${CHECKPOINT_NAME:-Tianji-pretrain_spark_0-wuji-ee-0}"
CHECKPOINT_ROOT="${OUT_DIR:-${ADAPTER_DIR}/checkpoints}"
CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${CHECKPOINT_NAME}"

mkdir -p "${CHECKPOINT_ROOT}"

echo "[download_spark0_pretrained_ckpt] repo=${REPO_ID}"
echo "[download_spark0_pretrained_ckpt] subdirectory=${CHECKPOINT_NAME}"
echo "[download_spark0_pretrained_ckpt] out=${CHECKPOINT_DIR}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
    echo "[download_spark0_pretrained_ckpt] HF_ENDPOINT=${HF_ENDPOINT}"
fi

if command -v hf >/dev/null 2>&1; then
    hf download "${REPO_ID}" \
        --include "${CHECKPOINT_NAME}/**" \
        --local-dir "${CHECKPOINT_ROOT}"
elif command -v python >/dev/null 2>&1 && python -c "import huggingface_hub" >/dev/null 2>&1; then
    REPO_ID="${REPO_ID}" CHECKPOINT_NAME="${CHECKPOINT_NAME}" \
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["REPO_ID"],
    allow_patterns=[f'{os.environ["CHECKPOINT_NAME"]}/**'],
    local_dir=os.environ["CHECKPOINT_ROOT"],
)
PY
else
    echo "[download_spark0_pretrained_ckpt] ERROR: need hf CLI or python package huggingface_hub" >&2
    echo "[download_spark0_pretrained_ckpt]   install with: pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "[download_spark0_pretrained_ckpt] ERROR: checkpoint directory was not created: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

if ! find "${CHECKPOINT_DIR}" -type f -print -quit | grep -q .; then
    echo "[download_spark0_pretrained_ckpt] ERROR: checkpoint directory is empty: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

echo "[download_spark0_pretrained_ckpt] done -> ${CHECKPOINT_DIR}"
