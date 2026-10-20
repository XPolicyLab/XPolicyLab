#!/bin/bash
# Download Qwen3-VL-4B-Instruct, the base VLM for Spark-0.
# Destination matches framework.qwenvl.base_vlm:
#   ./playground/Pretrained_models/Qwen3-VL-4B-Instruct  (relative to source_starvla/)
#
# Usage:
#   bash scripts/download_qwen3_vl_4b_instruct.sh
#   OUT_DIR=/abs/path/Qwen3-VL-4B-Instruct bash scripts/download_qwen3_vl_4b_instruct.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_qwen3_vl_4b_instruct.sh   # CN mirror
#
# Requires: huggingface_hub (`pip install -U huggingface_hub`) with the
# `hf` CLI (preferred) or a Python import fallback. Auth only needed if the
# repo is gated for your account (Qwen3-VL-4B-Instruct is public as of writing).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(dirname "${SCRIPT_DIR}")"
STARVLA_ROOT="${ADAPTER_DIR}/source_starvla"

REPO_ID="${HF_REPO_ID:-Qwen/Qwen3-VL-4B-Instruct}"
OUT_DIR="${OUT_DIR:-${STARVLA_ROOT}/playground/Pretrained_models/Qwen3-VL-4B-Instruct}"

mkdir -p "$(dirname "${OUT_DIR}")"

echo "[download_qwen3_vl_4b_instruct] repo=${REPO_ID}"
echo "[download_qwen3_vl_4b_instruct] out=${OUT_DIR}"
if [[ -n "${HF_ENDPOINT:-}" ]]; then
    echo "[download_qwen3_vl_4b_instruct] HF_ENDPOINT=${HF_ENDPOINT}"
fi

if command -v hf >/dev/null 2>&1; then
    hf download "${REPO_ID}" --local-dir "${OUT_DIR}"
elif python -c "import huggingface_hub" >/dev/null 2>&1; then
    python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${REPO_ID}", local_dir="${OUT_DIR}")
PY
else
    echo "[download_qwen3_vl_4b_instruct] ERROR: need hf CLI or python package huggingface_hub" >&2
    echo "[download_qwen3_vl_4b_instruct]   pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
fi

if [[ ! -f "${OUT_DIR}/config.json" ]]; then
    echo "[download_qwen3_vl_4b_instruct] WARNING: ${OUT_DIR}/config.json missing; check the download." >&2
fi

echo "[download_qwen3_vl_4b_instruct] done -> ${OUT_DIR}"
echo "[download_qwen3_vl_4b_instruct] point framework.qwenvl.base_vlm at this dir"
