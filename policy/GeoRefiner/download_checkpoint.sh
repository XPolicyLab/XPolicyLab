#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${GEOREFINER_DOWNLOAD_DIR:-${POLICY_DIR}/checkpoints/georefiner}"
HF_REPO_ID="${GEOREFINER_HF_REPO_ID:-Hazel500am/X-VLA-GeoRefiner-RoboDojo}"
HF_FILENAME="${GEOREFINER_HF_FILENAME:-georefiner_xvla_trained_fp32.pt}"
HF_REVISION="${GEOREFINER_HF_REVISION:-44870539ffabd538a10849809c5bc34e3c93bbff}"
LOCAL_FILENAME="georefiner_xvla_trained_fp32.pt"

command -v hf >/dev/null 2>&1 || {
    echo "hf is required (install the huggingface_hub package)." >&2
    exit 1
}

mkdir -p "${TARGET_DIR}"
hf download \
    "${HF_REPO_ID}" \
    "${HF_FILENAME}" \
    "dinov2_processor/*" \
    "clip_tokenizer/*" \
    --revision "${HF_REVISION}" \
    --local-dir "${TARGET_DIR}"

downloaded_checkpoint="${TARGET_DIR}/${HF_FILENAME}"
runtime_checkpoint="${TARGET_DIR}/${LOCAL_FILENAME}"
if [[ ! -f "${downloaded_checkpoint}" ]]; then
    echo "Downloaded checkpoint not found: ${downloaded_checkpoint}" >&2
    exit 1
fi
if [[ "${downloaded_checkpoint}" != "${runtime_checkpoint}" ]]; then
    if [[ -e "${runtime_checkpoint}" || -L "${runtime_checkpoint}" ]]; then
        echo "Refusing to replace existing runtime checkpoint: ${runtime_checkpoint}" >&2
        exit 1
    fi
    ln -s "${HF_FILENAME}" "${runtime_checkpoint}"
fi

echo "[GeoRefiner] Assets ready in ${TARGET_DIR}"
