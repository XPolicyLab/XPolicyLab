#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${GEOREFINER_DOWNLOAD_DIR:-${POLICY_DIR}/checkpoints/georefiner}"
HF_REPO_ID="${GEOREFINER_HF_REPO_ID:-Hazel500am/X-VLA-GeoRefiner-RoboDojo}"
HF_FILENAME="${GEOREFINER_HF_FILENAME:-georefiner_xvla_trained_fp32.pt}"
HF_REVISION="${GEOREFINER_HF_REVISION:-44870539ffabd538a10849809c5bc34e3c93bbff}"
LOCAL_FILENAME="georefiner_xvla_trained_fp32.pt"
XVLA_RUN_NAME="${XVLA_RUN_NAME:-RoboDojo-sim-arx_x5-ee-0}"
XVLA_CHECKPOINT_NAME="${XVLA_CHECKPOINT_NAME:-ckpt-100000}"
XVLA_HF_DIR="${XVLA_HF_DIR:-xvla/${XVLA_RUN_NAME}/${XVLA_CHECKPOINT_NAME}}"
XVLA_TARGET_DIR="${XVLA_DOWNLOAD_DIR:-${POLICY_DIR}/../X_VLA/checkpoints/${XVLA_RUN_NAME}/${XVLA_CHECKPOINT_NAME}}"

command -v hf >/dev/null 2>&1 || {
    echo "hf is required (install the huggingface_hub package)." >&2
    exit 1
}

mkdir -p "${POLICY_DIR}/checkpoints"
staging_dir="$(mktemp -d "${POLICY_DIR}/checkpoints/.hf-download.XXXXXX")"
cleanup() {
    rm -rf -- "${staging_dir}"
}
trap cleanup EXIT

if [[ ! -f "${XVLA_TARGET_DIR}/model.safetensors" || \
      ! -f "${XVLA_TARGET_DIR}/preprocessor_config.json" || \
      ! -f "${XVLA_TARGET_DIR}/tokenizer.json" ]]; then
    hf download \
        "${HF_REPO_ID}" \
        --include "${XVLA_HF_DIR}/*" \
        --revision "${HF_REVISION}" \
        --local-dir "${staging_dir}"

    downloaded_xvla_dir="${staging_dir}/${XVLA_HF_DIR}"
    if [[ ! -f "${downloaded_xvla_dir}/model.safetensors" ]]; then
        echo "Downloaded X-VLA checkpoint not found: ${downloaded_xvla_dir}" >&2
        exit 1
    fi

    mkdir -p "${XVLA_TARGET_DIR}"
    for downloaded_file in "${downloaded_xvla_dir}"/*; do
        mv -f -- "${downloaded_file}" "${XVLA_TARGET_DIR}/"
    done
fi

mkdir -p "${TARGET_DIR}"
hf download \
    "${HF_REPO_ID}" \
    --include "${HF_FILENAME}" \
    --include "dinov2_processor/*" \
    --include "clip_tokenizer/*" \
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

echo "[GeoRefiner] X-VLA checkpoint ready in ${XVLA_TARGET_DIR}"
echo "[GeoRefiner] Assets ready in ${TARGET_DIR}"
