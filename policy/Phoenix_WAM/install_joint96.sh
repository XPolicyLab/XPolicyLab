#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${1:-phoenix-wam}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FFMPEG_BIN="${PHOENIX_FFMPEG:-ffmpeg}"
FFMPEG_FILTERS="$("${FFMPEG_BIN}" -hide_banner -filters 2>&1)"
for required_filter in scale_cuda hwupload_cuda; do
    if ! grep -qw "${required_filter}" <<< "${FFMPEG_FILTERS}"; then
        echo "FFmpeg must provide ${required_filter}; install a CUDA-enabled build and set PHOENIX_FFMPEG." >&2
        exit 1
    fi
done
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "${SCRIPT_DIR}/requirements-joint96.txt"
python -m pip install -e "${XPL_ROOT}"
echo "Installed ${ENV_NAME}. Run download_joint96_weights.sh; source/config/tokenizer are checked into joint96_source/."
