#!/bin/bash
set -euo pipefail

# Download a released EWAM checkpoint plus the backbone files needed
# at inference time into policy/EWAM/checkpoints/.
#
#   bash download_checkpoints.sh [robotwin-c2r|robotwin-indomain|robodojo]   (default: robotwin-c2r)
#
# Honors HF_ENDPOINT (e.g. https://hf-mirror.com) and HF_TOKEN.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${EWAM_CKPT_DOWNLOAD_DIR:-${SCRIPT_DIR}/checkpoints}"
setting="${1:-robotwin-c2r}"

case "${setting}" in
    robotwin-c2r|robotwin-indomain|robodojo) ;;
    *) echo "unknown setting: ${setting} (expected robotwin-c2r, robotwin-indomain or robodojo)" >&2; exit 1 ;;
esac

mkdir -p "${CKPT_DIR}"

python - "${CKPT_DIR}" "${setting}" <<'PY'
import sys
from huggingface_hub import snapshot_download

ckpt_dir, setting = sys.argv[1], sys.argv[2]

# EWAM weights (the full model: video, VLM, and action experts): checkpoints/<setting>/mp_rank_00_model_states.pt
snapshot_download("HaoWang00/EWAM", local_dir=ckpt_dir, allow_patterns=[f"{setting}/*"])

# Wan2.2: config + VAE + T5 encoder/tokenizer only (DiT weights come from the EWAM checkpoint).
snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B",
    local_dir=f"{ckpt_dir}/Wan2.2-TI2V-5B",
    allow_patterns=["config.json", "Wan2.2_VAE.pth", "models_t5_umt5-xxl-enc-bf16.pth", "google/*"],
)

# Qwen3-VL: processor + config only (VLM weights come from the EWAM checkpoint).
snapshot_download(
    "Qwen/Qwen3-VL-2B-Instruct",
    local_dir=f"{ckpt_dir}/Qwen3-VL-2B-Instruct",
    ignore_patterns=["*.safetensors", "*.bin"],
)
PY

echo "[EWAM] checkpoint: ${CKPT_DIR}/${setting}/mp_rank_00_model_states.pt"
