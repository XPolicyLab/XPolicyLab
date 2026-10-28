#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_ENDPOINT=https://huggingface.co

# One pinned revision holds the evaluated checkpoint, its frozen config, and unmodified mirrors of the
# Cosmos3-Nano subset and Wan2.2 VAE it loads (with their licenses and notices), in the layout below.
python - "${SCRIPT_DIR}" <<'PY'
import sys

from huggingface_hub import hf_hub_download, snapshot_download

REPO, REVISION = "yuanty/G07-preview-RoboDojo", "76e2aca133372464d61fefb8127f003fffd99adc"
# Fetch the small config first, so download problems surface before the large files.
hf_hub_download(REPO, "checkpoints/robodojo/inference_config.yaml", revision=REVISION, local_dir=sys.argv[1])
snapshot_download(REPO, revision=REVISION, allow_patterns=["checkpoints/*"], local_dir=sys.argv[1])
PY

cd "${SCRIPT_DIR}/checkpoints"
sha256sum --check <<'SUMS'
b6201aeef74307948116555ef04ef1951105c9c22fab715b5fb67e80ef00ed6a  robodojo/checkpoints/weights/step_020904.pt
0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996  DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
SUMS

echo 'Model assets ready. Use ckpt_name=robodojo.'
