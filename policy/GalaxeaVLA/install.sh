#!/bin/bash
# GalaxeaVLA (galaxea_fm) environment setup for XPolicyLab.
#
# Builds an isolated uv virtualenv inside the upstream project dir and installs
# galaxea_fm in editable mode. It does NOT download multi-GB weights/backbones;
# those steps are printed for you to run manually (see INSTALLATION.md).
#
# Usage:  bash install.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${SCRIPT_DIR}/GalaxeaVLA"

if ! command -v uv >/dev/null 2>&1; then
    echo "[install] 'uv' not found. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

cd "${UPSTREAM_DIR}"

# Optional China mirrors (matches upstream README); comment out if not needed.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

echo "[install] uv sync (python 3.10, torch 2.7.1 cu128) ..."
uv sync --index-strategy unsafe-best-match

echo "[install] uv pip install -e . (+ dev) ..."
source .venv/bin/activate
uv pip install -e .
uv pip install -e .[dev]

echo
echo "[install] core env ready. Remaining MANUAL steps (NOT run here):"
cat <<'EOF'
  1) System ffmpeg (for av/mp4 dataset encoding):
       sudo apt install -y ffmpeg

  2) PaliGemma-3B backbone (required by g0plus tokenizer + vision tower):
       hf download google/paligemma-3b-pt-224 \
         --local-dir weights/paligemma-3b-pt-224
     (G0Tiny uses HuggingFaceTB/SmolVLM2-500M-Video-Instruct instead.)

  3) G0Plus_3B_base checkpoint (default deploy weights):
       hf download OpenGalaxea/G0-VLA --include "G0Plus_3B_base/*" \
         --local-dir checkpoints

  4) Point deploy.yml / GALAXEA_PALIGEMMA_PATH at the backbone dir from step 2.
EOF
echo "[install] done."
