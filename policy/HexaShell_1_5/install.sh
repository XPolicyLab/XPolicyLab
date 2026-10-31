#!/bin/bash
# HexaShell-1.5 policy environment: Python 3.11 with requirements.txt (policy server), torch 2.7 and SAM3
# (segmentation service), and the SAM3 checkpoint. Settings come from config.env next to this file.
#   bash install.sh               # conda environment $POLICY_CONDA_ENV (default hexashell)
#   bash install.sh /path/to/venv # or a venv at this path (created with python3.11 when missing)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM3_COMMIT=c1f6b1e83967c912f675d8975d437ccdc41e17d8   # facebookresearch/sam3, the commit our runs used

if [[ ! -f "${SCRIPT_DIR}/config.env" ]]; then
    cp "${SCRIPT_DIR}/config.env.example" "${SCRIPT_DIR}/config.env"
    echo "[INSTALL] created config.env from config.env.example; fill it in and run install.sh again" >&2
    exit 1
fi
if grep -q "/path/to/" "${SCRIPT_DIR}/config.env"; then
    echo "[INSTALL] config.env still holds placeholder paths (/path/to/...); fill it in first" >&2
    exit 1
fi
set -a
source "${SCRIPT_DIR}/config.env"
set +a

venv=${1:-}
if [[ -n "${venv}" ]]; then
    [[ -x "${venv}/bin/python" ]] || python3.11 -m venv "${venv}"
    py="${venv}/bin/python"
else
    env_name=${POLICY_CONDA_ENV:-hexashell}
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "${env_name}"; then
        conda create -y -n "${env_name}" python=3.11
    fi
    conda activate "${env_name}"
    py="$(command -v python)"
fi
echo "[INSTALL] policy environment: ${py}"

"${py}" -m pip install -r "${SCRIPT_DIR}/requirements.txt"
"${py}" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
"${py}" -m pip install timm==1.0.30 "sam3 @ git+https://github.com/facebookresearch/sam3.git@${SAM3_COMMIT}"

# facebook/sam3 on Hugging Face is gated: accept its license there, then log in (hf auth login) or set HF_TOKEN
ckpt="${SAM3_CHECKPOINT_PATH}"
if [[ ! -f "${ckpt}" ]]; then
    mkdir -p "$(dirname "${ckpt}")"
    "${py}" - "${ckpt}" <<'EOF'
import shutil, sys
from huggingface_hub import hf_hub_download
shutil.copyfile(hf_hub_download("facebook/sam3", "sam3.pt"), sys.argv[1])
EOF
fi
echo "[INSTALL] SAM3 checkpoint: ${ckpt}"
echo "[INSTALL] done; pass $([[ -n "${venv}" ]] && echo "${venv}" || echo "${env_name}") as <policy_env> to eval.sh"
