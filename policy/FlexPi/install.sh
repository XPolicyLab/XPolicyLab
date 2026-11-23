#!/bin/bash
set -euo pipefail

# Mirrors flex-pi's docs/INSTALL.md §1a (conda) against the vendored tree, plus
# an editable XPolicyLab install so the policy server can import
# XPolicyLab.policy.FlexPi.model. The upstream docs say to run §1 exactly as
# written — the pins are load-bearing, so do not "simplify" the order here.
#
# Usage: bash install.sh [env_name] [--train] [--trt]
#   --train  adds cuda-nvcc (DeepSpeed shells out to `nvcc --version` at import)
#   --trt    adds the TensorRT extra (only needed to serve TRT engines)

ENV_NAME="flexpi"
WANT_TRAIN=false
WANT_TRT=false
for arg in "$@"; do
    case "${arg}" in
        --train) WANT_TRAIN=true ;;
        --trt)   WANT_TRT=true ;;
        -*)      echo "[ERROR] unknown flag: ${arg}" >&2; exit 1 ;;
        *)       ENV_NAME="${arg}" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FLEXPI_DIR="${SCRIPT_DIR}/FlexPi"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

pip install -U pip
pip install -e "${FLEXPI_DIR}" --extra-index-url https://download.pytorch.org/whl/cu128
# torchcodec comes from PyPI, not the cu128 index.
pip install --force-reinstall --no-deps \
    --index-url https://pypi.org/simple torchcodec==0.5
conda install -c conda-forge 'ffmpeg=7' -y

if [[ "${WANT_TRAIN}" == true ]]; then
    # Headers-only nvidia-cuda-nvcc-cu12 does NOT satisfy DeepSpeed's check.
    conda install -c nvidia 'cuda-nvcc=12.8' -y
fi
if [[ "${WANT_TRT}" == true ]]; then
    # Pinned: a TRT engine is tied to the TensorRT minor that baked it.
    pip install -e "${FLEXPI_DIR}[trt]"
fi

# Required: setup_policy_server.py imports XPolicyLab.policy.FlexPi.model, and
# model.py imports XPolicyLab.utils.process_data. XPolicyLab only asks for
# opencv-python-headless>=4.8, which resolves to a release that needs numpy 2
# and breaks flexpi's numpy==1.26.4 pin, so both stay pinned here.
pip install -e "${XPL_ROOT}" "numpy==1.26.4" "opencv-python-headless==4.8.1.78"

echo
echo "[OK] env '${ENV_NAME}' ready."
echo "Next: fetch the base weights (flex-pi docs/INSTALL.md §2) into"
echo "  ${FLEXPI_DIR}/checkpoints"
echo "or point DIFFSYNTH_MODEL_BASE_PATH at an existing weights dir."
