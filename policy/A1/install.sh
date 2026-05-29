#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A1_DIR="${SCRIPT_DIR}/A1"
policy_conda_env="${1:-a1}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

echo -e "\033[33m[A1 Install] Installing A1 package...\033[0m"

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${policy_conda_env}"; then
    conda create -y -n "${policy_conda_env}" python=3.10
fi
conda activate "${policy_conda_env}"

cd "${A1_DIR}"
# Keep A1 aligned with its upstream recommended runtime.
python -m pip install -U setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-build-isolation -e .[all]
python -m pip install "transformers>=4.37.1,<5"
python -m pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
python -m pip install -r requirements.txt

echo -e "\033[33m[A1 Install] Installing XPolicyLab package...\033[0m"
cd "${SCRIPT_DIR}/../.."
python -m pip install --no-build-isolation -e .

echo -e "\033[33m[A1 Install] Installation complete.\033[0m"
