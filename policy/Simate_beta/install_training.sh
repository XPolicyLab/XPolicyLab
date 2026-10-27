#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="${SIMATE_TRAIN_ROOT:-${POLICY_DIR}/training_source}"
REVISION=8ae595cda0eea1595afdb240749c7b595872f5c0
PYTHON="${PYTHON:-python}"

if [[ "${1:-}" == "--help" ]]; then
    echo "Activate a training environment with CUDA PyTorch, then run: bash install_training.sh"
    echo "SIMATE_TRAIN_ROOT selects the checkout; PYTHON selects the interpreter."
    exit 0
fi
[[ $# == 0 ]] || { echo "Unexpected arguments." >&2; exit 2; }
"${PYTHON}" -c 'import torch; assert torch.version.cuda, "Install a CUDA build of PyTorch first."'
if [[ ! -e "${SOURCE}" ]]; then
    git clone https://github.com/SIPAILab/simate-beta.git "${SOURCE}"
    git -C "${SOURCE}" checkout --detach "${REVISION}"
fi
[[ "$(git -C "${SOURCE}" rev-parse HEAD)" == "${REVISION}" ]] || {
    echo "Training checkout must be at ${REVISION}; existing checkout was not changed." >&2
    exit 1
}
[[ -z "$(git -C "${SOURCE}" status --porcelain --untracked-files=no)" ]] || {
    echo "Training checkout has tracked modifications." >&2
    exit 1
}
"${PYTHON}" -m pip install -r "${SOURCE}/requirements.txt"
"${PYTHON}" -m pip install 'huggingface_hub>=0.34'
"${PYTHON}" -m pip install --no-deps --no-build-isolation -e "${SOURCE}"
"${PYTHON}" -c 'import sipai, pyarrow, omegaconf, safetensors; print("Training dependencies imported successfully.")'
