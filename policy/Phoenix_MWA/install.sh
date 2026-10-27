#!/usr/bin/env bash
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:-phoenix-mwa}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$NAME" python=3.10 -y
conda activate "$NAME"
python -m pip install -e "$P/runtime[train,logging,modelscope]"
python -m pip install 'websockets>=15,<17' 'msgpack>=1,<2' msgpack-numpy opencv-python-headless h5py sentencepiece
# Install only the policy environment; use the official RoboTwin setup for the simulator.
