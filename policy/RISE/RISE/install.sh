#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OFFLINE_DIR="${ROOT_DIR}/policy_and_value/policy_offline_and_value"
DYNAMICS_DIR="${ROOT_DIR}/dynamics"

(
  cd "${OFFLINE_DIR}"

  # offline learning & value
  pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
  pip install --use-deprecated=legacy-resolver -e .
  pip install "git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
  pip install datasets==3.6.0
  pip install kornia
  cp -r ./src/openpi_value/models_pytorch/transformers_replace/* "$(python -c "import os; import transformers; print(os.path.dirname(transformers.__file__))")"

  # online learning
  pip install rlinf[embodied]

  # mini lerobot for data convertion
  cd mini_lerobot
  pip install -e .
)

(
  cd "${DYNAMICS_DIR}"

  # dynamics model
  pip install -e .
)

pip install torchcodec==0.2
