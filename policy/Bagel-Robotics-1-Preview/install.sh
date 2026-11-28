#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_DIR="${1:-${SCRIPT_DIR}/.venv}"
PYTHON="${PYTHON_BIN:-python3}"

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${PYTHON}" -m venv "${ENV_DIR}"
fi
PIP="${ENV_DIR}/bin/python"
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  "${PIP}" -m pip install torch torchvision --index-url "${TORCH_INDEX_URL}"
fi
"${PIP}" -m pip install -e "${SCRIPT_DIR}/runtime" -e "${XPL_ROOT}"
"${PIP}" -m pip check
"${PIP}" -c 'import torch, vpp2.policy, XPolicyLab; print("Bagel-Robotics-1-Preview runtime ready; CUDA:", torch.cuda.is_available())'
echo "Policy environment: ${ENV_DIR}"
