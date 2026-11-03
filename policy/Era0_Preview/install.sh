#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Dependencies are installed only when the user runs this script.
python3.11 -m venv "${POLICY_VENV:-${DIR}/.venv}"
PY="${POLICY_VENV:-${DIR}/.venv}/bin/python"
"${PY}" -m pip install --upgrade pip
"${PY}" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu129
"${PY}" -m pip install -r "${DIR}/requirements.txt"
"${PY}" -m pip install --no-build-isolation flash-attn==2.8.3 causal-conv1d==1.6.2.post1
