#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${G05_REAL_PYTHON:-$(command -v python3)}"
BUNDLE_ROOT="${G05_REAL_CHECKPOINT_PATH:-${SCRIPT_DIR}/checkpoints/arx_x5}"
RUNTIME_ROOT="${BUNDLE_ROOT}/inference_runtime"

[[ -x "${PYTHON_BIN}" ]] || { echo "set G05_REAL_PYTHON to a Python executable" >&2; exit 2; }
[[ -d "${RUNTIME_ROOT}/src/g05" ]] || {
  echo "missing inference runtime: ${RUNTIME_ROOT}" >&2
  echo "download and extract the robot-specific model bundle first" >&2
  exit 2
}
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"
"${PYTHON_BIN}" -m pip install -e "${RUNTIME_ROOT}/third_party/galaxea_dataset"
"${PYTHON_BIN}" -m pip install -e "${RUNTIME_ROOT}/third_party/galaxea_tokenizer"
"${PYTHON_BIN}" -m pip install -e "${RUNTIME_ROOT}"
