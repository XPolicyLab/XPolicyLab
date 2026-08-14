#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
G05_REAL_ROOT="${G05_REAL_ROOT:-${SCRIPT_DIR}/runtime/G05}"
TOKENIZER_ROOT="${G05_REAL_TOKENIZER_ROOT:-${SCRIPT_DIR}/runtime/galaxea_tokenizer}"
DATASET_ROOT="${G05_REAL_DATASET_ROOT:-${SCRIPT_DIR}/runtime/galaxea_dataset}"
PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Set G05_PYTHON to a Python executable" >&2; exit 2; }
[[ -f "${G05_REAL_ROOT}/pyproject.toml" ]] || { echo "G0.5 real runtime not found: ${G05_REAL_ROOT}" >&2; exit 2; }
[[ -f "${TOKENIZER_ROOT}/pyproject.toml" ]] || { echo "G0.5 real tokenizer runtime not found: ${TOKENIZER_ROOT}" >&2; exit 2; }
[[ -f "${DATASET_ROOT}/pyproject.toml" ]] || { echo "G0.5 real dataset runtime not found: ${DATASET_ROOT}" >&2; exit 2; }

"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"
"${PYTHON_BIN}" -m pip install -e "${DATASET_ROOT}"
"${PYTHON_BIN}" -m pip install -e "${TOKENIZER_ROOT}"
"${PYTHON_BIN}" -m pip install -e "${G05_REAL_ROOT}"
