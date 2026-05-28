#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STARVLA_ROOT="${SCRIPT_DIR}/source_starvla"

python -m pip install -r "${STARVLA_ROOT}/requirements.txt"
python -m pip install -e "${STARVLA_ROOT}"
python -m pip install -e "${ROOT_DIR}"
