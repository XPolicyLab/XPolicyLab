#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
PYTHON_BIN="${KINRT_PYTHON_BIN:-${OPENPI_ROOT}/.venv/bin/python}"
if [[ -z "${KINRT_PYTHON_BIN:-}" && ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi
destination="${POLICY_DIR}/checkpoints/KinRT-RoboDojo-Full35-60k"
if [[ $# -gt 0 && "$1" != --* ]]; then
  destination=$1
  shift
fi

"${PYTHON_BIN}" "${POLICY_DIR}/full35_assets.py" download \
  --destination "${destination}" \
  --repo-id "${KINRT_HF_REPO_ID:-Gleez/kinrt-robodojo-full35-a800-60k}" \
  --revision "${KINRT_HF_REVISION:-9460d07a9c7677ef3c72ece08df1f34eba7e45c7}" \
  "$@"
