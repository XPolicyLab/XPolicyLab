#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
export PYTHONPATH="${WORKSPACE_ROOT}:${SCRIPT_DIR}/../RLinf${PYTHONPATH:+:${PYTHONPATH}}"
# Explicit pytest paths select a subset. With flags alone, run the whole suite.
for arg in "$@"; do
  if [[ -e "${arg%%::*}" ]]; then
    exec "${RL_PYTHON:-python}" -m pytest "$@"
  fi
done
exec "${RL_PYTHON:-python}" -m pytest "${SCRIPT_DIR}/../tests" "$@"
