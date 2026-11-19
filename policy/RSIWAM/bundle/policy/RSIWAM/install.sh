#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
: "${RSIWAM_XPL_ROOT:?Set RSIWAM_XPL_ROOT to an installed XPolicyLab checkout}"
# The caller chooses the policy environment and installs a matching CUDA torch
# build first. The launcher sets CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA.
# This package does not install or modify the simulator environment.
python -m pip install -r "$PACKAGE_ROOT/requirements.txt" -c "$PACKAGE_ROOT/constraints-reference.txt"
python -m pip install -e "$RSIWAM_XPL_ROOT"
printf '%s\n' 'Dependencies installed. Review constraints-reference.txt; this is not a fully locked environment.'
