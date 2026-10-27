#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY_DIR="$root"
source "$POLICY_DIR/configs/runtime.conf"
python_bin="${PYTHON_BIN:-${POLICY_PYTHON:-python3}}"
mode="${1:-train}"
case "$mode" in
  infer) requirements="$root/go3_nano_preview/requirements-infer.txt" ;;
  train) requirements="$root/go3_nano_preview/requirements-train.txt" ;;
  *) echo 'Usage: install.sh [infer|train]; set PYTHON_BIN to the target Python 3.12 executable' >&2; exit 2 ;;
esac
"$python_bin" -m pip install -r "$requirements"
