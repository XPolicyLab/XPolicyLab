#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "${POLICY_DIR}/training.py" prepare "$@"
