#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${POLICY_PYTHON:-python}" "$POLICY_DIR/workflow.py" convert "$@"
