#!/usr/bin/env bash
# Optional native reproduction only. This never calls qsub or the Miyabi pipeline.
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${POLICY_PYTHON:-python}" "$POLICY_DIR/workflow.py" train "$@"
