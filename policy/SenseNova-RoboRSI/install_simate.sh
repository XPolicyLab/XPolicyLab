#!/usr/bin/env bash
# Complete only the Simate backend; install.sh also prepares host/CLI/tools/ICL.
set -euo pipefail
unset CODEX_API_KEY OPENAI_API_KEY TOKENHUB_API_KEY
POLICY_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROBORSI_BOOTSTRAP_PYTHON:-python3}" "${POLICY_DIR}/simate_install.py" "$@"
