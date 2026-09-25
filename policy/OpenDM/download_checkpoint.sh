#!/usr/bin/env bash
set -euo pipefail
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${POLICY_DIR}/../../..${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
if [[ $# -eq 0 ]]; then set -- policy; fi
exec python -m XPolicyLab.policy.OpenDM.scripts.download_assets "$@"
