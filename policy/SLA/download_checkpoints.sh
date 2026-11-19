#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
sla_python uv
if [[ $# -eq 0 ]]; then set -- robotwin; fi
exec "${PYTHON_BIN}" -m XPolicyLab.policy.SLA.download_checkpoints "$@"
