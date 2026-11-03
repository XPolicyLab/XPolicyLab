#!/usr/bin/env bash
# Online knowledge training: real practice -> review agent -> frozen knowledge.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$("${KNOWIN_AGENT_PYTHON:-python3}" "${SCRIPT_DIR}/_paths.py")"
export PYTHONPATH="${SCRIPT_DIR}:${PACKAGE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${KNOWIN_AGENT_PYTHON:-python3}" -m knowin_agent.training "$@"
