#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: bash install.sh /path/to/eval/python" >&2
  exit 1
fi
"$1" -m pip install "openai==3.8.0" "pillow==12.3.0"
echo "Use an authenticated Codex executable on the continuing agent host; no provider credential is included in artifacts."
