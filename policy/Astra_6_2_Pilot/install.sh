#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
python -m pip install -e "$root" 'openai==3.57.0'
