#!/bin/bash
# Downloads the weights of the services this policy uses into checkpoints/
# (ModelScope and hf-mirror, sha256-checked): sam3 wall-oss.
#   bash download_checkpoints.sh [python]
set -euo pipefail
PY=${1:-python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
"$PY" the-harness/scripts/harness fetch sam3 wall-oss --deploy "$HERE/deploy.toml"
