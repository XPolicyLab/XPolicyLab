#!/bin/bash
set -euo pipefail
PACKAGE=$(cd -- "$(dirname -- "$0")" && pwd)
RUNTIME=${1:?Pass the official RoboDojo runtime directory}
cd "$RUNTIME"
python XPolicyLab/setup_policy_server.py --config_path "$PACKAGE/deploy.yml"
