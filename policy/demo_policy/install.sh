#!/bin/bash
set -euo pipefail

# demo_policy has no separate package metadata; install XPolicyLab itself so
# imports such as `XPolicyLab.policy.demo_policy.model` and `client_server.ws`
# resolve in the policy environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python -m pip install -e "${XPL_ROOT}"