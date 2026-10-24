#!/bin/bash
# Installs Harness Policy for this XPolicyLab policy into the policy environment.
#   bash install.sh [python of the policy environment]          (default: python)
# Tested with the harness-policy conda environment (the-harness/environment/setup.sh),
# which also runs RoboDojo; any environment with RoboDojo's (torch 2.7, cuRobo) and
# XPolicyLab's packages works after this script.
set -euo pipefail
PY=${1:-python}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -d "$HERE/the-harness" ]; then
  git clone https://github.com/EvolvingLMMs-Lab/the-harness.git "$HERE/the-harness"
  git -C "$HERE/the-harness" checkout 32c267a7224576126a9e273f9b810ac6e0396a32
fi
"$PY" -m pip install -e "$XPL_ROOT"
"$PY" -m pip install --constraint "$HERE/the-harness/environment/constraints.txt" \
    -r "$HERE/the-harness/environment/requirements-skills.txt"
"$PY" -c "import sys; sys.path[:0] = ['$HERE/the-harness/src', '$HERE/the-harness']; import harness_policy.connectors.package"
echo "Harness Policy installed for $PY"
