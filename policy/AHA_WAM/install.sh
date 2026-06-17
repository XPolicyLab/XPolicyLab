#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
AHA_WAM_PROJECT_ROOT="${AHA_WAM_PROJECT_ROOT:-${ROOT_DIR}/XPolicyLab/policy/AHA_WAM/AHAWAM}"

pip install -e "${ROOT_DIR}/XPolicyLab"
pip install -e "${AHA_WAM_PROJECT_ROOT}"
pip install pyyaml opencv-python
