#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ELAVA_ROOT="${AHA_WAM_ELAVA_ROOT:-/mnt/petrelfs/caijisong/linglong/project/fastwam/elava-prior-only/elava}"

pip install -e "${ROOT_DIR}/XPolicyLab"
pip install -e "${ELAVA_ROOT}"
pip install pyyaml opencv-python
