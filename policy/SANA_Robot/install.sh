#!/bin/bash
set -euo pipefail

# Usage: bash install.sh [env_name]
# Creates the policy environment with the SANA repository's own recipe and installs XPolicyLab into it.
#   SANA_ROBOT_ROOT   an existing checkout of the SANA repository to use instead of cloning one
#   SANA_REPO         the repository to clone        (default: https://github.com/NVlabs/Sana.git)
#   SANA_REF          the branch or tag to check out (default: main)
ENV_NAME=${1:-sana-robot}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SANA_DIR="${SANA_ROBOT_ROOT:-${SCRIPT_DIR}/Sana}"

if [[ ! -d "${SANA_DIR}/sana_robot" ]]; then
    git clone --depth 1 --branch "${SANA_REF:-main}" "${SANA_REPO:-https://github.com/NVlabs/Sana.git}" "${SANA_DIR}"
fi

cd "${SANA_DIR}"
bash environment_setup.sh "${ENV_NAME}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python -m pip install -e "${XPL_ROOT}"

echo "[SANA_Robot] environment ${ENV_NAME} is ready; the SANA repository is at ${SANA_DIR}"
