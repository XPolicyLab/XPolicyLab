#!/bin/bash
# Model assets for Auriga's perception service (the policy itself has no trained weights).
# Wraps Auriga's services/vggt_height/download_weights.sh, which verifies every file against the
# SHA-256 of the official upstream file (weights.sha256). Default sources need no token:
# VGGT-Omega and SAM 3.1 from ModelScope. Only these two are needed (8.1 GB): the service does not
# load SAM 2.1. These are byte for byte the weights of the v5 runs (sha256 compared with the production copies):
#   vggt_omega_1b_512.pt           facebook/VGGT-Omega @ 1041e80f          c02da418...b0667934
#   sam3.1/sam3.1_multiplex.pt     facebook/sam3.1 @ daa63191              0567debe...b3541cb6
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/config_env.sh"
load_config_env "${SCRIPT_DIR}/config.env"

abs_path() {
    local value=$1
    if [[ "${value}" = /* ]]; then echo "${value}"; else echo "${SCRIPT_DIR}/${value#./}"; fi
}

auriga_src="$(abs_path "${AURIGA_SRC:-third_party/Auriga}")"
vggt_home="$(abs_path "${VGGT_HOME:-third_party/vggt_home}")"
mkdir -p "${vggt_home}"
VGGT_HOME="${vggt_home}" \
ONLY=vggt,sam31 \
VGGT_SOURCE="${VGGT_SOURCE:-modelscope}" \
SAM31_SOURCE="${SAM31_SOURCE:-modelscope}" \
    bash "${auriga_src}/services/vggt_height/download_weights.sh"

# ARX X5 robot meshes (RoboDojo assets, pinned Hugging Face revision, SHA-256 checked).
bash "${SCRIPT_DIR}/download_robot_meshes.sh"
