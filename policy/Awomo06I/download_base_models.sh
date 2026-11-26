#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR=${1:-"${SCRIPT_DIR}/base_models"}
mkdir -p "${TARGET_DIR}/flux2" "${TARGET_DIR}/qwen3-vl"
hf download black-forest-labs/FLUX.2-klein-base-4B --local-dir "${TARGET_DIR}/flux2"
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "${TARGET_DIR}/qwen3-vl"
echo "export AWOMO06I_FLUX2_BASE=${TARGET_DIR}/flux2"
echo "export AWOMO06I_QWEN_VL=${TARGET_DIR}/qwen3-vl"
