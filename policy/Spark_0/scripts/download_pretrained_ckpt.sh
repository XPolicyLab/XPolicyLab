#!/bin/bash
# Download the three Spark-0 pretrained assets by calling each script:
#   Qwen3-VL-4B-Instruct, DA3-LARGE, and the pretraining checkpoint.
#
# Usage:
#   bash scripts/download_pretrained_ckpt.sh
#   HF_ENDPOINT=https://hf-mirror.com bash scripts/download_pretrained_ckpt.sh
#
# OUT_DIR is per script (each has its own default). Set it only when running
# a single script; this wrapper leaves each destination at its default.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/download_qwen3_vl_4b_instruct.sh"
bash "${SCRIPT_DIR}/download_da3_large.sh"
bash "${SCRIPT_DIR}/download_spark0_pretrained_ckpt.sh"
