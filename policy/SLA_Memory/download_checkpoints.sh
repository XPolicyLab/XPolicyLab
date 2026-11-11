#!/bin/bash
# Usage: bash download_checkpoints.sh <training|robodojo_real>
#   training       SLA base weights, the initialization for train.sh, into checkpoints/SLA_base/
#   robodojo_real  the released real-robot checkpoint, weights and norm stats without the optimizer state,
#                  into checkpoints/SLA_Memory_robodojo_real_ckpt/
# Both downloads are pinned to fixed Hugging Face revisions.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_BIN="${POLICY_DIR}/openpi/.venv/bin/hf"

case "${1:-}" in
    training)
        "${HF_BIN}" download haofuly/SLA_base --revision 5bbdf3012ab643554aa2eb83ddeae9ace3dce476 \
            --local-dir "${POLICY_DIR}/checkpoints/SLA_base"
        ;;
    robodojo_real)
        "${HF_BIN}" download haofuly/SLA_benchmark_ckpts --revision 7f6880569c3be777291e646fad1b4951245e8553 \
            --include "SLA_Memory_robodojo_real_ckpt/*" --exclude "*/train_state/*" \
            --local-dir "${POLICY_DIR}/checkpoints"
        echo "[SLA_Memory] To evaluate it, set train_config_name: train_robodojo_real and extra_frames_first: false in deploy.yml"
        ;;
    *)
        echo "Usage: $0 <training|robodojo_real>" >&2
        exit 1
        ;;
esac
