#!/bin/bash
# Download the Mibot-RSI knowledge release named <ckpt_name> (one frozen snapshot per task, pinned by sha256 in
# release.json) from the dataset commit release.json pins into checkpoints/mibot-rsi-knowledge-<ckpt_name>/, and
# link checkpoints/RoboDojo-<ckpt_name>-arx_x5-joint-<seed>/ to it for each seed.
# The dataset is private: log in with `huggingface-cli login` or export HF_TOKEN. MIBOT_RSI_ARTIFACTS_DIR=<dir> reads a
# local copy of the dataset instead.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: bash download_checkpoint.sh <ckpt_name> [seeds, e.g. 1 or 0,1,2] [python]" >&2
    exit 1
fi
ckpt_name=$1
seeds=${2:-1}
python_bin=${3:-python3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${python_bin}" "${SCRIPT_DIR}/artifacts.py" knowledge --ckpt "${ckpt_name}" --seeds "${seeds}"
