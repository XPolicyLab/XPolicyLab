#!/usr/bin/env bash
# Single-node launch: all GPUs on this host.
#
#   bash scripts/train_single_node.sh configs/train/mobile_smoke.yaml [overrides...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${MMABC_PYTHON:-$(command -v python)}
CONFIG=${1:?usage: train_single_node.sh <config> [overrides...]}
shift || true

# shellcheck source=/dev/null
source "$REPO/configs/cluster/nccl.env"
source "$REPO/scripts/cluster_common.sh"
load_wandb_env

NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

cd "$REPO"
exec "$PY" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC" \
  mmabc/train/entrypoint.py "$CONFIG" "$@"
