#!/usr/bin/env bash
# Download the evaluated checkpoint into ./checkpoints/RoboDojo-sim-arx_x5-joint-0
# (params/ + assets/RoboDojo_lerobot_v21_merged/norm_stats.json, ~12 GB).
#
set -euo pipefail
CKPT_REPO="${CKPT_REPO:-ByteMelodist/hachimi-robodojo-ckpt}"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${POLICY_DIR}/checkpoints"
mkdir -p "${DEST}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  pip install -U "huggingface_hub[cli]"
fi
huggingface-cli download "${CKPT_REPO}" --repo-type model --local-dir "${DEST}"

test -d "${DEST}/RoboDojo-sim-arx_x5-joint-0/params" || {
  echo "[Hachimi][ERROR] expected ${DEST}/RoboDojo-sim-arx_x5-joint-0/{params,assets}" >&2
  exit 1
}
echo "[Hachimi] checkpoint ready at ${DEST}/RoboDojo-sim-arx_x5-joint-0"
