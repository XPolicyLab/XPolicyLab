#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [local-checkpoint-or-https-url]" >&2
  echo "With no argument, download robodojo.pt from InternRobotics/InternW0-Delta-RoboDojo." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/checkpoints"
DEST="${DEST_DIR}/robodojo.pt"
EXPECTED_SHA="344d4221d628462fb8d1f7263fdfeab74ed4701e7e0af25dde4a693b22968aaa"
SOURCE="${1:-}"
mkdir -p "${DEST_DIR}"

if [[ -z "${SOURCE}" ]]; then
  python - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="InternRobotics/InternW0-Delta-RoboDojo",
    filename="robodojo.pt",
    local_dir="${DEST_DIR}",
)
PY
elif [[ "${SOURCE}" == https://* ]]; then
  command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
  curl --fail --location --continue-at - --output "${DEST}" "${SOURCE}"
else
  source_path="$(cd "$(dirname "${SOURCE}")" && pwd)/$(basename "${SOURCE}")"
  [[ -f "${source_path}" ]] || { echo "Checkpoint not found: ${source_path}" >&2; exit 1; }
  if [[ "${source_path}" != "${DEST}" ]]; then
    cp --reflink=auto "${source_path}" "${DEST}"
  fi
fi

actual_sha=$(sha256sum "${DEST}" | awk '{print $1}')
if [[ "${actual_sha}" != "${EXPECTED_SHA}" ]]; then
  echo "Checkpoint SHA256 mismatch: ${actual_sha}" >&2
  exit 1
fi
echo "Checkpoint ready: ${DEST}"
