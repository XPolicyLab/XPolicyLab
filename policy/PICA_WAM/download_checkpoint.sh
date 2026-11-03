#!/usr/bin/env bash
set -euo pipefail

# Usage: bash download_checkpoint.sh [local-checkpoint-or-https-url]
# With no source, downloads internw0_delta/robotwin.pt from the Hugging Face
# repo xdzouyd/PICA-WAM into checkpoints/robotwin.pt; the SHA256 is checked.

[[ $# -le 1 ]] || { echo "Usage: $0 [local-checkpoint-or-https-url]" >&2; exit 2; }

REPO_ID="xdzouyd/PICA-WAM"
REPO_FILE="internw0_delta/robotwin.pt"
EXPECTED_SHA="0884098dd352123c72556271ffd1660fbf416ad0cca35a483b6cdc46f73b0016"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${SCRIPT_DIR}/checkpoints"
DEST="${DEST_DIR}/robotwin.pt"
SOURCE="${1:-}"
mkdir -p "${DEST_DIR}"

if [[ -z "${SOURCE}" ]]; then
  python - "${REPO_ID}" "${REPO_FILE}" "${DEST_DIR}" <<'PY'
import sys
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], local_dir=sys.argv[3])
PY
  mv -f "${DEST_DIR}/${REPO_FILE}" "${DEST}"
  rmdir "${DEST_DIR}/$(dirname "${REPO_FILE}")" 2>/dev/null || true
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
