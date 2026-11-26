#!/usr/bin/env bash
# Fetch the only weight this adapter needs for leaderboard/sim eval: SAM 3.
# Prefer ModelScope, then Hugging Face / hf-mirror. sha256-checked.
#   bash download_checkpoint.sh [python]
set -euo pipefail
PY=${1:-python3}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${HERE}/checkpoints/sam3"
OUT="${OUT_DIR}/sam3.pt"
# Identity recorded for the development artifact (ModelScope facebook/sam3).
EXPECT_SHA256="9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"

mkdir -p "${OUT_DIR}"
if [[ -f "${OUT}" ]]; then
  got="$("${PY}" - <<'PY' "${OUT}"
import hashlib, sys
h=hashlib.sha256()
with open(sys.argv[1],"rb") as f:
    for chunk in iter(lambda: f.read(1<<20), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"
  if [[ "${got}" == "${EXPECT_SHA256}" ]]; then
    echo "[download] sam3.pt already present and hash matches: ${OUT}"
    exit 0
  fi
  echo "[download] existing file hash mismatch (${got}); re-downloading" >&2
  rm -f "${OUT}"
fi

download_modelscope() {
  "${PY}" - <<'PY' "${OUT_DIR}"
import sys
from pathlib import Path
out_dir = Path(sys.argv[1])
try:
    from modelscope.hub.file_download import model_file_download
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"modelscope unavailable: {exc}")
path = model_file_download(model_id="facebook/sam3", file_path="sam3.pt", local_dir=str(out_dir))
print(path)
PY
}

download_hf() {
  local url="https://hf-mirror.com/facebook/sam3/resolve/main/sam3.pt"
  if command -v wget >/dev/null 2>&1; then
    wget -O "${OUT}.partial" "${url}"
  else
    curl -L --fail -o "${OUT}.partial" "${url}"
  fi
  mv "${OUT}.partial" "${OUT}"
}

echo "[download] fetching facebook/sam3 sam3.pt → ${OUT}"
if download_modelscope; then
  # modelscope may place under nested dir
  if [[ ! -f "${OUT}" ]]; then
    found=$(find "${OUT_DIR}" -name 'sam3.pt' | head -1 || true)
    if [[ -n "${found}" && "${found}" != "${OUT}" ]]; then
      mv "${found}" "${OUT}"
    fi
  fi
else
  echo "[download] ModelScope failed; trying hf-mirror" >&2
  download_hf
fi

got="$("${PY}" - <<'PY' "${OUT}"
import hashlib, sys
h=hashlib.sha256()
with open(sys.argv[1],"rb") as f:
    for chunk in iter(lambda: f.read(1<<20), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)"
if [[ "${got}" != "${EXPECT_SHA256}" ]]; then
  echo "[download][ERROR] sha256 mismatch: got ${got} expected ${EXPECT_SHA256}" >&2
  exit 1
fi
echo "[download] ok ${OUT}"
echo "[download] next: start the service, e.g."
echo "  HARNESS_SAM3_URL=http://127.0.0.1:8810 python -u scripts/serve_sam3.py --checkpoint ${OUT} --host 127.0.0.1 --port 8810"
