#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_ROOT="${1:-${SCRIPT_DIR}/assets}"

python -c "import modelscope" 2>/dev/null || {
  echo "Python package 'modelscope' not found. Activate the policy environment first." >&2
  exit 1
}

download_model() {
  local model_id="$1"
  local destination="${ASSET_ROOT}/${model_id}"
  local selection="$2"
  mkdir -p "${destination}"
  echo "Downloading ${model_id} to ${destination}"
  python - "${model_id}" "${destination}" "${selection}" <<'PY'
import sys
from modelscope import snapshot_download

patterns = None
if sys.argv[3] == "wan-eval":
    patterns = [
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "google/umt5-xxl/*",
    ]
kwargs = {"local_dir": sys.argv[2]}
if patterns:
    kwargs["allow_file_pattern"] = patterns
snapshot_download(sys.argv[1], **kwargs)
PY
}

download_model "Wan-AI/Wan2.2-TI2V-5B" "wan-eval"
download_model "Alibaba-DAMO-Academy/RynnBrain1.1-2B" "full"

echo "Model assets are ready under ${ASSET_ROOT}"
