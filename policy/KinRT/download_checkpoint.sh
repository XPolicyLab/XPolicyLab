#!/usr/bin/env bash
set -euo pipefail

repo_id=${KINRT_HF_REPO_ID:-Gleez/KinRT-RoboDojo}
revision=${KINRT_HF_REVISION:-main}
destination=${1:-./checkpoints/KinRT-RoboDojo-10k}

python - "${repo_id}" "${revision}" "${destination}" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

repo_id, revision, destination = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision,
    local_dir=Path(destination),
)
PY

(
  cd "${destination}"
  sha256sum -c SHA256SUMS
)
