#!/usr/bin/env bash
set -euo pipefail
case "${1:-prepared}" in
    prepared) ;;
    *) echo "Usage: download_weights.sh [prepared]" >&2; exit 2 ;;
esac
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate awomo05
WEIGHTS_DIR=${AWOMO05_WEIGHTS:-$(cd "$(dirname "$0")" && pwd)/weights}
mkdir -p "$WEIGHTS_DIR"
export AWOMO05_PREPARED_MODEL_REPO=${AWOMO05_PREPARED_MODEL_REPO:-Auwomo/Awomo-0.5I-Robodojo}
export AWOMO05_PREPARED_MODEL_REVISION=${AWOMO05_PREPARED_MODEL_REVISION:-bb15096482e0e60a505e80953635d819802d9916}
python - "$WEIGHTS_DIR" <<'PY'
import hashlib
import os
from pathlib import Path
import re
import sys
from huggingface_hub import snapshot_download

revision = os.environ["AWOMO05_PREPARED_MODEL_REVISION"]
if not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise ValueError("A full immutable model revision is required")
directory = Path(sys.argv[1])
snapshot_download(os.environ["AWOMO05_PREPARED_MODEL_REPO"], revision=revision,
    allow_patterns=["model.pt", "awomo_config.json", "dataset_stats.json", "SHA256SUMS"], local_dir=directory)
expected_names = {"model.pt", "awomo_config.json", "dataset_stats.json"}
seen = set()
for line in (directory / "SHA256SUMS").read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    if name not in expected_names or name in seen or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Invalid prepared checksum manifest")
    seen.add(name)
    digest = hashlib.sha256()
    with (directory / name).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {name}")
if seen != expected_names:
    raise ValueError("Incomplete prepared checkpoint")
snapshot_download("Qwen/Qwen3-VL-4B-Instruct",
    local_dir=directory / "Qwen3-VL-4B-Instruct")
snapshot_download("black-forest-labs/FLUX.2-klein-base-4B", allow_patterns=["vae/*"],
    local_dir=directory / "FLUX.2-klein-base-4B")
print(f"Downloaded SFT checkpoint, Qwen3-VL and VAE: {directory}")
PY
