#!/usr/bin/env bash
set -e

cd ..

TARGET_DIR="ctx_demo"

echo "==> Installing Python packages"
pip install -U huggingface_hub hf_transfer

echo "==> Start downloading dataset"
python - <<'PY'
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

snapshot_download(
    repo_id="Shh319/ctx_demo",
    repo_type="dataset",
    local_dir="ctx_demo",
    local_dir_use_symlinks=False,
    resume_download=True,
)

print("Download finished: ctx_demo")
PY

echo "==> Done"
du -sh "${TARGET_DIR}" || true