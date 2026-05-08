#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOWNLOAD_ROOT="${PROJECT_ROOT}/.hf_download_cache/robodojo_demo"

REPO_ID="${HF_REPO_ID:-DaMiTian/RoboDojo_demo_data}"
REMOTE_SUBDIR="${HF_REMOTE_SUBDIR:-archives/robodojo_tmp}"
HF_REVISION="${HF_REVISION:-main}"

TARGET_DATA_DIR="${PROJECT_ROOT}/data"
TARGET_ENV_CFG_DIR="${PROJECT_ROOT}/env_cfg"

echo "==> TEST root: ${PROJECT_ROOT}"
echo "==> Repo: ${REPO_ID}"
echo "==> Remote subdir: ${REMOTE_SUBDIR}"

mkdir -p "${DOWNLOAD_ROOT}"

if ! command -v python3 >/dev/null 2>&1; then
	echo "python3 未找到" >&2
	exit 1
fi

echo "==> Ensuring Python dependencies"
python3 - <<'PY'
import importlib.util
import subprocess
import sys

missing = [
		name for name in ("huggingface_hub", "hf_transfer")
		if importlib.util.find_spec(name) is None
]
if missing:
		subprocess.check_call([
				sys.executable,
				"-m",
				"pip",
				"install",
				"-U",
				"huggingface_hub",
				"hf_transfer",
		])
PY

echo "==> Downloading archive parts from Hugging Face"
REPO_ID="${REPO_ID}" REMOTE_SUBDIR="${REMOTE_SUBDIR}" HF_REVISION="${HF_REVISION}" DOWNLOAD_ROOT="${DOWNLOAD_ROOT}" python3 - <<'PY'
import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
remote_subdir = os.environ["REMOTE_SUBDIR"].strip("/")
revision = os.environ["HF_REVISION"]
download_root = Path(os.environ["DOWNLOAD_ROOT"])

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

patterns = [f"{remote_subdir}/*"] if remote_subdir else ["*"]

snapshot_download(
		repo_id=repo_id,
		repo_type="dataset",
		revision=revision,
		local_dir=str(download_root),
		local_dir_use_symlinks=False,
		resume_download=True,
		allow_patterns=patterns,
)

print(f"Downloaded to: {download_root}")
PY

ARCHIVE_DIR="${DOWNLOAD_ROOT}/${REMOTE_SUBDIR}"
if [[ ! -d "${ARCHIVE_DIR}" ]]; then
	echo "未找到下载目录: ${ARCHIVE_DIR}" >&2
	exit 1
fi

echo "==> Verifying downloaded files"
if [[ -f "${ARCHIVE_DIR}/SHA256SUMS" ]]; then
	(
		cd "${ARCHIVE_DIR}"
		sha256sum -c SHA256SUMS
	)
else
	echo "警告: 未发现 SHA256SUMS，跳过校验"
fi

shopt -s nullglob
parts=("${ARCHIVE_DIR}"/*.part-*)
shopt -u nullglob

if [[ ${#parts[@]} -eq 0 ]]; then
	echo "未找到分片文件: ${ARCHIVE_DIR}/*.part-*" >&2
	exit 1
fi

first_part="$(basename "${parts[0]}")"
archive_name="${first_part%.part-*}"
archive_path="${ARCHIVE_DIR}/${archive_name}"
extract_root="${DOWNLOAD_ROOT}/extracted"

echo "==> Reassembling archive: ${archive_name}"
cat "${ARCHIVE_DIR}"/*.part-* > "${archive_path}"

rm -rf "${extract_root}"
mkdir -p "${extract_root}"

echo "==> Extracting selected paths"
case "${archive_name}" in
	*.tar.zst)
		tar -I zstd -xf "${archive_path}" -C "${extract_root}" tmp/data tmp/env_cfg
		;;
	*.tar.gz)
		tar -xzf "${archive_path}" -C "${extract_root}" tmp/data tmp/env_cfg
		;;
	*.tar)
		tar -xf "${archive_path}" -C "${extract_root}" tmp/data tmp/env_cfg
		;;
	*)
		echo "不支持的压缩格式: ${archive_name}" >&2
		exit 1
		;;
esac

if [[ ! -d "${extract_root}/tmp/data" || ! -d "${extract_root}/tmp/env_cfg" ]]; then
	echo "压缩包内未找到 tmp/data 或 tmp/env_cfg" >&2
	exit 1
fi

echo "==> Restoring into ${PROJECT_ROOT}"
rm -rf "${TARGET_DATA_DIR}" "${TARGET_ENV_CFG_DIR}"
mv "${extract_root}/tmp/data" "${TARGET_DATA_DIR}"
mv "${extract_root}/tmp/env_cfg" "${TARGET_ENV_CFG_DIR}"

echo "==> Done"
du -sh "${TARGET_DATA_DIR}" "${TARGET_ENV_CFG_DIR}"
