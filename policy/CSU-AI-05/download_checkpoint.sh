#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PUBLIC_CHECKPOINT_URL="${CSU_AI05_PUBLIC_CHECKPOINT_URL:-https://huggingface.co/datasets/ShaoRun/vla_ro/tree/main/checkpoints/CSU-AI-05}"
URL="${1:-${CSU_AI05_CHECKPOINT_URL:-${PUBLIC_CHECKPOINT_URL}}}"
DESTINATION="${2:-${CSU_AI05_CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/RoboDojo-CSU-AI-05-arx_x5-joint-0}}"
MANIFEST="${CSU_AI05_CHECKSUM_MANIFEST:-${SCRIPT_DIR}/checkpoint_sha256.txt}"
PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing checksum manifest: ${MANIFEST}" >&2
  exit 3
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set G05_PYTHON to a valid Python executable." >&2
  exit 3
fi
if [[ -e "${DESTINATION}" ]]; then
  echo "Destination already exists; refusing to overwrite: ${DESTINATION}" >&2
  exit 3
fi

parent="$(dirname "${DESTINATION}")"
mkdir -p "${parent}"
tmp_root="$(mktemp -d "${parent}/.csu-ai-05-download.XXXXXX")"
cleanup() {
  rm -rf -- "${tmp_root}"
}
trap cleanup EXIT
payload=""
hf_tree_regex='^https://huggingface\.co/datasets/([A-Za-z0-9._-]+/[A-Za-z0-9._-]+)/tree/([^/]+)/(.+)$'

if [[ "${URL}" =~ ${hf_tree_regex} ]]; then
  repo_id="${BASH_REMATCH[1]}"
  revision="${BASH_REMATCH[2]}"
  subfolder="${BASH_REMATCH[3]%/}"
  hf_root="${tmp_root}/huggingface"
  mkdir -p "${hf_root}"

  "${PYTHON_BIN}" - "${repo_id}" "${revision}" "${subfolder}" "${hf_root}" <<'PY'
from pathlib import Path
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise RuntimeError(
        "Hugging Face folder downloads require `huggingface_hub` in G05_PYTHON. "
        "Install it with: python -m pip install huggingface_hub"
    ) from exc

repo_id, revision, subfolder, local_dir = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision=revision,
    allow_patterns=[f"{subfolder}/*", f"{subfolder}/**"],
    local_dir=Path(local_dir),
)
PY
  payload="${hf_root}/${subfolder}"
  if [[ ! -d "${payload}" ]]; then
    echo "Hugging Face download did not create the expected folder: ${payload}" >&2
    exit 4
  fi
else
  archive="${tmp_root}/checkpoint.archive"
  extract_root="${tmp_root}/extracted"
  mkdir -p "${extract_root}"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --silent --show-error --output "${archive}" "${URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget --tries=3 --output-document="${archive}" "${URL}"
  else
    echo "Install curl or wget to download the checkpoint archive." >&2
    exit 4
  fi

  "${PYTHON_BIN}" - "${archive}" "${extract_root}" <<'PY'
from pathlib import Path
import os
import sys
import tarfile
import zipfile

archive = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()

def safe_target(name: str) -> Path:
    target = (root / name).resolve()
    if os.path.commonpath((str(root), str(target))) != str(root):
        raise RuntimeError(f"Archive member escapes destination: {name}")
    return target

if zipfile.is_zipfile(archive):
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            safe_target(member.filename)
        handle.extractall(root)
elif tarfile.is_tarfile(archive):
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            safe_target(member.name)
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
        handle.extractall(root, members=members)
else:
    raise RuntimeError("Checkpoint download is neither a zip nor a tar archive")
PY

  payload="${extract_root}"
  if [[ ! -f "${payload}/action_tokenizer.pt" ]]; then
    mapfile -t top_level_dirs < <(find "${extract_root}" -mindepth 1 -maxdepth 1 -type d)
    if [[ "${#top_level_dirs[@]}" -eq 1 && -f "${top_level_dirs[0]}/action_tokenizer.pt" ]]; then
      payload="${top_level_dirs[0]}"
    fi
  fi
fi

(
  cd "${payload}"
  sha256sum --check "${MANIFEST}"
)

mv -- "${payload}" "${DESTINATION}"
trap - EXIT
rm -rf -- "${tmp_root}"

cat <<EOF
CSU-AI-05 bundle downloaded and verified:
  ${DESTINATION}

Before evaluation:
  export G05_CKPT_PATH=${DESTINATION}
  export ROBODOJO_G05_ACTION_SOURCE=fm
EOF
