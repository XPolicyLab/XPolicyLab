#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${1:-${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}}"
KINRT_REPO="${KINRT_SOURCE_REPO:-https://github.com/gleeacast/KinRT.git}"
KINRT_REV="${KINRT_SOURCE_REV:-108368422539b79d2be2c596750de1fab1cfd8fd}"
KINRT_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo"

if [[ ! -f "${OPENPI_ROOT}/pyproject.toml" && "${OPENPI_ROOT}" == "${DEFAULT_OPENPI_ROOT}" ]]; then
  if [[ -e "${KINRT_ROOT}" ]]; then
    echo "[KinRT][ERROR] Expected an empty clone target: ${KINRT_ROOT}" >&2
    exit 1
  fi
  git clone "${KINRT_REPO}" "${KINRT_ROOT}"
  git -C "${KINRT_ROOT}" checkout --detach "${KINRT_REV}"
fi

if [[ ! -f "${OPENPI_ROOT}/pyproject.toml" ]]; then
  echo "[KinRT][ERROR] OpenPI project not found: ${OPENPI_ROOT}" >&2
  exit 1
fi

SOURCE_ROOT="$(cd "${OPENPI_ROOT}/../.." && pwd)"
if git -C "${SOURCE_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  actual_rev="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
  if [[ "${actual_rev}" != "${KINRT_REV}" && "${KINRT_ALLOW_UNPINNED_SOURCE:-0}" != "1" ]]; then
    echo "[KinRT][ERROR] Source revision mismatch: ${actual_rev}" >&2
    echo "[KinRT][ERROR] Expected: ${KINRT_REV}" >&2
    echo "[KinRT][ERROR] Set KINRT_ALLOW_UNPINNED_SOURCE=1 only for development." >&2
    exit 1
  fi
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" ]]; then
  echo "[KinRT][ERROR] uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

cd "${OPENPI_ROOT}"
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" sync --group lerobot
UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 "${UV_BIN}" pip install -e .
"${UV_BIN}" pip uninstall opencv-python || true
"${UV_BIN}" pip install opencv-python-headless==4.11.0.86 scikit-learn joblib
"${UV_BIN}" pip install -e "${XPL_ROOT}"
"${UV_BIN}" run python -c "import XPolicyLab, openpi; print('KinRT policy environment is ready')"
