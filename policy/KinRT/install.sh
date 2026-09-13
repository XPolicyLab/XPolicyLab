#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo/policy/pi05"
OPENPI_ROOT="${1:-${KINRT_OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}}"
KINRT_REPO="${KINRT_SOURCE_REPO:-https://github.com/gleeacast/KinRT.git}"
KINRT_REV="${KINRT_SOURCE_REV:-590d52802cde804cdc2d0ccb672c1a3a90d76f91}"
KINRT_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/KinRT_RoboDojo"
DEFAULT_LEROBOT_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)/LeRobot_KinRT_Full35"
LEROBOT_ROOT="${KINRT_LEROBOT_ROOT:-${DEFAULT_LEROBOT_ROOT}}"
LEROBOT_REV=8fff0fde7c79f23a93d845d1a50e985de01f8b8a

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "[KinRT][ERROR] This environment requires Linux and CUDA 12; native Windows JAX CUDA is unsupported." >&2
  exit 1
fi
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" ]]; then
  for candidate in "${HOME}/.local/bin/uv" "${HOME}/.cargo/bin/uv"; do
    if [[ -x "${candidate}" ]]; then
      UV_BIN="${candidate}"
      break
    fi
  done
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "[KinRT][ERROR] uv is required. Install it from https://docs.astral.sh/uv/." >&2
  exit 1
fi

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
if [[ "$(git -C "${SOURCE_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)" != "${SOURCE_ROOT}" ]]; then
  echo "[KinRT][ERROR] Cannot verify the KinRT checkout root: ${SOURCE_ROOT}" >&2
  exit 1
fi
actual_rev="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
if [[ "${actual_rev}" != "${KINRT_REV}" && "${KINRT_ALLOW_UNPINNED_SOURCE:-0}" != "1" ]]; then
  echo "[KinRT][ERROR] Source revision mismatch: ${actual_rev}" >&2
  echo "[KinRT][ERROR] Expected: ${KINRT_REV}" >&2
  echo "[KinRT][ERROR] Set KINRT_ALLOW_UNPINNED_SOURCE=1 only for development." >&2
  exit 1
fi

if [[ ! -f "${LEROBOT_ROOT}/src/lerobot/datasets/lerobot_dataset.py" ]]; then
  if [[ "${LEROBOT_ROOT}" != "${DEFAULT_LEROBOT_ROOT}" || -e "${LEROBOT_ROOT}" ]]; then
    echo "[KinRT][ERROR] Supply an existing pinned LeRobot checkout: ${LEROBOT_ROOT}" >&2
    exit 1
  fi
  git clone https://github.com/huggingface/lerobot.git "${LEROBOT_ROOT}"
  git -C "${LEROBOT_ROOT}" checkout --detach "${LEROBOT_REV}"
fi
LEROBOT_ROOT="$(cd "${LEROBOT_ROOT}" && pwd)"
if [[ "$(git -C "${LEROBOT_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)" != "${LEROBOT_ROOT}" ]] || \
   [[ "$(git -C "${LEROBOT_ROOT}" rev-parse HEAD 2>/dev/null || true)" != "${LEROBOT_REV}" ]]; then
  echo "[KinRT][ERROR] Expected LeRobot ${LEROBOT_REV} at ${LEROBOT_ROOT}." >&2
  exit 1
fi
if [[ -n "$(git -C "${LEROBOT_ROOT}" status --porcelain --untracked-files=all -- src/lerobot)" ]]; then
  echo "[KinRT][ERROR] The LeRobot source has local changes; supply a clean pinned checkout." >&2
  exit 1
fi

cd "${OPENPI_ROOT}"
OPENPI_ROOT="$(pwd)"
PYTHON_BIN="${OPENPI_ROOT}/.venv/bin/python"
if [[ ! -e "${OPENPI_ROOT}/.venv" ]]; then
  "${UV_BIN}" venv --python 3.11 "${OPENPI_ROOT}/.venv"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[KinRT][ERROR] Existing .venv has no Linux Python: ${OPENPI_ROOT}/.venv" >&2
  exit 1
fi
"${PYTHON_BIN}" -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11 or newer is required'"
"${PYTHON_BIN}" "${POLICY_DIR}/prepare_full35_source.py" "${OPENPI_ROOT}" --check
UV_PROJECT_ENVIRONMENT="${OPENPI_ROOT}/.venv" UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1 \
  "${PYTHON_BIN}" "${POLICY_DIR}/sync_full35_dependencies.py" \
    "${OPENPI_ROOT}" --uv "${UV_BIN}" --mirror "${KINRT_PYPI_MIRROR:-pypi}"
"${PYTHON_BIN}" "${POLICY_DIR}/prepare_full35_source.py" "${OPENPI_ROOT}"
"${UV_BIN}" pip uninstall --python "${PYTHON_BIN}" opencv-python
# The v3 source overlay imports Accelerate; the old runtime lock omits it and psutil.
# Upstream model imports also require pytest from the otherwise disabled dev group.
# Reinstall headless: uninstalling the GUI wheel removes their shared cv2 files.
"${UV_BIN}" pip install --python "${PYTHON_BIN}" --reinstall-package opencv-python-headless \
  opencv-python-headless==4.11.0.86 scikit-learn==1.8.0 joblib==1.5.3 \
  accelerate==1.10.1 psutil==7.2.2 msgpack-numpy==0.4.8 pytest==9.0.3
"${UV_BIN}" pip install --python "${PYTHON_BIN}" -e "${XPL_ROOT}"
"${PYTHON_BIN}" "${POLICY_DIR}/prepare_tokenizer.py"

# The delivered run used this v3 source overlay with the older locked dependencies.
"${PYTHON_BIN}" - "${LEROBOT_ROOT}" <<'PYOVERLAY'
from pathlib import Path
import sys
import sysconfig

source = (Path(sys.argv[1]) / "src").resolve()
overlay = Path(sysconfig.get_path("purelib")) / "kinrt_full35_lerobot.pth"
overlay.write_text(f"import sys; sys.path.insert(0, {str(source)!r})\n", encoding="utf-8")
print(f"[KinRT] LeRobot source overlay: {source}")
PYOVERLAY

"${PYTHON_BIN}" - "${LEROBOT_ROOT}" "${XPL_ROOT}" <<'PYVERIFY'
from pathlib import Path
import sys

import XPolicyLab
import XPolicyLab.policy.KinRT.model as kinrt_model
import cv2
import openpi
import lerobot.datasets.lerobot_dataset as lerobot_dataset

expected = (Path(sys.argv[1]) / "src").resolve()
actual = Path(lerobot_dataset.__file__).resolve()
if expected not in actual.parents or lerobot_dataset.CODEBASE_VERSION != "v3.0":
    raise RuntimeError(f"Unexpected LeRobot dataset implementation: {actual}")
expected_adapter = (Path(sys.argv[2]) / "policy/KinRT/model.py").resolve()
actual_adapter = Path(kinrt_model.__file__).resolve()
if actual_adapter != expected_adapter:
    raise RuntimeError(f"Unexpected KinRT adapter: {actual_adapter}")
print(f"[KinRT] Policy imports ready; LeRobot v3.0: {actual}")
print(f"[KinRT] KinRT adapter imports ready: {actual_adapter}")
print(f"[KinRT] OpenCV ready: {cv2.__version__} ({cv2.__file__})")
print("[KinRT] Model loading and GPU inference have not been tested by this installer.")
PYVERIFY
