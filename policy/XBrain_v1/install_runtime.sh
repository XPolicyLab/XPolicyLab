#!/usr/bin/env bash
set -euo pipefail

# Optional local-development helper for a conda-pack archive. This is not part
# of the official GitHub installation path. No private/default author path is
# embedded here: callers must provide both the archive and destination.
ARCHIVE_PATH="${XBRAIN_RUNTIME_ARCHIVE:-}"
RUNTIME_ROOT="${XBRAIN_RUNTIME_ROOT:-}"
ENV_NAME="${XBRAIN_RUNTIME_ENV_NAME:-xbrain_v1_runtime}"
ENV_ROOT="${RUNTIME_ROOT}/${ENV_NAME}"

if [[ -z "${ARCHIVE_PATH}" || -z "${RUNTIME_ROOT}" ]]; then
  echo "[XBrain_v1][ERROR] optional archive install requires:" >&2
  echo "  XBRAIN_RUNTIME_ARCHIVE=/path/to/runtime.zip" >&2
  echo "  XBRAIN_RUNTIME_ROOT=/path/to/runtime-root" >&2
  echo "For the official installation, use policy/XBrain_v1/install.sh (bundled inference runtime)." >&2
  exit 2
fi
if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "[XBrain_v1][ERROR] runtime archive not found: ${ARCHIVE_PATH}" >&2
  exit 1
fi

mkdir -p "${RUNTIME_ROOT}"
if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  echo "[XBrain_v1] expanding ${ARCHIVE_PATH} into ${RUNTIME_ROOT}"
  unzip -q -n "${ARCHIVE_PATH}" -d "${RUNTIME_ROOT}"
fi

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  echo "[XBrain_v1][ERROR] archive did not produce ${ENV_ROOT}/bin/python" >&2
  exit 1
fi

if [[ -x "${ENV_ROOT}/bin/conda-unpack" ]]; then
  "${ENV_ROOT}/bin/conda-unpack" || true
fi

if [[ "${XBRAIN_INSTALL_PY_DEPS:-0}" == "1" ]]; then
  echo "[XBrain_v1] installing inference dependencies"
  "${ENV_ROOT}/bin/python" -m pip install -r "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-inference.txt"
fi

"${ENV_ROOT}/bin/python" - <<'PY'
import importlib
import sys

print("python:", sys.executable)
for name in ("numpy", "torch", "msgpack", "cv2", "yaml"):
    module = importlib.import_module(name)
    print(f"{name}: OK", getattr(module, "__version__", ""))
import torch
print("cuda_available:", torch.cuda.is_available())
PY

echo "[XBrain_v1] runtime ready: ${ENV_ROOT}"
