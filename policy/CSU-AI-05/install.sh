#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
G05_ROOT="${G05_ROOT:-${SCRIPT_DIR}/G05}"

PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Set G05_PYTHON to a valid Python executable." >&2
  exit 2
fi

echo "[G05 install] python=${PYTHON_BIN}"
echo "[G05 install] xpolicylab=${XPL_ROOT}"
echo "[G05 install] g05_root=${G05_ROOT}"

if [[ ! -f "${G05_ROOT}/pyproject.toml" ]]; then
  echo "G05 source checkout not found under ${G05_ROOT}." >&2
  exit 3
fi

"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"
"${PYTHON_BIN}" -m pip install -e "${G05_ROOT}"

cat <<'EOF'

CSU-AI-05 and its evaluated G05 runtime are installed.

Before running evaluation, set:

  export G05_PYTHON=/path/to/python
  export G05_CKPT_PATH=/path/to/CSU-AI-05/checkpoints/CSU_AI_05.pt

G05_ROOT defaults to policy/CSU-AI-05/G05. Override it only when intentionally
testing another compatible checkout.
EOF
