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

"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"

# The real-evaluation runtime ships checkpoint-compatible tokenizer and
# dataset packages under third_party. Install them before G05 so pip never
# substitutes an incompatible public package with the same name.
for runtime_dep in galaxea_dataset galaxea_tokenizer; do
  dep_root="${G05_ROOT}/third_party/${runtime_dep}"
  if [[ -f "${dep_root}/pyproject.toml" ]]; then
    "${PYTHON_BIN}" -m pip install -e "${dep_root}"
  fi
done

if [[ -f "${G05_ROOT}/pyproject.toml" ]]; then
  "${PYTHON_BIN}" -m pip install -e "${G05_ROOT}"
fi

cat <<'EOF'

G05 adapter-side dependencies are installed.

Before running evaluation, set:

  export G05_PYTHON=/path/to/python
  export G05_CKPT_PATH=/path/to/extracted/g05/checkpoint

The vendored G05 checkout under policy/G05/G05 is used by default. Override
G05_ROOT only if you want to use another compatible G05 checkout.
EOF
