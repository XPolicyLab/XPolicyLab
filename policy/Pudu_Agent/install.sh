#!/usr/bin/env bash
# Pudu_Agent holds no VLA checkpoint. Install the client-side
# provider dependencies into the Python that runs the RoboDojo client.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "[INSTALL][ERROR] usage: bash install.sh /path/to/python" >&2
    exit 1
fi

python_bin=$1
if [[ ! -x "${python_bin}" ]]; then
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        echo "[INSTALL][ERROR] Python not found: ${python_bin}" >&2
        exit 1
    fi
    python_bin="$(command -v "${python_bin}")"
fi

if command -v uv >/dev/null 2>&1; then
  uv pip install --python "${python_bin}" "openai==3.8.0" "pillow==12.3.0"
else
  "${python_bin}" -m pip install "openai==3.8.0" "pillow==12.3.0"
fi

echo "[INSTALL] Pudu_Agent uses openai==3.8.0 and pillow==12.3.0 in the eval client."
echo "[INSTALL] Set ARK_API_KEY before evaluating. L4_INSPECT_PLANNER defaults to astra (gpt-6-astra)."
