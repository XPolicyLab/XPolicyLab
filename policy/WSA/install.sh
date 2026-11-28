#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WSA_DIR="${SCRIPT_DIR}/WSA"

python -m pip install "transformers==4.57.1"
python -m pip install -e "${WSA_DIR}"
python -m pip install -e "${XPL_ROOT}"
python -m pip install \
    tyro matplotlib mediapy scipy loguru omegaconf rich pytest \
    "websockets>=14.0" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8"

# WSA-Base relies on the upstream cached-inference Qwen3-VL patch.
transformers_dir="$(python -c 'from pathlib import Path; import transformers; print(Path(transformers.__file__).resolve().parent)')"
cp -r "${WSA_DIR}/src/lerobot/policies/WSA_Base/transformers_replace/models/." "${transformers_dir}/models/"

echo "[WSA] installed WSA, XPolicyLab, websocket dependencies, and the Qwen3-VL patch."
