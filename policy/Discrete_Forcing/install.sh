#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
model_root="${DF_ROOT:-${SCRIPT_DIR}/source_discrete_forcing}"
revision=acce1e9f3ec81fb345a40083f64a0031146f4398

if [[ ! -e "$model_root" ]]; then
    git clone https://github.com/Jbo-Wang/discrete_forcing.git "$model_root"
    git -C "$model_root" checkout "$revision"
fi
test -f "$model_root/starVLA/model/framework/QwenPILF_v3.py"
python -m pip install pip==23.0.1 setuptools==80.9.0 wheel ninja packaging
python -m pip install -r "$model_root/requirements-train.txt"
python -m pip install --no-build-isolation flash-attn==2.7.4.post1 causal-conv1d==1.6.1
python -m pip install --no-deps -e "$model_root"
python -m pip install -e "$XPL_ROOT"
