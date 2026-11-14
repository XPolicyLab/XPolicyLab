#!/bin/bash
set -euo pipefail

ENV_NAME=${1:-awomo06i}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT=${AWOMO06I_RUNTIME_ROOT:?Set AWOMO06I_RUNTIME_ROOT to the historical runtime root}
[[ -f "${XPL_ROOT}/setup_policy_server.py" ]] || { echo "Copy policy/Awomo06I into an XPolicyLab checkout before installing" >&2; exit 1; }
[[ -f "${RUNTIME_ROOT}/Automodel/pyproject.toml" ]] || { echo "Missing historical Automodel runtime" >&2; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.11 -y
fi
conda activate "${ENV_NAME}"
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'numpy==1.26.4' 'huggingface_hub[hf_xet]'
python -m pip install -e "${XPL_ROOT}"
python -m pip install --no-deps -e "${RUNTIME_ROOT}/Automodel"

python - <<'PY'
import importlib.util
required = ('torch', 'torchvision', 'numpy', 'transformers', 'diffusers', 'safetensors', 'omegaconf', 'nemo_automodel')
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit('Install missing dependencies from the matching historical runtime environment: ' + ', '.join(missing))
print('[Awomo06I] Basic imports are present; real-checkpoint startup has not been validated.')
PY
