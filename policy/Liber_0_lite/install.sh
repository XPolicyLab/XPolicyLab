#!/usr/bin/env bash
set -euo pipefail
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
policy_env=${1:?Pass a Python 3.10 environment prefix}
"$policy_env/bin/python" -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
"$policy_env/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt" -e "$XPL_ROOT"
"$policy_env/bin/python" - <<'PY'
import os, sys, torch
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False
print(sys.executable, torch.__version__, torch.version.cuda)
assert os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] == '0'
assert torch.get_float32_matmul_precision() == 'highest'
assert not torch.backends.cuda.matmul.allow_tf32
assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 11)
import omegaconf, transformers, torchvision, accelerate
PY
