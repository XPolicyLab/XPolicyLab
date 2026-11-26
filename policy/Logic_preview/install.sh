#!/usr/bin/env bash
set -euo pipefail
policy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin=${PYTHON_BIN:-python}
"$python_bin" - <<'PY'
import sys
import torch
import torchvision

print(f"Policy runtime: {sys.executable} (Python {sys.version.split()[0]}), "
      f"torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA {torch.version.cuda}",
      flush=True)
if not torch.cuda.is_available():
    raise SystemExit("The policy needs a visible NVIDIA GPU. Check the job's GPU allocation, driver and CUDA-enabled PyTorch build.")
PY
"$python_bin" -m pip install -e "$policy_dir/model_code" -e "$policy_dir/../.."
"$python_bin" -m pip check
