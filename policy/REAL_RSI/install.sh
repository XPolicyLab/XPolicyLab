#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -c "${SCRIPT_DIR}/constraints.txt" torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c "${SCRIPT_DIR}/constraints.txt" -r "${SCRIPT_DIR}/requirements.txt"
python -m pip install -c "${SCRIPT_DIR}/constraints.txt" -e "${SCRIPT_DIR}/../.."
python -m pip check
python - <<'PY'
from sam3.model_builder import build_sam3_image_model
import pyrualean.robots

pyrualean.robots.get_adapter("robodojo_xpl_rgb").host_module()
print("PyRUA-Lean and SAM3 import")
PY
echo "Next: python ${SCRIPT_DIR}/download_models.py"
