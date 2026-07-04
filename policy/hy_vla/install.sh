#!/usr/bin/env bash
set -euo pipefail

# Installs the hy_vla policy environment.
#
# The policy server runs inside the Hy-Embodied uv venv (torch 2.7 + the
# HunYuanVLMoT transformers fork + flash_attn). This script materializes that
# venv and installs XPolicyLab into it (editable) so the server can import both
# the `hy_vla`/`robotwin_eval` model code and the `XPolicyLab` runtime.

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
HY_VLA_ROOT="${HY_VLA_ROOT:-${POLICY_DIR}/Hy-Embodied-0.5-VLA}"
HY_VLA_REPO="https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

if [[ ! -d "${HY_VLA_ROOT}" ]]; then
  echo "[hy_vla] cloning Hy-Embodied source into ${HY_VLA_ROOT}"
  git clone "${HY_VLA_REPO}" "${HY_VLA_ROOT}"
fi

cd "${HY_VLA_ROOT}"
echo "[hy_vla] uv sync in ${HY_VLA_ROOT}"
UV_LINK_MODE=copy uv sync

# Overlay RoboDojo post-training support onto the public Hy-Embodied clone.
# The public repo does not ship the RoboDojo dataset loader / config / scripts;
# this copies them in and wires the `robodojo` dataset branch (idempotent).
echo "[hy_vla] overlaying RoboDojo post-training support"
uv run python "${POLICY_DIR}/apply_robodojo_overlay.py" "${HY_VLA_ROOT}"

# Make XPolicyLab importable inside the Hy-Embodied venv.
uv pip install -e "${XPOLICYLAB_ROOT}"
uv run python -c "import XPolicyLab; import hy_vla; import robotwin_eval; from hy_vla.data.robodojo_dataset import RoboDojoVLADataset; print('hy_vla env ok (robodojo overlay ok)')"

echo "[hy_vla] Installation finished."
echo "[hy_vla] Activate: source ${HY_VLA_ROOT}/.venv/bin/activate"
echo "[hy_vla] Download a checkpoint (e.g. into ${HY_VLA_ROOT}) before eval; see deploy.yml ckpt_path."
